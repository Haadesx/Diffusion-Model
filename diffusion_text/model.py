import math
import torch
import torch.nn as nn


class TimestepEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        x = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # DiT-style adaptive layernorm conditioning on timestep
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, c):
        s1, g1, gt1, s2, g2, gt2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1) + s1
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gt1 * self.drop(h)
        h = self.norm2(x) * (1 + g2) + s2
        x = x + gt2 * self.ff(h)
        return x


class D3PMTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12,
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.t_emb = TimestepEmb(d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "ada" not in name:
                nn.init.normal_(p, std=0.02)
            elif "norm" in name or "LayerNorm" in name:
                if "weight" in name:
                    nn.init.ones_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

        for m in self.modules():
            if isinstance(m, Block):
                nn.init.zeros_(m.ada[-1].weight)
                nn.init.zeros_(m.ada[-1].bias)

    def forward(self, x, t):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        c = self.t_proj(self.t_emb(t))
        h = self.drop(h + c.unsqueeze(1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
