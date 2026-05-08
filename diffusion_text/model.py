import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, d_model, max_period=10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, t):
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.ln1(x) * (1 + scale_msa) + shift_msa
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * self.dropout(h)
        
        h = self.ln2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ff(h)
        return x


class D3PMTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12,
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalTimestepEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "adaLN_modulation" not in name:
                # Use a more conservative initialization standard for Transformers
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "ln" in name or "LayerNorm" in name:
                if "weight" in name:
                    nn.init.ones_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
                    
        # Zero-out adaLN modulation terminal layer for skip connection stability (identity function at init)
        for m in self.modules():
            if isinstance(m, TransformerBlock):
                nn.init.zeros_(m.adaLN_modulation[-1].weight)
                nn.init.zeros_(m.adaLN_modulation[-1].bias)

    def forward(self, x, t):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)

        h = self.token_emb(x) + self.pos_emb(positions)
        t_emb = self.time_proj(self.time_emb(t))
        
        # We can still add time embedding globally, but DiT mostly relies on AdaLN per layer.
        h = h + t_emb.unsqueeze(1)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
