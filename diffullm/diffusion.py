import math
import torch
import torch.nn.functional as F


def mask_prob(t, T, schedule="cosine"):
    r = t / T
    if schedule == "linear":
        return r
    elif schedule == "cosine":
        return 1.0 - math.cos(r * math.pi / 2)
    raise ValueError(f"unknown schedule: {schedule}")


def mask_prob_batch(t, T, schedule="cosine"):
    r = t.float() / T
    if schedule == "linear":
        return r
    elif schedule == "cosine":
        return 1.0 - torch.cos(r * math.pi / 2)
    raise ValueError(f"unknown schedule: {schedule}")


def corrupt(x0, t, T, mask_id, schedule="cosine", pad_id=None, force_at_least_one=True):
    p = mask_prob_batch(t, T, schedule)
    probs = p.unsqueeze(1).expand_as(x0)
    noise = torch.rand(x0.shape, device=x0.device)
    masked = noise < probs.float()

    valid = torch.ones_like(masked, dtype=torch.bool)
    if pad_id is not None:
        valid = x0 != pad_id
        masked = masked & valid

    if force_at_least_one:
        empty = (masked.sum(dim=1) == 0) & valid.any(dim=1)
        if empty.any():
            scores = torch.rand(x0.shape, device=x0.device).masked_fill(~valid, -1.0)
            pick = scores.argmax(dim=1)
            rows = torch.arange(x0.size(0), device=x0.device)[empty]
            masked[rows, pick[empty]] = True

    xt = x0.clone()
    xt[masked] = mask_id
    return xt, masked


def compute_loss(model, x0, t, T, mask_id, pad_id=0,
                 schedule="cosine", loss_weight_masked=2.0, loss_mode="weighted"):
    xt, masked = corrupt(x0, t, T, mask_id, schedule, pad_id=pad_id)
    logits = model(xt, t)

    B, L, V = logits.shape
    per_tok = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1), reduction="none").reshape(B, L)
    not_pad = (x0 != pad_id).float()

    if loss_mode == "masked_only":
        w = masked.float() * not_pad
    elif loss_mode == "weighted":
        w = torch.ones_like(per_tok)
        w[masked] = loss_weight_masked
        w = w * not_pad
    else:
        raise ValueError(f"unknown loss_mode: {loss_mode}")

    denom = w.sum()
    loss = (per_tok * w).sum() / denom if denom > 0 else per_tok.mean()
    return loss, logits, xt, masked


@torch.no_grad()
def sample(model, length, T, mask_id, pad_id, device,
           schedule="cosine", num_samples=1, top_k=50,
           temperature=1.0, prefix_ids=None):
    model.eval()

    x = torch.full((num_samples, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(num_samples, length, dtype=torch.bool, device=device)

    if prefix_ids is not None:
        plen = min(len(prefix_ids), length)
        p = torch.tensor(prefix_ids[:plen], dtype=torch.long, device=device)
        x[:, :plen] = p
        fixed[:, :plen] = True

    for step in range(T, 0, -1):
        t_batch = torch.full((num_samples,), step, dtype=torch.long, device=device)
        logits = model(x, t_batch) / max(temperature, 1e-8)

        if top_k > 0:
            vals, _ = logits.topk(min(top_k, logits.size(-1)), dim=-1)
            logits = logits.masked_fill(logits < vals[:, :, -1:], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        proposed = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(num_samples, length)
        confidence, _ = probs.max(dim=-1)

        n_fixed = max(1, int((1.0 - mask_prob(step - 1, T, schedule)) * length))

        for b in range(num_samples):
            conf = confidence[b].clone()
            conf[fixed[b]] = -1.0
            unfixed = (~fixed[b]).sum().item()
            if unfixed == 0:
                continue
            n_now = min(max(0, n_fixed - fixed[b].sum().item()), unfixed)
            if n_now > 0:
                _, idx = conf.topk(n_now)
                x[b, idx] = proposed[b, idx]
                fixed[b, idx] = True

    # mop up any remaining masks
    still_masked = x == mask_id
    if still_masked.any():
        t1 = torch.ones(num_samples, dtype=torch.long, device=device)
        logits = model(x, t1) / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        proposed = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(num_samples, length)
        x[still_masked] = proposed[still_masked]

    model.train()
    return x
