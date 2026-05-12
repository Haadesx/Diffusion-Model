import math
import torch
import torch.nn.functional as F

def mask_schedule(t, T, schedule="cosine"):
    ratio = t / T
    if schedule == "linear":
        return ratio
    elif schedule == "cosine":
        return 1.0 - math.cos(ratio * math.pi / 2)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

def mask_schedule_tensor(t, T, schedule="cosine"):
    ratio = t.float() / T
    if schedule == "linear":
        return ratio
    elif schedule == "cosine":
        return 1.0 - torch.cos(ratio * math.pi / 2)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

def forward_corrupt(x0, t, T, mask_id, schedule="cosine", pad_id=None, ensure_min_mask=True):
    p_mask = mask_schedule_tensor(t, T, schedule)
    mask_probs = p_mask.unsqueeze(1).expand_as(x0)
    rand = torch.rand(x0.shape, device=x0.device)
    mask = rand < mask_probs.float()
    valid_positions = torch.ones_like(mask, dtype=torch.bool)
    if pad_id is not None:
        valid_positions = x0 != pad_id
        mask = mask & valid_positions
    if ensure_min_mask:
        empty_rows = (mask.sum(dim=1) == 0) & valid_positions.any(dim=1)
        if empty_rows.any():
            scores = torch.rand(x0.shape, device=x0.device).masked_fill(~valid_positions, -1.0)
            fallback_idx = scores.argmax(dim=1)
            rows = torch.arange(x0.size(0), device=x0.device)[empty_rows]
            mask[rows, fallback_idx[empty_rows]] = True
    xt = x0.clone()
    xt[mask] = mask_id
    return xt, mask

def compute_loss(model, x0, t, T, mask_id, pad_id=0,
                 schedule="cosine", loss_weight_masked=2.0,
                 loss_mode="weighted"):
    xt, mask = forward_corrupt(x0, t, T, mask_id, schedule, pad_id=pad_id)
    logits = model(xt, t)

    B, L, V = logits.shape
    loss_per_pos = F.cross_entropy(
        logits.reshape(-1, V), x0.reshape(-1), reduction="none"
    ).reshape(B, L)

    pad_mask = (x0 != pad_id).float()
    if loss_mode == "masked_only":
        weights = mask.float() * pad_mask
    elif loss_mode == "weighted":
        weights = torch.ones_like(loss_per_pos)
        weights[mask] = loss_weight_masked
        weights = weights * pad_mask
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    total_weight = weights.sum()
    if total_weight > 0:
        loss = (loss_per_pos * weights).sum() / total_weight
    else:
        loss = loss_per_pos.mean()

    return loss, logits, xt, mask

@torch.no_grad()
def sample(model, length, T, mask_id, pad_id, device,
           schedule="cosine", num_samples=1, top_k=50,
           temperature=1.0, prefix_ids=None):
    model.eval()

    x = torch.full((num_samples, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(num_samples, length, dtype=torch.bool, device=device)

    if prefix_ids is not None:
        prefix_len = min(len(prefix_ids), length)
        prefix_t = torch.tensor(prefix_ids[:prefix_len], dtype=torch.long, device=device)
        x[:, :prefix_len] = prefix_t
        fixed[:, :prefix_len] = True

    for step in range(T, 0, -1):
        t_batch = torch.full((num_samples,), step, dtype=torch.long, device=device)
        logits = model(x, t_batch)

        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            topk_vals, _ = logits.topk(min(top_k, logits.size(-1)), dim=-1)
            threshold = topk_vals[:, :, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = F.softmax(logits, dim=-1)

        flat_probs = probs.reshape(-1, probs.shape[-1])
        proposed = torch.multinomial(flat_probs, 1).reshape(num_samples, length)

        confidence, _ = probs.max(dim=-1)

        frac_to_fix = 1.0 - mask_schedule(step - 1, T, schedule)
        total_to_fix = max(1, int(frac_to_fix * length))

        for b in range(num_samples):
            unfixed_count = (~fixed[b]).sum().item()
            if unfixed_count == 0:
                continue

            conf = confidence[b].clone()
            conf[fixed[b]] = -1.0

            n_to_fix_now = max(0, total_to_fix - fixed[b].sum().item())
            n_to_fix_now = min(n_to_fix_now, unfixed_count)

            if n_to_fix_now > 0:
                _, top_idx = conf.topk(n_to_fix_now)
                x[b, top_idx] = proposed[b, top_idx]
                fixed[b, top_idx] = True

    still_masked = x == mask_id
    if still_masked.any():
        t_one = torch.ones(num_samples, dtype=torch.long, device=device)
        logits = model(x, t_one)
        logits = logits / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        proposed = torch.multinomial(flat_probs, 1).reshape(num_samples, length)
        x[still_masked] = proposed[still_masked]

    model.train()
    return x
