import os
import json
import math
import time
import logging

import torch

from diffusion_text.model import D3PMTransformer
from diffusion_text.diffusion import compute_loss
from diffusion_text.data import create_dataloader
from diffusion_text.utils import (
    get_device, set_seed, get_git_hash, save_json, timestamp,
    make_run_dir, setup_logging, update_global_registry,
)
from diffusion_text.progress import TrainingDashboard, console, print_success, print_info

logger = logging.getLogger("diffusion_text")


def build_model(config, vocab_size, device):
    mc = config["model"]
    seq_len = config["tokenization"]["seq_len"]
    model = D3PMTransformer(
        vocab_size=vocab_size,
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        d_ff=mc["d_ff"],
        max_seq_len=seq_len,
        dropout=mc["dropout"],
    )
    return model.to(device)


def build_optimizer(model, config):
    tc = config["train"]
    return torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                             weight_decay=tc["weight_decay"], betas=(0.9, 0.999))


def lr_schedule(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def sample_timesteps(batch_size, T, device, mode="uniform"):
    if mode == "uniform":
        return torch.randint(1, T + 1, (batch_size,), device=device)
    if mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
        return (u * (T - 1)).long() + 1
    if mode == "sqrt":
        u = torch.rand(batch_size, device=device)
        return (u.sqrt() * (T - 1)).long() + 1
    raise ValueError(f"unknown timestep mode: {mode}")


def try_amp(device):
    if device.type == "cuda":
        try:
            return (True, torch.bfloat16) if torch.cuda.is_bf16_supported() else (True, torch.float16)
        except Exception:
            return False, None
    if device.type == "mps":
        dummy = torch.randn(2, 2, device=device)
        for dt in [torch.bfloat16, torch.float16]:
            try:
                with torch.autocast(device_type="mps", dtype=dt):
                    _ = dummy @ dummy
                return True, dt
            except Exception:
                continue
    return False, None


def save_checkpoint(model, optimizer, step, run_dir, name="checkpoint"):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    path = os.path.join(run_dir, "checkpoints", f"{name}_step{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"saved: {path}")
    update_global_registry(os.path.dirname(run_dir), path, step=step)
    return path


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


@torch.no_grad()
def evaluate(model, val_loader, config, device, mask_id, pad_id, num_batches=None):
    model.eval()
    dc = config["diffusion"]
    T = dc["T"]
    total_loss, total_correct, total_masked, n = 0.0, 0, 0, 0

    for batch in val_loader:
        if num_batches and n >= num_batches:
            break
        x0 = batch.to(device)
        t = sample_timesteps(x0.shape[0], T, device, dc.get("timestep_sampling", "uniform"))
        loss, logits, xt, mask = compute_loss(
            model, x0, t, T, mask_id, pad_id,
            dc["schedule"], dc.get("loss_weight_masked", 2.0), dc.get("loss_mode", "weighted")
        )
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        valid = x0 != pad_id
        total_correct += ((preds == x0) & mask & valid).sum().item()
        total_masked += (mask & valid).sum().item()
        n += 1

    model.train()
    if n == 0:
        return {"val_loss": 0.0, "recon_accuracy": 0.0}
    return {
        "val_loss": total_loss / n,
        "recon_accuracy": total_correct / max(1, total_masked),
        "num_batches": n,
    }


def train(config, resume_checkpoint=None):
    tc = config["train"]
    dc = config["diffusion"]
    data_dir = config["paths"]["data_dir"]
    runs_dir = config["paths"]["runs_dir"]

    set_seed(tc["seed"])
    device = get_device()
    run_dir = make_run_dir(runs_dir, config.get("run", {}).get("name"))
    log = setup_logging(os.path.join(run_dir, "log.txt"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    with open(os.path.join(data_dir, "tokenizer", "tokenizer_meta.json")) as f:
        tok_meta = json.load(f)
    vocab_size = tok_meta["vocab_size"]
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]

    model = build_model(config, vocab_size, device)
    log.info(f"params: {model.count_parameters():,}  device: {device}")

    optimizer = build_optimizer(model, config)
    start_step = 0
    if resume_checkpoint:
        start_step = load_checkpoint(resume_checkpoint, model, optimizer, device)
        log.info(f"resumed from step {start_step}")

    use_amp, amp_dtype = try_amp(device)

    train_loader = create_dataloader(data_dir, split="train", batch_size=tc["batch_size"], shuffle=True)
    val_loader = create_dataloader(data_dir, split="val", batch_size=tc["batch_size"], shuffle=False)

    if len(train_loader.dataset) < tc["batch_size"]:
        raise ValueError("train split smaller than batch_size")
    if len(val_loader.dataset) < 1:
        raise ValueError("val split is empty")

    T = dc["T"]
    schedule = dc["schedule"]
    loss_weight = dc.get("loss_weight_masked", 2.0)
    loss_mode = dc.get("loss_mode", "weighted")
    t_mode = dc.get("timestep_sampling", "uniform")
    grad_accum = tc["grad_accum"]
    max_steps = tc["max_steps"]
    clip_norm = tc["clip_grad_norm"]
    explosion_thresh = tc.get("grad_norm_explosion_threshold", 100.0)

    save_json({
        "config": config, "device": str(device),
        "model_params": model.count_parameters(), "vocab_size": vocab_size,
        "git_hash": get_git_hash(), "amp_enabled": use_amp, "start_time": timestamp(),
    }, os.path.join(run_dir, "run_manifest.json"))

    model.train()
    optimizer.zero_grad()
    step = start_step
    best_val = float("inf")
    running_loss = 0.0
    skipped = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    dashboard = TrainingDashboard(
        max_steps=max_steps, device_name=str(device).upper(),
        amp_enabled=use_amp, model_params=model.count_parameters(),
        run_dir=run_dir, start_step=start_step,
    )

    with dashboard:
        while step < max_steps:
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x0 = batch.to(device)
                t = sample_timesteps(x0.shape[0], T, device, t_mode)

                if use_amp:
                    try:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            loss, *_ = compute_loss(model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode)
                        (loss / grad_accum).backward()
                    except Exception as e:
                        use_amp = False
                        dashboard.amp_enabled = False
                        log.warning(f"amp failed ({e}), falling back")
                        loss, *_ = compute_loss(model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode)
                        (loss / grad_accum).backward()
                else:
                    loss, *_ = compute_loss(model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode)
                    (loss / grad_accum).backward()

                running_loss += loss.item()

            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm if clip_norm > 0 else float("inf")
            )
            gnorm = gnorm.item() if torch.is_tensor(gnorm) else float(gnorm)

            if gnorm > explosion_thresh:
                log.warning(f"step {step+1}: grad norm {gnorm:.1f} > threshold, skipping")
                skipped += 1
                optimizer.zero_grad()
                step += 1
                dashboard.update(step, grad_norm=gnorm)
                continue

            current_lr = lr_schedule(step, tc["warmup_steps"], max_steps, tc["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                avg_loss = running_loss / 10
                dashboard.update(step, loss=avg_loss, lr=current_lr, grad_norm=gnorm)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "train_loss": avg_loss,
                        "lr": current_lr, "grad_norm": gnorm,
                        "elapsed_s": time.time() - t0,
                    }) + "\n")
                running_loss = 0.0
            else:
                dashboard.update(step)

            if step % tc["eval_every"] == 0:
                val = evaluate(model, val_loader, config, device, mask_id, pad_id,
                               num_batches=min(config["eval"].get("num_batches", 50), 50))
                log.info(f"step {step}: val_loss={val['val_loss']:.4f}  recon_acc={val['recon_accuracy']:.4f}")
                dashboard.update_val(val["val_loss"], val["recon_accuracy"])
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, **val}) + "\n")
                if val["val_loss"] < best_val:
                    best_val = val["val_loss"]
                    path = save_checkpoint(model, optimizer, step, run_dir, name="best")
                    update_global_registry(os.path.dirname(run_dir), path, val_loss=best_val, step=step)

            if step % tc["save_every"] == 0:
                save_checkpoint(model, optimizer, step, run_dir)

    save_checkpoint(model, optimizer, step, run_dir, name="final")

    save_json({
        "config": config, "end_time": timestamp(), "final_step": step,
        "best_val_loss": best_val, "skipped_steps": skipped,
    }, os.path.join(run_dir, "run_manifest.json"))

    console.print()
    print_success(f"done at step {step:,}")
    print_info(f"best val loss: {best_val:.4f}")
    print_info(f"run dir: {run_dir}")
    console.print()

    return run_dir
