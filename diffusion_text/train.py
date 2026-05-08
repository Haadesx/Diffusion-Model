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
    make_run_dir, setup_logging, sha256_file, update_global_registry,
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
    return torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.999),
    )


def lr_schedule(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_timesteps(batch_size, T, device, mode="uniform"):
    """Sample diffusion timesteps for a training or validation batch."""
    if mode == "uniform":
        return torch.randint(1, T + 1, (batch_size,), device=device)
    if mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
        return (u * (T - 1)).long() + 1
    if mode == "sqrt":
        u = torch.rand(batch_size, device=device)
        return (u.sqrt() * (T - 1)).long() + 1
    raise ValueError(f"Unknown timestep_sampling mode: {mode}")


def try_amp(device):
    if device.type == "cuda":
        try:
            # Check for bfloat16 first (more stable)
            if torch.cuda.is_bf16_supported():
                return True, torch.bfloat16
            return True, torch.float16
        except Exception:
            return False, None
    if device.type == "mps":
        try:
            # MPS excels with bfloat16 stability
            dummy = torch.randn(2, 2, device=device)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                _ = dummy @ dummy
            return True, torch.bfloat16
        except Exception:
            try:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    _ = dummy @ dummy
                return True, torch.float16
            except Exception:
                return False, None
    return False, None


def save_checkpoint(model, optimizer, step, run_dir, name="checkpoint"):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}_step{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logger.info(f"Saved checkpoint: {path}")

    # Update global latest checkpoint
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
    schedule = dc["schedule"]
    loss_weight = dc.get("loss_weight_masked", 2.0)
    loss_mode = dc.get("loss_mode", "weighted")
    timestep_sampling = dc.get("timestep_sampling", "uniform")

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        if num_batches and n_batches >= num_batches:
            break

        x0 = batch.to(device)
        B, L = x0.shape
        t = sample_timesteps(B, T, device, timestep_sampling)

        loss, logits, xt, mask = compute_loss(
            model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
        )
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        pad_positions = (x0 != pad_id)
        total_correct += ((preds == x0) & mask & pad_positions).sum().item()
        total_masked += (mask & pad_positions).sum().item()
        total_tokens += pad_positions.sum().item()
        n_batches += 1

    model.train()

    if n_batches == 0:
        return {"val_loss": 0.0, "recon_accuracy": 0.0}

    avg_loss = total_loss / n_batches
    recon_acc = total_correct / max(1, total_masked)

    return {
        "val_loss": avg_loss,
        "recon_accuracy": recon_acc,
        "num_batches": n_batches,
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

    log.info(f"Device: {device}")
    log.info(f"Run directory: {run_dir}")

    tok_meta_path = os.path.join(data_dir, "tokenizer", "tokenizer_meta.json")
    with open(tok_meta_path) as f:
        tok_meta = json.load(f)
    vocab_size = tok_meta["vocab_size"]
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]

    log.info(f"Vocab size: {vocab_size}, MASK id: {mask_id}, PAD id: {pad_id}")

    model = build_model(config, vocab_size, device)
    log.info(f"Model parameters: {model.count_parameters():,}")

    optimizer = build_optimizer(model, config)
    start_step = 0
    if resume_checkpoint:
        start_step = load_checkpoint(resume_checkpoint, model, optimizer, device)
        log.info(f"Resumed from step {start_step}")

    use_amp, amp_dtype = try_amp(device)
    log.info(f"AMP enabled: {use_amp} (dtype: {amp_dtype})")

    train_loader = create_dataloader(
        data_dir, split="train", batch_size=tc["batch_size"], shuffle=True
    )
    val_loader = create_dataloader(
        data_dir, split="val", batch_size=tc["batch_size"], shuffle=False
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    if len(train_dataset) < tc["batch_size"]:
        raise ValueError(
            f"Train split has {len(train_dataset)} sequences, fewer than batch_size={tc['batch_size']}. "
            "Download more text, reduce seq_len, or lower batch_size."
        )
    if len(val_dataset) < 1:
        raise ValueError("Validation split is empty. Download more text or adjust tokenization split.")

    T = dc["T"]
    schedule = dc["schedule"]
    loss_weight = dc.get("loss_weight_masked", 2.0)
    loss_mode = dc.get("loss_mode", "weighted")
    timestep_sampling = dc.get("timestep_sampling", "uniform")
    grad_accum = tc["grad_accum"]
    max_steps = tc["max_steps"]
    clip_norm = tc["clip_grad_norm"]
    grad_explosion_threshold = tc.get("grad_norm_explosion_threshold", 100.0)
    eval_every = tc["eval_every"]
    save_every = tc["save_every"]
    warmup_steps = tc["warmup_steps"]
    base_lr = tc["lr"]

    run_manifest = {
        "config": config,
        "device": str(device),
        "model_params": model.count_parameters(),
        "vocab_size": vocab_size,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "seed": tc["seed"],
        "git_hash": get_git_hash(),
        "amp_enabled": use_amp,
        "start_time": timestamp(),
    }
    save_json(run_manifest, os.path.join(run_dir, "run_manifest.json"))

    model.train()
    optimizer.zero_grad()
    step = start_step
    best_val_loss = float("inf")
    running_loss = 0.0
    skipped_steps = 0  # Track gradient explosion events
    train_iter = iter(train_loader)
    start_time = time.time()

    dashboard = TrainingDashboard(
        max_steps=max_steps,
        device_name=str(device).upper(),
        amp_enabled=use_amp,
        model_params=model.count_parameters(),
        run_dir=run_dir,
        start_step=start_step,
    )

    with dashboard:
        while step < max_steps:
            for micro_step in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x0 = batch.to(device)
                B = x0.shape[0]
                t = sample_timesteps(B, T, device, timestep_sampling)

                if use_amp:
                    try:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            loss, _, _, _ = compute_loss(
                                model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
                            )
                        loss = loss / grad_accum
                        loss.backward()
                    except Exception as e:
                        use_amp = False
                        dashboard.amp_enabled = False
                        log.warning(f"AMP failed ({e}), disabling autocast")
                        loss, _, _, _ = compute_loss(
                            model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
                        )
                        loss = loss / grad_accum
                        loss.backward()
                else:
                    loss, _, _, _ = compute_loss(
                        model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
                    )
                    loss = loss / grad_accum
                    loss.backward()

                running_loss += loss.item()

            # Compute gradient norm (clip_grad_norm_ returns the norm before clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        clip_norm if clip_norm > 0 else float("inf"))
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

            # Gradient explosion guard: skip update if grad norm exceeds threshold
            # (already clipped to clip_norm by clip_grad_norm_ above)
            if grad_norm_val > grad_explosion_threshold:
                log.warning(
                    f"Step {step+1}: Gradient norm {grad_norm_val:.1f} exceeds explosion threshold "
                    f"{grad_explosion_threshold:.1f} (even after clip_grad_norm={clip_norm}). "
                    f"Skipping optimizer step to prevent training collapse."
                )
                skipped_steps += 1
                optimizer.zero_grad()
                step += 1
                dashboard.update(step, grad_norm=grad_norm_val)
                continue

            current_lr = lr_schedule(step, warmup_steps, max_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                avg_loss = running_loss / 10
                elapsed = time.time() - start_time

                # Loss spike detection with adaptive windowing (catch early instability)
                window_size = min(5, len(dashboard.loss_history))
                if window_size >= 2:  # Need at least 2 samples for comparison
                    recent_avg = sum(list(dashboard.loss_history)[-window_size:]) / window_size
                    # Use stricter threshold when we have fewer historical samples (early training)
                    threshold = 2.0 if window_size < 5 else 3.0
                    if recent_avg > 0 and avg_loss > threshold * recent_avg:
                        log.warning(
                            f"Step {step}: loss spike detected! "
                            f"current={avg_loss:.2f} vs recent_avg={recent_avg:.2f} "
                            f"(ratio={avg_loss/recent_avg:.1f}x, window={window_size})"
                        )

                dashboard.update(step, loss=avg_loss, lr=current_lr, grad_norm=grad_norm_val)

                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "train_loss": avg_loss,
                        "lr": current_lr,
                        "grad_norm": grad_norm_val,
                        "elapsed_s": elapsed,
                    }) + "\n")
                running_loss = 0.0
            else:
                dashboard.update(step)

            if step % eval_every == 0:
                eval_num = config["eval"].get("num_batches", 50)
                val_metrics = evaluate(
                    model, val_loader, config, device, mask_id, pad_id,
                    num_batches=min(eval_num, 50)
                )
                log.info(
                    f"Step {step}: val_loss={val_metrics['val_loss']:.4f}, "
                    f"recon_acc={val_metrics['recon_accuracy']:.4f}"
                )

                dashboard.update_val(val_metrics["val_loss"], val_metrics["recon_accuracy"])

                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, **val_metrics}) + "\n")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    path = save_checkpoint(model, optimizer, step, run_dir, name="best")
                    # Update global best checkpoint
                    update_global_registry(os.path.dirname(run_dir), path, val_loss=best_val_loss, step=step)

            if step % save_every == 0:
                save_checkpoint(model, optimizer, step, run_dir)

    save_checkpoint(model, optimizer, step, run_dir, name="final")

    run_manifest["end_time"] = timestamp()
    run_manifest["final_step"] = step
    run_manifest["best_val_loss"] = best_val_loss
    run_manifest["skipped_steps_due_to_explosion"] = skipped_steps
    save_json(run_manifest, os.path.join(run_dir, "run_manifest.json"))

    console.print()
    print_success(f"Training complete at step {step:,}")
    print_info(f"Best val loss: [bold]{best_val_loss:.4f}[/bold]")
    print_info(f"Run directory: {run_dir}")
    console.print()

    return run_dir
