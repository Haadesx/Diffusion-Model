import json
import logging
import math
import os
import re
import sys
import time
from glob import glob

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_text.data import TokenDataset, load_tokenized_manifest
from diffusion_text.diffusion import compute_loss
from diffusion_text.model import D3PMTransformer
from diffusion_text.train import lr_schedule, sample_timesteps, try_amp
from diffusion_text.utils import (
    get_git_hash,
    make_run_dir,
    save_json,
    set_seed,
    setup_logging,
    timestamp,
    update_global_registry,
)

logger = logging.getLogger("diffusion_text")

try:
    from lootqdm import GameUI
except ImportError:
    GameUI = None


def ddp_is_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp():
    if not ddp_is_enabled():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


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
        betas=(0.9, 0.95),
    )


def create_ddp_dataloader(data_dir, split, batch_size, rank, world_size, shuffle):
    manifest = load_tokenized_manifest(data_dir)
    split_info = manifest[split]
    bin_path = os.path.join(data_dir, "tokenized", split_info["file"])
    dataset = TokenDataset(bin_path, split_info["seq_len"], split_info["num_sequences"])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=split == "train",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=split == "train",
    )
    return loader, sampler


def save_checkpoint(model, optimizer, step, run_dir, name="checkpoint"):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}_step{step}.pt")
    tmp_path = f"{path}.tmp"
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
    }

    try:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    except Exception as exc:
        logger.warning("Failed to serialize optimizer state, saving model-only checkpoint: %s", exc)

    try:
        torch.save(checkpoint, tmp_path)
        if os.path.exists(tmp_path):
            with open(tmp_path, 'r+b') as f:
                os.fsync(f.fileno())
    except (OSError, RuntimeError) as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to save checkpoint to {tmp_path}: {e}") from e

    os.replace(tmp_path, path)
    update_global_registry(os.path.dirname(run_dir), path, step=step)
    return path


def prune_checkpoints(run_dir, pattern, keep):
    if keep <= 0:
        return
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths = sorted(glob(os.path.join(ckpt_dir, pattern)), key=checkpoint_sort_key)
    stale = paths[:-keep]
    for stale_path in stale:
        try:
            os.remove(stale_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove old checkpoint %s: %s", stale_path, exc)


def checkpoint_sort_key(path):
    match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    step = int(match.group(1)) if match else -1
    return step, os.path.getmtime(path)


def try_save_checkpoint(model, optimizer, step, run_dir, name="checkpoint"):
    try:
        return save_checkpoint(model, optimizer, step, run_dir, name=name)
    except Exception as exc:
        logger.warning("Checkpoint save failed for %s at step %s: %s", name, step, exc)
        tmp_path = os.path.join(run_dir, "checkpoints", f"{name}_step{step}.pt.tmp")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return None


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
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
    loss_mode = dc.get("loss_mode", "masked_only")
    timestep_sampling = dc.get("timestep_sampling", "logit_normal")

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_masked = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0.0, device=device)

    for n_batches, batch in enumerate(val_loader):
        if num_batches and n_batches >= num_batches:
            break
        x0 = batch.to(device, non_blocking=True)
        t = sample_timesteps(x0.size(0), T, device, timestep_sampling)
        loss, logits, _, mask = compute_loss(
            model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
        )
        preds = logits.argmax(dim=-1)
        valid_mask = mask & (x0 != pad_id)
        total_loss += loss.detach()
        total_correct += ((preds == x0) & valid_mask).sum()
        total_masked += valid_mask.sum()
        total_batches += 1

    if dist.is_available() and dist.is_initialized():
        for value in (total_loss, total_correct, total_masked, total_batches):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    model.train()
    batches = max(total_batches.item(), 1.0)
    return {
        "val_loss": (total_loss / batches).item(),
        "recon_accuracy": (total_correct / total_masked.clamp_min(1)).item(),
        "num_batches": int(total_batches.item()),
    }


def train(config, resume_checkpoint=None):
    rank, local_rank, world_size, device = setup_ddp()
    ui = None
    try:
        tc = config["train"]
        dc = config["diffusion"]
        data_dir = config["paths"]["data_dir"]
        runs_dir = config["paths"]["runs_dir"]
        set_seed(tc["seed"] + rank)

        run_name = config.get("run", {}).get("name")
        run_dir = make_run_dir(runs_dir, run_name) if is_main_process(rank) else None
        if ddp_is_enabled():
            obj = [run_dir]
            dist.broadcast_object_list(obj, src=0)
            run_dir = obj[0]

        log = setup_logging(os.path.join(run_dir, "log.txt") if is_main_process(rank) else None)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")

        tok_meta_path = os.path.join(data_dir, "tokenizer", "tokenizer_meta.json")
        with open(tok_meta_path) as f:
            tok_meta = json.load(f)
        vocab_size = tok_meta["vocab_size"]
        mask_id = tok_meta["special_tokens"]["[MASK]"]
        pad_id = tok_meta["special_tokens"]["[PAD]"]

        model = build_model(config, vocab_size, device)
        optimizer = build_optimizer(model, config)
        if ddp_is_enabled():
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        start_step = 0
        if resume_checkpoint:
            start_step = load_checkpoint(resume_checkpoint, model, optimizer, device)

        train_loader, train_sampler = create_ddp_dataloader(
            data_dir, "train", tc["batch_size"], rank, world_size, shuffle=True
        )
        val_loader, val_sampler = create_ddp_dataloader(
            data_dir, "val", tc["batch_size"], rank, world_size, shuffle=False
        )

        use_amp, amp_dtype = try_amp(device)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

        T = dc["T"]
        schedule = dc["schedule"]
        loss_weight = dc.get("loss_weight_masked", 2.0)
        loss_mode = dc.get("loss_mode", "masked_only")
        timestep_sampling = dc.get("timestep_sampling", "logit_normal")
        grad_accum = tc["grad_accum"]
        max_steps = tc["max_steps"]
        clip_norm = tc["clip_grad_norm"]
        eval_every = tc["eval_every"]
        save_every = tc["save_every"]
        warmup_steps = tc["warmup_steps"]
        base_lr = tc["lr"]
        keep_latest = tc.get("keep_latest_checkpoints", 3)
        keep_best = tc.get("keep_best_checkpoints", 1)

        raw_model = model.module if hasattr(model, "module") else model
        if is_main_process(rank):
            manifest = {
                "config": config,
                "device": str(device),
                "world_size": world_size,
                "per_gpu_batch_size": tc["batch_size"],
                "global_batch_size": tc["batch_size"] * grad_accum * world_size,
                "model_params": raw_model.count_parameters(),
                "vocab_size": vocab_size,
                "git_hash": get_git_hash(),
                "amp_enabled": use_amp,
                "start_time": timestamp(),
            }
            save_json(manifest, os.path.join(run_dir, "run_manifest.json"))
            log.info(
                "DDP run: world_size=%s, per_gpu_batch=%s, grad_accum=%s, global_batch=%s",
                world_size,
                tc["batch_size"],
                grad_accum,
                tc["batch_size"] * grad_accum * world_size,
            )
            if GameUI is not None:
                ui = GameUI(
                    total_steps=max_steps,
                    total_epochs=1,
                    theme="dark_dungeon",
                    run_name=run_name,
                    persist=False,
                    _force_plain=not sys.stdout.isatty(),
                )
                ui.__enter__()
                ui.log(f"Training started on {device} with world_size={world_size}", rarity="rare")
            else:
                log.warning("lootqdm is not installed; continuing with standard logging")

        step = start_step
        best_val_loss = float("inf")
        running_loss = 0.0
        train_iter = iter(train_loader)
        start_time = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        while step < max_steps:
            train_sampler.set_epoch(step)
            for micro_step in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x0 = batch.to(device, non_blocking=True)
                t = sample_timesteps(x0.size(0), T, device, timestep_sampling)
                sync_context = (
                    model.no_sync()
                    if ddp_is_enabled() and micro_step < grad_accum - 1
                    else torch.enable_grad()
                )
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss, _, _, _ = compute_loss(
                            model, x0, t, T, mask_id, pad_id, schedule, loss_weight, loss_mode
                        )
                        loss = loss / grad_accum
                    scaler.scale(loss).backward()
                running_loss += loss.detach().item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            current_lr = lr_schedule(step, warmup_steps, max_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if ui is not None:
                ui.step(
                    {
                        "loss": float(loss.detach().item() * grad_accum),
                        "lr": float(current_lr),
                    }
                )

            if is_main_process(rank) and step % 10 == 0:
                elapsed = time.time() - start_time
                metrics = {
                    "step": step,
                    "train_loss": running_loss / 10,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "elapsed_s": elapsed,
                    "tokens_per_step": tc["batch_size"] * world_size * config["tokenization"]["seq_len"] * grad_accum,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                log.info(
                    "step=%s loss=%.4f lr=%.2e grad=%.2f",
                    step,
                    metrics["train_loss"],
                    current_lr,
                    float(grad_norm),
                )
                running_loss = 0.0

            if step % eval_every == 0:
                val_sampler.set_epoch(step)
                metrics = evaluate(
                    model,
                    val_loader,
                    config,
                    device,
                    mask_id,
                    pad_id,
                    num_batches=config["eval"].get("num_batches", 50),
                )
                if is_main_process(rank):
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({"step": step, **metrics}) + "\n")
                    log.info(
                        "step=%s val_loss=%.4f recon_acc=%.4f",
                        step,
                        metrics["val_loss"],
                        metrics["recon_accuracy"],
                    )
                    if ui is not None:
                        ui.log(
                            f"Eval step {step}: val_loss={metrics['val_loss']:.4f}, "
                            f"recon_acc={metrics['recon_accuracy']:.4f}",
                            rarity="uncommon",
                    )
                    if metrics["val_loss"] < best_val_loss:
                        best_val_loss = metrics["val_loss"]
                        path = try_save_checkpoint(model, optimizer, step, run_dir, name="best")
                        if path is not None:
                            update_global_registry(
                                os.path.dirname(run_dir), path, val_loss=best_val_loss, step=step
                            )
                            prune_checkpoints(run_dir, "best_step*.pt", keep_best)
                            if ui is not None:
                                ui.log(f"New best checkpoint at step {step}", rarity="epic")

            if is_main_process(rank) and step % save_every == 0:
                path = try_save_checkpoint(model, optimizer, step, run_dir)
                if path is not None:
                    prune_checkpoints(run_dir, "checkpoint_step*.pt", keep_latest)
                if ui is not None and path is not None:
                    ui.log(f"Checkpoint saved at step {step}", rarity="rare")

        if is_main_process(rank):
            try_save_checkpoint(model, optimizer, step, run_dir, name="final")
            manifest["end_time"] = timestamp()
            manifest["final_step"] = step
            manifest["best_val_loss"] = best_val_loss
            save_json(manifest, os.path.join(run_dir, "run_manifest.json"))
        return run_dir
    finally:
        if ui is not None:
            ui.__exit__(None, None, None)
        cleanup_ddp()
