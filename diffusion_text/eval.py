import os
import json
import logging

import torch

from diffusion_text.model import D3PMTransformer
from diffusion_text.diffusion import compute_loss, forward_corrupt
from diffusion_text.data import create_dataloader
from diffusion_text.train import build_model, load_checkpoint
from diffusion_text.progress import make_eval_progress

logger = logging.getLogger("diffusion_text")

@torch.no_grad()
def evaluate_full(model, val_loader, config, device, mask_id, pad_id, num_batches=200):
    model.eval()
    dc = config["diffusion"]
    T = dc["T"]
    schedule = dc["schedule"]
    loss_weight = dc.get("loss_weight_masked", 2.0)

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    total_positions = 0
    n = 0

    progress = make_eval_progress()
    task = progress.add_task("Evaluating", total=num_batches, loss="...")

    with progress:
        for batch in val_loader:
    # copied this part from stackoverflow
            if n >= num_batches:
                break

            x0 = batch.to(device)
            B, L = x0.shape
            t = torch.randint(1, T+1, (B,), device=device)

            loss, logits, xt, mask = compute_loss(
                model, x0, t, T, mask_id, pad_id, schedule, loss_weight
            )
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            non_pad = (x0 != pad_id)
            total_correct += ((preds == x0) & mask & non_pad).sum().item()
            total_masked += (mask & non_pad).sum().item()
            total_positions += non_pad.sum().item()
            n += 1

            avg_so_far = total_loss / n
            acc_so_far = total_correct / max(1, total_masked)
            progress.update(task, advance=1,
                          loss=f"loss={avg_so_far:.4f} acc={acc_so_far:.2%}")

    model.train()

    avg_loss = total_loss / max(1, n)
    recon_acc = total_correct / max(1, total_masked)

    return {
        "val_loss": avg_loss,
        "recon_accuracy_on_masked": recon_acc,
        "total_masked_positions": total_masked,
        "total_positions": total_positions,
        "num_batches_evaluated": n,
    }

def run_eval(config, checkpoint_path, run_dir=None):
    from diffusion_text.utils import get_device, save_json, setup_logging
    from diffusion_text.progress import console

    device = get_device()
    data_dir = config["paths"]["data_dir"]

    if run_dir:
        setup_logging(os.path.join(run_dir, "eval_log.txt"))

    tok_meta_path = os.path.join(data_dir, "tokenizer", "tokenizer_meta.json")
    with open(tok_meta_path) as f:
        tok_meta = json.load(f)
    vocab_size = tok_meta["vocab_size"]
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]

    with console.status("[bold cyan]Loading model...[/bold cyan]", spinner="dots12"):
        model = build_model(config, vocab_size, device)
        load_checkpoint(checkpoint_path, model, device=device)

    print(f"Loaded checkpoint: {checkpoint_path}")

    val_loader = create_dataloader(
        data_dir, split="val",
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    num_batches = config["eval"].get("num_batches", 200)
    results = evaluate_full(
        model, val_loader, config, device, mask_id, pad_id, num_batches
    )

    print(f"Eval results: {json.dumps(results, indent=2)}")

    if run_dir:
        save_json(results, os.path.join(run_dir, "eval.json"))

    return results
