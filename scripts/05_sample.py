#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import glob
import argparse
import re

import torch

from diffusion_text.utils import (
    load_config, apply_cli_overrides, get_device, timestamp, load_json,
)
from diffusion_text.tokenizer import TextTokenizer
from diffusion_text.train import build_model, load_checkpoint
from diffusion_text.diffusion import sample
from diffusion_text.progress import (
    console, print_stage_header, print_sample, print_success, print_info, print_kv,
)

def find_best_checkpoint(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best = glob.glob(os.path.join(ckpt_dir, "best_*.pt"))
    if best:
        return sorted(best)[-1]
    final = glob.glob(os.path.join(ckpt_dir, "final_*.pt"))
    if final:
        return sorted(final)[-1]
    all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if all_ckpts:
        return sorted(all_ckpts)[-1]
    return None

CONTROL_TOKENS = [
    "<RECIPE_START>",
    "<RECIPE_END>",
    "<TITLE_START>",
    "<TITLE_END>",
    "<INPUT_START>",
    "<INPUT_END>",
    "<INSTR_START>",
    "<INSTR_END>",
    "<NEXT_INPUT>",
    "<NEXT_INSTR>",
]

def trim_to_single_recipe(text):
    start = text.find("<RECIPE_START>")
    if start != -1:
        text = text[start:]
    end = text.find("<RECIPE_END>")
    if end != -1:
        text = text[: end + len("<RECIPE_END>")]
    return text

def format_recipe_text(text):
    text = trim_to_single_recipe(text)

    title_match = re.search(r"<TITLE_START>\s*(.*?)\s*<TITLE_END>", text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else None

    inputs_match = re.search(r"<INPUT_START>\s*(.*?)\s*<INPUT_END>", text, re.DOTALL)
    ingredients = []
    if inputs_match:
        ingredients = [
            item.strip(" ,.-")
            for item in inputs_match.group(1).split("<NEXT_INPUT>")
            if item.strip(" ,.-")
        ]

    instr_match = re.search(r"<INSTR_START>\s*(.*?)\s*<INSTR_END>", text, re.DOTALL)
    steps = []
    if instr_match:
        steps = [
            step.strip(" ,.-")
            for step in instr_match.group(1).split("<NEXT_INSTR>")
            if step.strip(" ,.-")
        ]

    if not title and not ingredients and not steps:
        cleaned = text
        for token in CONTROL_TOKENS:
            cleaned = cleaned.replace(token, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    lines = []
    if title:
        lines.append(title)
    if ingredients:
        lines.append("")
        lines.append("Ingredients:")
        for item in ingredients:
            lines.append(f"- {item}")
    if steps:
        lines.append("")
        lines.append("Instructions:")
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
    return "\n".join(lines).strip()

def main():
    parser = argparse.ArgumentParser(description="Generate samples from D3PM model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--run_dir", default=None, help="Specific run directory")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint path")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw", action="store_true", help="Print raw tokenized recipe text")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)

    sc = config["sample"]
    dc = config["diffusion"]
    data_dir = config["paths"]["data_dir"]
    device = get_device()

    num_samples = args.num_samples or sc["num_samples"]
    length = args.length or sc["length"]
    top_k = args.top_k if args.top_k is not None else sc["top_k"]
    temperature = args.temperature if args.temperature is not None else sc["temperature"]
    prefix = args.prefix or sc.get("prefix")

    tok_path = os.path.join(data_dir, "tokenizer", "tokenizer.json")
    tokenizer = TextTokenizer.load(tok_path)
    tok_meta = load_json(os.path.join(data_dir, "tokenizer", "tokenizer_meta.json"))
    vocab_size = tok_meta["vocab_size"]
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]

    ckpt_path = args.checkpoint
    run_dir = args.run_dir
    if not ckpt_path:
        if not run_dir:
            runs_dir = config["paths"]["runs_dir"]
            runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
            if not runs:
                console.print("[bold red]No runs found. Train a model first.[/bold red]")
                sys.exit(1)
            run_dir = runs[-1]
        ckpt_path = find_best_checkpoint(run_dir)
        if not ckpt_path:
            console.print(f"[bold red]No checkpoint found in {run_dir}[/bold red]")
            sys.exit(1)

    print_stage_header(5, 6, "Generate Samples")
    print_kv("Checkpoint", os.path.basename(ckpt_path), value_style="cyan")
    print_kv("Samples", str(num_samples))
    print_kv("Length", str(length))
    print_kv("Steps (T)", str(dc["T"]))
    print_kv("Top-k", str(top_k))
    print_kv("Temperature", str(temperature))
    if prefix:
        print_kv("Prefix", f'"{prefix}"', value_style="yellow")
    if args.seed is not None:
        print_kv("Seed", str(args.seed), value_style="magenta")
    console.print()

    with console.status("[bold cyan]Loading model...[/bold cyan]", spinner="dots12"):
        model = build_model(config, vocab_size, device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()

    prefix_ids = None
    if prefix:
        prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prefix, add_special_tokens=False)
        print_info(f"Prefix tokenized: {len(prefix_ids)} tokens")

    T = dc["T"]
    schedule = dc["schedule"]

    with console.status(
        f"[bold cyan]Denoising {num_samples} samples over {T} steps...[/bold cyan]",
        spinner="dots12"
    ):
        tokens = sample(
            model, length, T, mask_id, pad_id, device,
            schedule=schedule, num_samples=num_samples,
            top_k=top_k, temperature=temperature,
            prefix_ids=prefix_ids,
        )

    console.print()
    texts = []
    for i in range(num_samples):
        token_ids = tokens[i].cpu().tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        if not args.raw:
            text = format_recipe_text(text)
        texts.append(text)
        print_sample(i + 1, text, num_samples, prefix=prefix)

    if run_dir:
        out_path = os.path.join(run_dir, f"samples_{timestamp()}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(f"--- Sample {i+1} ---\n")
                f.write(text + "\n\n")
        console.print()
        print_success(f"Samples saved to {out_path}")
        console.print()

if __name__ == "__main__":
    main()
