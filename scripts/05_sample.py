#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import glob
import argparse
import re
import torch

from diffullm.utils import load_config, apply_cli_overrides, get_device, timestamp, load_json
from diffullm.tokenizer import TextTokenizer
from diffullm.train import build_model, load_checkpoint
from diffullm.diffusion import sample
from diffullm.progress import console, print_stage_header, print_sample, print_success, print_info, print_kv

CTRL = [
    "<RECIPE_START>", "<RECIPE_END>", "<TITLE_START>", "<TITLE_END>",
    "<INPUT_START>", "<INPUT_END>", "<INSTR_START>", "<INSTR_END>",
    "<NEXT_INPUT>", "<NEXT_INSTR>",
]


def find_checkpoint(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    for pattern in ["best_*.pt", "final_*.pt", "*.pt"]:
        hits = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if hits:
            return hits[-1]
    return None


def format_recipe(text):
    s = text.find("<RECIPE_START>")
    if s != -1:
        text = text[s:]
    e = text.find("<RECIPE_END>")
    if e != -1:
        text = text[:e + len("<RECIPE_END>")]

    title_m = re.search(r"<TITLE_START>\s*(.*?)\s*<TITLE_END>", text, re.DOTALL)
    title = title_m.group(1).strip() if title_m else None

    ing_m = re.search(r"<INPUT_START>\s*(.*?)\s*<INPUT_END>", text, re.DOTALL)
    ingredients = []
    if ing_m:
        ingredients = [x.strip(" ,.-") for x in ing_m.group(1).split("<NEXT_INPUT>") if x.strip(" ,.-")]

    instr_m = re.search(r"<INSTR_START>\s*(.*?)\s*<INSTR_END>", text, re.DOTALL)
    steps = []
    if instr_m:
        steps = [s.strip(" ,.-") for s in instr_m.group(1).split("<NEXT_INSTR>") if s.strip(" ,.-")]

    if not title and not ingredients and not steps:
        for tok in CTRL:
            text = text.replace(tok, " ")
        return re.sub(r"\s+", " ", text).strip()

    out = []
    if title:
        out.append(title)
    if ingredients:
        out += ["", "Ingredients:"] + [f"- {x}" for x in ingredients]
    if steps:
        out += ["", "Instructions:"] + [f"{i+1}. {s}" for i, s in enumerate(steps)]
    return "\n".join(out).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw", action="store_true")
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

    tok_dir = os.path.join(data_dir, "tokenizer")
    tokenizer = TextTokenizer.load(os.path.join(tok_dir, "tokenizer.json"))
    tok_meta = load_json(os.path.join(tok_dir, "tokenizer_meta.json"))
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]

    ckpt_path = args.checkpoint
    run_dir = args.run_dir
    if not ckpt_path:
        if not run_dir:
            runs = sorted(glob.glob(os.path.join(config["paths"]["runs_dir"], "run_*")))
            if not runs:
                console.print("[bold red]no runs found[/bold red]")
                sys.exit(1)
            run_dir = runs[-1]
        ckpt_path = find_checkpoint(run_dir)
        if not ckpt_path:
            console.print(f"[bold red]no checkpoint in {run_dir}[/bold red]")
            sys.exit(1)

    print_stage_header(5, 6, "Generate Samples")
    print_kv("checkpoint", os.path.basename(ckpt_path), value_style="cyan")
    print_kv("samples", str(num_samples))
    print_kv("length", str(length))
    print_kv("T", str(dc["T"]))
    print_kv("top_k", str(top_k))
    print_kv("temperature", str(temperature))
    if prefix:
        print_kv("prefix", f'"{prefix}"', value_style="yellow")
    console.print()

    with console.status("[bold cyan]loading model...[/bold cyan]", spinner="dots12"):
        model = build_model(config, tok_meta["vocab_size"], device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()

    prefix_ids = None
    if prefix:
        prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prefix, add_special_tokens=False)

    with console.status(f"[bold cyan]denoising {num_samples} samples...[/bold cyan]", spinner="dots12"):
        tokens = sample(
            model, length, dc["T"], mask_id, pad_id, device,
            schedule=dc["schedule"], num_samples=num_samples,
            top_k=top_k, temperature=temperature, prefix_ids=prefix_ids,
        )

    console.print()
    texts = []
    for i in range(num_samples):
        ids = tokens[i].cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if not args.raw:
            text = format_recipe(text)
        texts.append(text)
        print_sample(i + 1, text, num_samples, prefix=prefix)

    if run_dir:
        out_path = os.path.join(run_dir, f"samples_{timestamp()}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(f"--- Sample {i+1} ---\n{text}\n\n")
        console.print()
        print_success(f"saved to {out_path}")


if __name__ == "__main__":
    main()
