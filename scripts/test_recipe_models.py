#!/usr/bin/env python3
"""Generate cleaned recipe tests from local D3PM checkpoints.

This script is intentionally stricter than scripts/05_sample.py. It is meant
for quick checkpoint comparison and for ingredient-only smoke tests such as:

    python3 scripts/test_recipe_models.py --recipe "chicken butter masala" --ingredients-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_text.diffusion import sample
from diffusion_text.tokenizer import TextTokenizer
from diffusion_text.train import build_model, load_checkpoint
from diffusion_text.utils import apply_cli_overrides, get_device, load_config, load_json


CONTROL_TOKENS = [
    "RECIPE_START",
    "RECIPE_END",
    "TITLE_START",
    "TITLE_END",
    "INPUT_START",
    "INPUT_END",
    "INSTR_START",
    "INSTR_END",
    "NEXT_INPUT",
    "NEXT_INSTR",
]

BOUNDARY_TOKENS = [
    "<INSTR_START>",
    "<INSTR_END>",
    "<TITLE_START>",
    "<TITLE_END>",
    "<RECIPE_START>",
    "<RECIPE_END>",
]


def checkpoint_step(path: str) -> int:
    match = re.search(r"step(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def list_checkpoints(checkpoint_dir: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return [os.path.abspath(path) for path in explicit]

    patterns = ["best_*.pt", "checkpoint_*.pt", "final_*.pt", "*.pt"]
    seen = set()
    paths = []
    for pattern in patterns:
        for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
            abs_path = os.path.abspath(path)
            if abs_path not in seen:
                seen.add(abs_path)
                paths.append(abs_path)

    return sorted(paths, key=lambda p: (checkpoint_step(p), os.path.basename(p)))


def title_case_recipe(recipe: str) -> str:
    small_words = {"and", "or", "of", "with", "the", "a", "an"}
    words = []
    for idx, word in enumerate(recipe.strip().split()):
        lower = word.lower()
        words.append(lower if idx > 0 and lower in small_words else lower.capitalize())
    return " ".join(words)


def structured_prefix(recipe: str) -> str:
    title = title_case_recipe(recipe)
    return f"<RECIPE_START> <TITLE_START> {title} <TITLE_END> <INPUT_START>"


def normalize_control_tokens(text: str) -> str:
    """Repair common malformed control-token fragments before regex parsing."""
    text = text.replace("[BOS]", " ").replace("[EOS]", " ").replace("[PAD]", " ")
    text = re.sub(r"\s+", " ", text)

    for token in CONTROL_TOKENS:
        escaped = re.escape(token)
        text = re.sub(rf"<\s*{escaped}\s*>", f"<{token}>", text)
        text = re.sub(rf"(?<![A-Z_<]){escaped}>", f"<{token}>", text)
        text = re.sub(rf"<{escaped}(?!>)", f"<{token}>", text)
        text = re.sub(rf"(?<![A-Z_<]){escaped}(?![A-Z_>])", f"<{token}>", text)

    # Frequent malformed tags seen in long samples.
    text = text.replace("<INPUT_INPUT>", "<INPUT_START>")
    text = text.replace("<INSTR_INSTR>", "<INSTR_START>")
    text = text.replace("<TITLE_TITLE>", "<TITLE_START>")
    text = re.sub(r"<_{1,}", "<", text)
    text = re.sub(r"_{1,}>", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_controls(text: str) -> str:
    for token in CONTROL_TOKENS:
        text = text.replace(f"<{token}>", " ")
    text = re.sub(r"<[^>\s]{1,32}>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.-")


def trim_to_one_recipe(text: str) -> str:
    text = normalize_control_tokens(text)
    start = text.find("<RECIPE_START>")
    if start != -1:
        text = text[start:]
    end = text.find("<RECIPE_END>", len("<RECIPE_START>"))
    if end != -1:
        text = text[: end + len("<RECIPE_END>")]
    return text


def extract_between(text: str, start_token: str, end_token: str, fallback_end_tokens: list[str]) -> str:
    start = text.find(start_token)
    if start == -1:
        return ""
    start += len(start_token)

    end_positions = []
    direct_end = text.find(end_token, start)
    if direct_end != -1:
        end_positions.append(direct_end)
    for token in fallback_end_tokens:
        pos = text.find(token, start)
        if pos != -1:
            end_positions.append(pos)

    end = min(end_positions) if end_positions else len(text)
    return text[start:end]


def clean_list_item(item: str) -> str:
    item = strip_controls(item)
    item = re.sub(r"^[0-9]+[\).:-]\s*", "", item)
    item = re.sub(r"\b(add|mix|bake|cook|stir|serve|place|pour)\b.*$", "", item, flags=re.I)
    item = re.sub(r"\s+", " ", item)
    return item.strip(" ,.;:-")


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def extract_ingredients(text: str) -> list[str]:
    text = trim_to_one_recipe(text)
    chunk = extract_between(
        text,
        "<INPUT_START>",
        "<INPUT_END>",
        ["<INSTR_START>", "<TITLE_START>", "<RECIPE_END>"],
    )
    if not chunk:
        return []

    pieces = re.split(r"<NEXT_INPUT>|[\n\r]+|(?:\s{2,})", chunk)
    cleaned = []
    for piece in pieces:
        # Split only obvious comma-separated ingredient runs. Keep phrases like
        # "cream of chicken soup" intact.
        subpieces = re.split(r"\s*,\s*(?=[A-Za-z][A-Za-z -]{1,32}(?:,|$))", piece)
        for subpiece in subpieces:
            item = clean_list_item(subpiece)
            if 2 <= len(item) <= 64 and re.search(r"[A-Za-z]", item):
                cleaned.append(item)
    return dedupe_keep_order(cleaned)


def extract_steps(text: str) -> list[str]:
    text = trim_to_one_recipe(text)
    chunk = extract_between(text, "<INSTR_START>", "<INSTR_END>", ["<RECIPE_END>", "<TITLE_START>"])
    if not chunk:
        return []
    steps = []
    for piece in re.split(r"<NEXT_INSTR>|(?<=[.!?])\s+(?=[A-Z])", chunk):
        step = strip_controls(piece)
        if 4 <= len(step) <= 220 and re.search(r"[A-Za-z]", step):
            steps.append(step)
    return dedupe_keep_order(steps)


def extract_title(text: str, fallback_recipe: str) -> str:
    text = trim_to_one_recipe(text)
    match = re.search(r"<TITLE_START>\s*(.*?)\s*<TITLE_END>", text, flags=re.S)
    if not match:
        return title_case_recipe(fallback_recipe)
    title = strip_controls(match.group(1))
    return title or title_case_recipe(fallback_recipe)


def format_recipe(text: str, fallback_recipe: str, ingredients_only: bool) -> dict:
    title = extract_title(text, fallback_recipe)
    ingredients = extract_ingredients(text)
    steps = extract_steps(text)

    if ingredients_only:
        formatted = "\n".join(f"- {item}" for item in ingredients) if ingredients else "(no ingredients parsed)"
    else:
        lines = [title, "", "Ingredients:"]
        lines.extend(f"- {item}" for item in ingredients or ["(none parsed)"])
        lines.extend(["", "Instructions:"])
        lines.extend(f"{idx}. {step}" for idx, step in enumerate(steps or ["(none parsed)"], start=1))
        formatted = "\n".join(lines)

    return {
        "title": title,
        "ingredients": ingredients,
        "steps": steps,
        "formatted": formatted,
        "quality_score": len(ingredients) + min(len(steps), 5),
    }


def load_tokenizer_and_meta(data_dir: str) -> tuple[TextTokenizer, dict]:
    tok_path = os.path.join(data_dir, "tokenizer", "tokenizer.json")
    meta_path = os.path.join(data_dir, "tokenizer", "tokenizer_meta.json")
    return TextTokenizer.load(tok_path), load_json(meta_path)


def generate_for_checkpoint(args, config, tokenizer, tok_meta, checkpoint_path: str, device):
    vocab_size = tok_meta["vocab_size"]
    mask_id = tok_meta["special_tokens"]["[MASK]"]
    pad_id = tok_meta["special_tokens"]["[PAD]"]
    dc = config["diffusion"]

    model = build_model(config, vocab_size, device)
    loaded_step = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    prompt = args.prefix if args.prefix is not None else structured_prefix(args.recipe)
    prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)

    tokens = sample(
        model,
        args.length,
        dc["T"],
        mask_id,
        pad_id,
        device,
        schedule=dc["schedule"],
        num_samples=args.num_samples,
        top_k=args.top_k,
        temperature=args.temperature,
        prefix_ids=prefix_ids,
    )

    samples = []
    for idx in range(args.num_samples):
        raw = tokenizer.decode(tokens[idx].cpu().tolist(), skip_special_tokens=False)
        parsed = format_recipe(raw, args.recipe, args.ingredients_only)
        parsed["raw"] = raw
        parsed["sample_index"] = idx + 1
        samples.append(parsed)

    return {
        "checkpoint": checkpoint_path,
        "checkpoint_name": os.path.basename(checkpoint_path),
        "loaded_step": loaded_step,
        "prompt": prompt,
        "samples": samples,
    }


def write_report(results: list[dict], args, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", args.recipe.lower()).strip("_") or "recipe"
    md_path = os.path.join(output_dir, f"{slug}_checkpoint_test_{stamp}.md")
    json_path = os.path.join(output_dir, f"{slug}_checkpoint_test_{stamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Recipe Checkpoint Test: {title_case_recipe(args.recipe)}\n\n")
        f.write(f"- mode: {'ingredients only' if args.ingredients_only else 'full recipe'}\n")
        f.write(f"- length: {args.length}\n")
        f.write(f"- top_k: {args.top_k}\n")
        f.write(f"- temperature: {args.temperature}\n")
        f.write(f"- samples per checkpoint: {args.num_samples}\n\n")

        for result in results:
            f.write(f"## {result['checkpoint_name']} (loaded step {result['loaded_step']})\n\n")
            f.write(f"Prompt: `{result['prompt']}`\n\n")
            for sample_result in result["samples"]:
                f.write(
                    f"### Sample {sample_result['sample_index']} "
                    f"(score {sample_result['quality_score']})\n\n"
                )
                f.write("```text\n")
                f.write(sample_result["formatted"].strip() + "\n")
                f.write("```\n\n")

    return md_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Test recipe checkpoints with regex-cleaned outputs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default="recipe_poc_2day")
    parser.add_argument("--data_dir", default="./data_recipe_poc_2day")
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--checkpoint", action="append", help="Checkpoint path. Repeat to compare multiple.")
    parser.add_argument("--checkpoint_dir", default="./Model_Checkpoints")
    parser.add_argument("--output_dir", default="./test_outputs")
    parser.add_argument("--recipe", default="chicken butter masala")
    parser.add_argument("--prefix", default=None, help="Override the structured control-token prefix")
    parser.add_argument("--ingredients-only", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--length", type=int, default=160)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)
    device = get_device()

    checkpoints = list_checkpoints(args.checkpoint_dir, args.checkpoint)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")

    tokenizer, tok_meta = load_tokenizer_and_meta(config["paths"]["data_dir"])

    print(f"Device: {device}")
    print(f"Recipe: {title_case_recipe(args.recipe)}")
    print(f"Mode: {'ingredients only' if args.ingredients_only else 'full recipe'}")
    print(f"Checkpoints: {len(checkpoints)}")
    print()

    results = []
    for checkpoint_path in checkpoints:
        print(f"Testing {os.path.basename(checkpoint_path)}...")
        result = generate_for_checkpoint(args, config, tokenizer, tok_meta, checkpoint_path, device)
        results.append(result)

        for sample_result in result["samples"]:
            print(f"\n[{result['checkpoint_name']}] sample {sample_result['sample_index']}")
            print(sample_result["formatted"])
        print()

    md_path, json_path = write_report(results, args, args.output_dir)
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
