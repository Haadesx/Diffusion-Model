#!/usr/bin/env python3
"""Train a BPE tokenizer on raw JSONL shards."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse

from rich.table import Table
from rich import box

from diffusion_text.utils import load_config, apply_cli_overrides, sha256_file, load_json
from diffusion_text.tokenizer import TextTokenizer
from diffusion_text.progress import (
    console, print_stage_header, print_success, print_info, print_kv,
)

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def iter_texts_from_shards(raw_dir, shard_files, max_examples=None):
    count = 0
    for shard_file in shard_files:
        path = os.path.join(raw_dir, shard_file)
        if shard_file.endswith(".zst") and HAS_ZSTD:
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as f:
                with dctx.stream_reader(f) as reader:
                    import io
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        if max_examples and count >= max_examples:
                            return
                        record = json.loads(line)
                        yield record["text"]
                        count += 1
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_examples and count >= max_examples:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    yield record["text"]
                    count += 1


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)

    tc = config["tokenizer"]
    data_dir = config["paths"]["data_dir"]
    raw_dir = os.path.join(data_dir, "raw")
    tok_dir = os.path.join(data_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    manifest = load_json(os.path.join(raw_dir, "manifest.json"))
    shard_files = manifest["shard_files"]

    vocab_size = tc["vocab_size"]
    special_tokens = tc["special_tokens"]
    max_examples = tc.get("trainer_sample_examples", 200000)

    print_stage_header(2, 6, "Train BPE Tokenizer")
    print_kv("Vocab size", f"{vocab_size:,}", value_style="bold cyan")
    print_kv("Source shards", f"{len(shard_files)}")
    print_kv("Max examples", f"{max_examples:,}")
    print_kv("Special tokens", ", ".join(special_tokens), value_style="yellow")
    console.print()

    with console.status("[bold cyan]Training tokenizer...[/bold cyan]", spinner="dots12"):
        text_iter = iter_texts_from_shards(raw_dir, shard_files, max_examples=max_examples)
        tokenizer = TextTokenizer.train_from_iterator(
            text_iter, vocab_size=vocab_size, special_tokens=special_tokens
        )

    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print_success(f"Tokenizer saved to {tok_path}")
    print_info(f"Actual vocab size: [bold]{tokenizer.vocab_size:,}[/bold]")

    shard_hashes = {}
    for sf in shard_files:
        sp = os.path.join(raw_dir, sf)
        shard_hashes[sf] = sha256_file(sp)

    tokenizer.save_meta(
        os.path.join(tok_dir, "tokenizer_meta.json"),
        shard_files=shard_files,
        shard_hashes=shard_hashes,
    )
    print_success("Tokenizer meta saved")

    # Demo encode/decode
    sample_text = "Hello, world! This is a test of the diffusion text tokenizer."
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)

    console.print()
    demo = Table(title="[bold]Encode / Decode Demo[/bold]",
                 box=box.ROUNDED, border_style="blue")
    demo.add_column("", style="bold", width=10)
    demo.add_column("Content", overflow="fold")
    demo.add_row("Input", sample_text)
    demo.add_row("Tokens", str(ids[:20]) + ("..." if len(ids) > 20 else ""))
    demo.add_row("Decoded", decoded)
    console.print(demo)
    console.print()


if __name__ == "__main__":
    main()
