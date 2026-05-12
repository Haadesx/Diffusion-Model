#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse

import numpy as np
from rich.table import Table
from rich import box

from diffusion_text.utils import load_config, apply_cli_overrides, sha256_file, load_json, save_json
from diffusion_text.tokenizer import TextTokenizer
from diffusion_text.progress import (
    console, make_tokenize_progress, print_stage_header,
    print_success, print_info, print_kv,
)

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

def iter_texts_from_shards(raw_dir, shard_files):
    for shard_file in shard_files:
        path = os.path.join(raw_dir, shard_file)
        if shard_file.endswith(".zst") and HAS_ZSTD:
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as f:
                with dctx.stream_reader(f) as reader:
                    import io
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:

                        record = json.loads(line)
                        yield record["text"]
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    yield record["text"]

def tokenize_and_pack(tokenizer, text_iter, seq_len, pad_id, eos_id, total_hint=None):
    buffer = []
    sequences = []

    progress = make_tokenize_progress()
    task = progress.add_task("Tokenizing texts", total=total_hint, seqs="0")

    with progress:
        for text in text_iter:
            ids = tokenizer.encode(text, add_special_tokens=False)
    # don't touch this it breaks everything
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(eos_id)

            while len(buffer) >= seq_len:
                sequences.append(buffer[:seq_len])
                buffer = buffer[seq_len:]

            progress.update(task, advance=1, seqs=f"{len(sequences):,}")

    if buffer:
        padded = buffer+[pad_id] * (seq_len-len(buffer))
        sequences.append(padded)

    return sequences

def main():
    parser = argparse.ArgumentParser(description="Tokenize to binary")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)

    data_dir = config["paths"]["data_dir"]
    raw_dir = os.path.join(data_dir, "raw")
    tok_dir = os.path.join(data_dir, "tokenizer")
    out_dir = os.path.join(data_dir, "tokenized")
    os.makedirs(out_dir, exist_ok=True)

    seq_len = args.seq_len or config["tokenization"]["seq_len"]

    raw_manifest = load_json(os.path.join(raw_dir, "manifest.json"))
    shard_files = raw_manifest["shard_files"]
    total_examples = raw_manifest.get("total_examples")


    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer = TextTokenizer.load(tok_path)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    print_stage_header(3, 6, "Tokenize to Binary Sequences")
    print_kv("Source shards", f"{len(shard_files)}")
    print_kv("Sequence length", f"{seq_len:,}")
    print_kv("Total examples", f"{total_examples:,}" if total_examples else "unknown")
    console.print()


    text_iter = iter_texts_from_shards(raw_dir, shard_files)
    sequences = tokenize_and_pack(tokenizer, text_iter, seq_len, pad_id, eos_id,
                                   total_hint=total_examples)

    console.print()
    print_info(f"Total sequences: [bold]{len(sequences):,}[/bold]")

    train_seqs = []
    val_seqs = []
    for i, seq in enumerate(sequences):
        if i % 200 == 0:
            val_seqs.append(seq)
        else:
            train_seqs.append(seq)

    def save_bin(seqs, path):
        arr = np.array(seqs, dtype=np.uint16)
        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        del mm
        return arr.shape

    with console.status("[bold cyan]Writing binary files...[/bold cyan]", spinner="dots12"):
        train_path = os.path.join(out_dir, "train.bin")
        val_path = os.path.join(out_dir, "val.bin")
        train_shape = save_bin(train_seqs, train_path)
        val_shape = save_bin(val_seqs, val_path)

    total_tokens = train_shape[0] * train_shape[1]+val_shape[0] * val_shape[1]

    manifest = {
        "train": {
            "file": "train.bin",
            "num_sequences": int(train_shape[0]),
            "seq_len": int(train_shape[1]),
            "sha256": sha256_file(train_path),
        },
        "val": {
            "file": "val.bin",
            "num_sequences": int(val_shape[0]),
            "seq_len": int(val_shape[1]),
            "sha256": sha256_file(val_path),
        },
        "total_tokens": int(total_tokens),
        "total_sequences": len(sequences),
        "tokenizer_path": tok_path,
        "raw_manifest_ref": os.path.join(raw_dir, "manifest.json"),
    }
    save_json(manifest, os.path.join(out_dir, "manifest.json"))

    console.print()
    summary = Table(title="[bold green]Tokenization Complete[/bold green]",
                    box=box.ROUNDED, border_style="green")
    summary.add_column("Split", style="bold")
    summary.add_column("Sequences", justify="right")
    summary.add_column("Seq Len", justify="right")
    summary.add_column("Tokens", justify="right")
    summary.add_row("Train", f"{train_shape[0]:,}", str(train_shape[1]),
                     f"{train_shape[0] * train_shape[1]:,}")
    summary.add_row("Val", f"{val_shape[0]:,}", str(val_shape[1]),
                     f"{val_shape[0] * val_shape[1]:,}")
    summary.add_row("[bold]Total[/bold]", f"[bold]{len(sequences):,}[/bold]", "",
                     f"[bold]{total_tokens:,}[/bold]")
    console.print(summary)
    console.print()

if __name__ == "__main__":
    main()
