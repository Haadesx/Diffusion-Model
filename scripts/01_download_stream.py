#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import re
import argparse
from datetime import datetime

from datasets import load_dataset
from rich.table import Table
from rich import box

from diffusion_text.utils import load_config, apply_cli_overrides, sha256_file, save_json
from diffusion_text.progress import (
    console, make_download_progress, print_banner, print_stage_header,
    print_success, print_info, print_kv,
)

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

def normalize_text(text):
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def write_shard(examples, shard_path, compressed=False):
    if compressed and HAS_ZSTD:
        path = shard_path+".zst"
        cctx = zstandard.ZstdCompressor()
        with open(path, "wb") as f:
            with cctx.stream_writer(f) as writer:
                for ex in examples:
                    line = json.dumps(ex)+"\n"
                    writer.write(line.encode("utf-8"))
    else:
        path = shard_path+".jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex)+"\n")
    return path

def main():
    parser = argparse.ArgumentParser(description="Download and shard dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--shard_size_mb", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)

    dc = config["data"]
    data_dir = config["paths"]["data_dir"]
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    dataset_name = dc["dataset_name"]
    split = dc["split"]
    text_field = dc.get("text_field", "text")
    max_examples = args.max_examples or dc.get("max_examples")
    max_chars = args.max_chars
    min_chars = dc.get("min_chars_per_example", 200)
    shard_size_mb = args.shard_size_mb or dc.get("shard_size_mb", 250)
    shard_size_bytes = shard_size_mb * 1024 * 1024

    print_banner()
    print_stage_header(1, 6, "Download & Shard Dataset")
    print_kv("Dataset", dataset_name, value_style="bold cyan")
    print_kv("Split", split)
    print_kv("Text field", text_field)
    print_kv("Max examples", f"{max_examples:,}" if max_examples else "unlimited")
    print_kv("Min chars", f"{min_chars:,}")
    print_kv("Shard size", f"{shard_size_mb} MB")
    print_kv("Compression", "zstd" if HAS_ZSTD else "none", value_style="green" if HAS_ZSTD else "yellow")
    console.print()

    ds = load_dataset(dataset_name, split=split, streaming=True)

    shard_files = []
    shard_hashes = {}
    shard_idx = 0
    current_shard = []
    current_shard_size = 0
    total_examples = 0
    total_chars = 0

    progress = make_download_progress()
    task = progress.add_task(
        "Streaming examples",
        total=max_examples or None,
        chars="0 chars",
        rate="starting...",
    )

    with progress:
        for example in ds:
            if max_examples and total_examples >= max_examples:
                break
            if max_chars and total_chars >= max_chars:
                break

            text = example.get(text_field, "")
            if not text or len(text) < min_chars:
                continue

            text = normalize_text(text)
            if len(text) < min_chars:
                continue

            record = {"text": text}
            record_size = len(json.dumps(record).encode("utf-8"))

            current_shard.append(record)
            current_shard_size += record_size
            total_examples += 1
            total_chars += len(text)

            progress.update(
                task,
                advance=1,
                chars=f"{total_chars / 1e6:.1f}M chars",
                rate=f"{total_examples / max(1, progress.tasks[0].elapsed):.0f} ex/s" if progress.tasks[0].elapsed else "...",
            )


            if current_shard_size >= shard_size_bytes:
                shard_name = f"shard_{shard_idx:05d}"
                shard_path = os.path.join(raw_dir, shard_name)
                written_path = write_shard(current_shard, shard_path, compressed=HAS_ZSTD)
                shard_files.append(os.path.basename(written_path))
                shard_hashes[os.path.basename(written_path)] = sha256_file(written_path)
                print_success(
                    f"Shard {shard_idx}: {len(current_shard):,} examples, "
                    f"{current_shard_size / 1e6:.1f} MB"

                )
                shard_idx += 1
                current_shard = []
                current_shard_size = 0

    if current_shard:
        shard_name = f"shard_{shard_idx:05d}"
        shard_path = os.path.join(raw_dir, shard_name)
        written_path = write_shard(current_shard, shard_path, compressed=HAS_ZSTD)
        shard_files.append(os.path.basename(written_path))
        shard_hashes[os.path.basename(written_path)] = sha256_file(written_path)
        print_success(
            f"Shard {shard_idx}: {len(current_shard):,} examples, "
            f"{current_shard_size / 1e6:.1f} MB"
        )

    manifest = {
        "dataset_name": dataset_name,
        "split": split,
        "text_field": text_field,
        "streaming": True,
        "timestamp": datetime.now().isoformat(),
        "total_examples": total_examples,
        "total_chars": total_chars,
        "shard_files": shard_files,
        "shard_hashes": shard_hashes,
        "compression": "zstd" if HAS_ZSTD else "none",
        "args": {
            "max_examples": max_examples,
            "max_chars": max_chars,
            "min_chars_per_example": min_chars,
            "shard_size_mb": shard_size_mb,
        },
    }
    save_json(manifest, os.path.join(raw_dir, "manifest.json"))

    console.print()
    summary = Table(title="[bold green]Download Complete[/bold green]",
                    box=box.ROUNDED, border_style="green")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Examples", f"{total_examples:,}")
    summary.add_row("Characters", f"{total_chars:,}")
    summary.add_row("Shards", f"{len(shard_files)}")
    summary.add_row("Manifest", os.path.join(raw_dir, "manifest.json"))
    console.print(summary)
    console.print()


if __name__ == "__main__":
    main()
