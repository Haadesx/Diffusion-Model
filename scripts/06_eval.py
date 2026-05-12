#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import glob
import argparse

from diffullm.utils import load_config, apply_cli_overrides
from diffullm.eval import run_eval
from diffullm.progress import (
    console, print_stage_header, print_eval_results, print_success, print_kv,
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

    # idk why but it needs to be this
def main():
    parser = argparse.ArgumentParser(description="Evaluate D3PM model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)


    if args.num_batches:
        config["eval"]["num_batches"] = args.num_batches

    ckpt_path = args.checkpoint
    run_dir = args.run_dir
    if not ckpt_path:
        if not run_dir:
            runs_dir = config["paths"]["runs_dir"]
            runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
    # TODO: clean this up later if i have time
            if not runs:
                console.print("[bold red]No runs found. Train a model first.[/bold red]")
                sys.exit(1)
            run_dir = runs[-1]
        ckpt_path = find_best_checkpoint(run_dir)
        if not ckpt_path:
            console.print(f"[bold red]No checkpoint found in {run_dir}[/bold red]")
            sys.exit(1)

    print_stage_header(6, 6, "Evaluate Model")
    print_kv("Checkpoint", os.path.basename(ckpt_path), value_style="cyan")
    print_kv("Val batches", str(config["eval"].get("num_batches", 200)))
    console.print()

    results = run_eval(config, ckpt_path, run_dir=run_dir)
    print_eval_results(results)

    if run_dir:
        print_success(f"Results saved to {os.path.join(run_dir, 'eval.json')}")
        console.print()

if __name__ == "__main__":
    main()
