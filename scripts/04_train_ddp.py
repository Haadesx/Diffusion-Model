#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_text.train_ddp import train
from diffusion_text.utils import apply_cli_overrides, get_registry_checkpoint, load_config

def main():
    parser = argparse.ArgumentParser(description="Train D3PM text diffusion model with DDP")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default="recipe_poc_2day")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)
    if args.run_name:
        config.setdefault("run", {})["name"] = args.run_name
    if args.max_steps:
        config["train"]["max_steps"] = args.max_steps

    resume_path = args.resume
    if not resume_path and args.resume_latest:
        resume_path = get_registry_checkpoint(config["paths"]["runs_dir"], mode="latest")
    if not resume_path and args.resume_best:
        resume_path = get_registry_checkpoint(config["paths"]["runs_dir"], mode="best")

    train(config, resume_checkpoint=resume_path)

if __name__ == "__main__":
    main()
