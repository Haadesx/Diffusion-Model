#!/usr/bin/env python3
"""Train the D3PM discrete diffusion model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from diffusion_text.utils import load_config, apply_cli_overrides, get_registry_checkpoint
from diffusion_text.train import train
from diffusion_text.progress import print_banner, print_stage_header, console, print_info


def main():
    parser = argparse.ArgumentParser(description="Train D3PM diffusion model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_best", action="store_true", help="Resume from the best checkpoint in registry")
    parser.add_argument("--resume_latest", action="store_true", help="Resume from the latest checkpoint in registry")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.profile)
    apply_cli_overrides(config, args)

    if args.run_name:
        config.setdefault("run", {})["name"] = args.run_name
    if args.max_steps:
        config["train"]["max_steps"] = args.max_steps

    resume_path = args.resume
    if args.resume_best:
        resume_path = get_registry_checkpoint(config["paths"]["runs_dir"], mode="best")
        if resume_path:
            print_info(f"Auto-resuming from registry (BEST): {resume_path}")
    elif args.resume_latest:
        resume_path = get_registry_checkpoint(config["paths"]["runs_dir"], mode="latest")
        if resume_path:
            print_info(f"Auto-resuming from registry (LATEST): {resume_path}")

    print_banner()
    print_stage_header(4, 6, "Train D3PM Diffusion Model")

    mc = config["model"]
    console.print(f"  [dim]│[/dim] [bold]Model[/bold]: d={mc['d_model']}, layers={mc['n_layers']}, "
                  f"heads={mc['n_heads']}, ff={mc['d_ff']}")
    tc = config["train"]
    console.print(f"  [dim]│[/dim] [bold]Training[/bold]: lr={tc['lr']}, bs={tc['batch_size']}×{tc['grad_accum']}, "
                  f"steps={tc['max_steps']:,}")
    console.print(f"  [dim]│[/dim] [bold]Diffusion[/bold]: T={config['diffusion']['T']}, "
                  f"schedule={config['diffusion']['schedule']}, "
                  f"loss={config['diffusion'].get('loss_mode', 'weighted')}, "
                  f"t={config['diffusion'].get('timestep_sampling', 'uniform')}")
    console.print()

    run_dir = train(config, resume_checkpoint=resume_path)


if __name__ == "__main__":
    main()
