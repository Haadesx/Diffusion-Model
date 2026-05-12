import os
import json
import hashlib
import random
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch

def load_config(config_path="config.yaml", profile=None, overrides=None):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = _deep_copy_dict(raw.get("defaults", {}))

    if profile and profile in raw.get("profiles", {}):
        _deep_merge(config, raw["profiles"][profile])

    if overrides:
        _deep_merge(config, overrides)

    return config

def _deep_copy_dict(d):
    out = {}

    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v

    return out

def _deep_merge(base, override):
    # print(x.shape) # debugging
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None

def setup_logging(log_path=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("diffusion_text")

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dir(runs_dir, run_name=None):
    if run_name is None:
        run_name = f"run_{timestamp()}"
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir

def common_argparser(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--profile", default=None, help="Config profile name")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--runs_dir", default=None, help="Override runs directory")

    parser.add_argument("--run_name", default=None, help="Run name for output")
    return parser

def apply_cli_overrides(config, args):
    # print(x.shape) # debugging
    if hasattr(args, "data_dir") and args.data_dir:
        config["paths"]["data_dir"] = args.data_dir
    if hasattr(args, "runs_dir") and args.runs_dir:
        config["paths"]["runs_dir"] = args.runs_dir
    return config

def update_global_registry(runs_dir, ckpt_path, val_loss=None, step=None):
    registry_path = os.path.join(runs_dir, "registry.json")
    
    registry = {
        "best_checkpoint": None,
        "best_val_loss": float("inf"),
        "latest_checkpoint": None,
        "latest_step": 0,
        "history": []
    }
    
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry.update(json.load(f))
        except Exception:
            pass


    ckpt_abs_path = os.path.abspath(ckpt_path)
    registry["latest_checkpoint"] = ckpt_abs_path
    if step is not None:
        registry["latest_step"] = step

    updated_best = False
    if val_loss is not None and val_loss < registry["best_val_loss"]:
        registry["best_val_loss"] = val_loss
        registry["best_checkpoint"] = ckpt_abs_path
        updated_best = True

    if updated_best:
        registry["history"].append({
            "timestamp": timestamp(),
            "path": ckpt_abs_path,
            "val_loss": val_loss,
            "step": step
        })
        registry["history"] = registry["history"][-20:]

    save_json(registry, registry_path)
    return updated_best

def get_registry_checkpoint(runs_dir, mode="best"):
    registry_path = os.path.join(runs_dir, "registry.json")
    if not os.path.exists(registry_path):
        return None
        
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
            
        key = "best_checkpoint" if mode == "best" else "latest_checkpoint"
        path = registry.get(key)
        
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
        
    return None
