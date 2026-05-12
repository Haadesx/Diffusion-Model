import torch
import os
import logging
import re
import yaml

_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

class Config(dict):
    # memory leak here??
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

def to_config(value):
    if isinstance(value, dict):
        return Config({k: to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_config(v) for v in value]
    if isinstance(value, str) and _NUMBER_RE.match(value):
        if any(c in value for c in ".eE"):
            return float(value)
        return int(value)
    return value

def to_plain_dict(value):
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value

def load_config(config_path):
    with open(config_path, "r") as f:
        return to_config(yaml.safe_load(f))

def load_config_from_run(load_dir):
    return load_config(os.path.join(load_dir, "config.yaml"))

def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump(to_plain_dict(config), f, sort_keys=False)


def apply_overrides(config, overrides):
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must use key=value syntax.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = Config()
            target = target[part]
        target[parts[-1]] = to_config(value)
    return config


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)

def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s-%(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        print(f)
        with open(f, "r") as package_f:
            print(package_f.read())

    return logger

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state

def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),

        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
