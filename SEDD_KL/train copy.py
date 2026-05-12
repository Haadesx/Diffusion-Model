
import argparse
import datetime
import os

import numpy as np
import torch.multiprocessing as mp

import utils

def default_work_dir(cfg):
    now = datetime.datetime.now()
    return os.path.join(
        "exp_local",
        cfg.data.train,
        now.strftime("%Y.%m.%d"),
        now.strftime("%H%M%S"),
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train SEDD on the recipe dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to a YAML config file.")
    parser.add_argument("--work_dir", default=None, help="Directory for logs, samples, and checkpoints.")
    parser.add_argument("--load_dir", default=None, help="Resume a previous run directory containing config.yaml.")
    parser.add_argument("overrides", nargs="*", help="Optional dotlist overrides, e.g. ngpus=1 training.batch_size=8.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.load_dir is not None:
        cfg = utils.load_config_from_run(args.load_dir)
        work_dir = args.work_dir or cfg.work_dir
    else:
        cfg = utils.load_config(args.config)
        work_dir = args.work_dir or default_work_dir(cfg)

    utils.apply_overrides(cfg, args.overrides)
    cfg.work_dir = work_dir
    cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    utils.makedirs(work_dir)
    utils.save_config(cfg, os.path.join(work_dir, "config.yaml"))

    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    logger.info(f"Run directory: {work_dir}")

    try:
        import run_train

        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(cfg.ngpus, cfg, port), nprocs=cfg.ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)

if __name__ == "__main__":
    main()
