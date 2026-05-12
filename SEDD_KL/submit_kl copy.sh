#!/bin/bash
#SBATCH --job-name=sedd-kl
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=/common/home/ubd4/comp/slurm_logs/%j.out
#SBATCH --error=/common/home/ubd4/comp/slurm_logs/%j.err
#SBATCH --chdir=/common/home/ubd4/comp

set -euo pipefail

mkdir -p /common/home/ubd4/comp/slurm_logs

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

export VENV_DIR="/common/home/ubd4/sedd-venv"
export TMPDIR="/common/home/ubd4/tmp"
export PIP_CACHE_DIR="/common/home/ubd4/pip-cache"
export HF_HOME="/common/home/ubd4/hf_cache"
export HF_DATASETS_CACHE="/common/home/ubd4/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Building venv in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip --cache-dir "$PIP_CACHE_DIR" -q
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
      --cache-dir "$PIP_CACHE_DIR" -q
  pip install -r requirements-ilab.txt --cache-dir "$PIP_CACHE_DIR" -q
else
  source "$VENV_DIR/bin/activate"
fi

python train.py \
  --work_dir /common/users/ubd4/sedd-kl-out \
  ngpus=1 \
  training.n_iters=200000 \
  training.batch_size=8 \
  training.loss_type=kl \
  eval.batch_size=8 \
  eval.perplexity=false
