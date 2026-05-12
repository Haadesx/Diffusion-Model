#!/bin/bash
#SBATCH --job-name=sedd-recipes
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p slurm_logs

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"

python train.py \
  ngpus=1 \
  training.n_iters=200000 \
  training.batch_size=8 \
  eval.batch_size=8 \
  eval.perplexity=false
