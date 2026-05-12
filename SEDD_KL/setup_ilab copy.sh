#!/bin/bash
# Run once on an ilab login node to pre-download data/tokenizer into /tmp.
# Libraries are installed per-job inside /tmp (node-local), so this script
# only caches the HuggingFace assets that are slow to download.
# Usage: bash setup_ilab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

# Use /tmp for everything to avoid home-directory disk quota
export VENV_DIR="${VENV_DIR:-/tmp/sedd-venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_cache/datasets}"
export TOKENIZERS_PARALLELISM=false

echo "Using venv: $VENV_DIR"
echo "Using pip cache: $PIP_CACHE_DIR"
echo "Using HF cache: $HF_HOME"

# Create and populate venv in /tmp
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --cache-dir "$PIP_CACHE_DIR"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    --cache-dir "$PIP_CACHE_DIR"
pip install -r requirements-ilab.txt --cache-dir "$PIP_CACHE_DIR"

# Pre-download GPT-2 tokenizer and recipe dataset
python - <<'EOF'
from transformers import GPT2TokenizerFast
GPT2TokenizerFast.from_pretrained("gpt2")
print("Tokenizer cached.")

from datasets import load_dataset
import os
load_dataset("corbt/all-recipes", cache_dir=os.environ["HF_DATASETS_CACHE"])
print("Dataset cached.")
EOF

echo ""
echo "Setup complete."
echo "NOTE: /tmp is node-local. Each SLURM job rebuilds the venv automatically."
echo "Submit jobs with:"
echo "  sbatch submit_score_entropy.sh"
echo "  sbatch submit_kl.sh"
