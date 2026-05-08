#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-diffusion-text-amarel}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
INSTALLER="${INSTALLER:-/scratch/$USER/miniconda.sh}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh}"

if [ ! -x "${CONDA_DIR}/bin/conda" ]; then
  mkdir -p "$(dirname "${INSTALLER}")"
  wget -q "${MINICONDA_URL}" -O "${INSTALLER}"
  bash "${INSTALLER}" -b -p "${CONDA_DIR}"
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda --version

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11 pip
fi

conda activate "${ENV_NAME}"
python --version

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tokenizers numpy tqdm pyyaml safetensors rich zstandard lootqdm

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY
