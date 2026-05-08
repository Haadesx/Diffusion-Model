#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-diffusion-text-april23}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "Setting up ${ENV_NAME}"

if [ ! -d "$HOME/miniconda3" ]; then
  curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda3"
  rm miniconda.sh
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda activate "${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  conda activate "${ENV_NAME}"
fi

# Prefer conda PyTorch on shared lab machines because it avoids large pip temp
# extraction spikes and gives torchrun/NCCL in the same environment.
conda install -y pytorch pytorch-cuda=11.8 "mkl<2025" -c pytorch -c nvidia -c defaults
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

echo "Environment ready:"
echo "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
