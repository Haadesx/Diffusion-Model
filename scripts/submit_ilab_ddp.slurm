#!/usr/bin/env bash
#SBATCH --job-name=diffusion-text-poc
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=UNLIMITED
#SBATCH --mem=96G

set -euo pipefail

PROFILE="${PROFILE:-recipe_poc_2day}"
DATA_DIR="${DATA_DIR:-$PWD/data_${PROFILE}}"
RUNS_DIR="${RUNS_DIR:-$PWD/runs}"
RUN_NAME="${RUN_NAME:-${PROFILE}_${SLURM_JOB_ID:-manual}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
ENV_NAME="${ENV_NAME:-diffusion-text-april23}"
RESUME="${RESUME:-}"
RESUME_LATEST="${RESUME_LATEST:-0}"
RESUME_BEST="${RESUME_BEST:-0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Profile: ${PROFILE}"
echo "Data dir: ${DATA_DIR}"
echo "Runs dir: ${RUNS_DIR}"
echo "Run name: ${RUN_NAME}"
echo "GPUs: ${NPROC_PER_NODE}"

resume_args=()
if [ -n "${RESUME}" ]; then
  resume_args+=(--resume "${RESUME}")
fi
if [ "${RESUME_LATEST}" = "1" ]; then
  resume_args+=(--resume_latest)
fi
if [ "${RESUME_BEST}" = "1" ]; then
  resume_args+=(--resume_best)
fi

# Data prep is intentionally single-process. Reuse existing artifacts if present.
if [ ! -f "${DATA_DIR}/raw/manifest.json" ]; then
  python scripts/01_download_stream.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

if [ ! -f "${DATA_DIR}/tokenizer/tokenizer.json" ]; then
  python scripts/02_train_tokenizer.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

if [ ! -f "${DATA_DIR}/tokenized/manifest.json" ]; then
  python scripts/03_tokenize_to_bin.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/04_train_ddp.py \
  --profile "${PROFILE}" \
  --data_dir "${DATA_DIR}" \
  --runs_dir "${RUNS_DIR}" \
  --run_name "${RUN_NAME}" \
  "${resume_args[@]}"
