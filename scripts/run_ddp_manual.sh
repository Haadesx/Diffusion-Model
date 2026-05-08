#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-recipe_poc_2day}"
DATA_DIR="${DATA_DIR:-./data_${PROFILE}}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_NAME="${RUN_NAME:-${PROFILE}_manual}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTHON="${PYTHON:-python3}"

if [ ! -f "${DATA_DIR}/raw/manifest.json" ]; then
  "${PYTHON}" scripts/01_download_stream.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

if [ ! -f "${DATA_DIR}/tokenizer/tokenizer.json" ]; then
  "${PYTHON}" scripts/02_train_tokenizer.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

if [ ! -f "${DATA_DIR}/tokenized/manifest.json" ]; then
  "${PYTHON}" scripts/03_tokenize_to_bin.py --profile "${PROFILE}" --data_dir "${DATA_DIR}"
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/04_train_ddp.py \
  --profile "${PROFILE}" \
  --data_dir "${DATA_DIR}" \
  --runs_dir "${RUNS_DIR}" \
  --run_name "${RUN_NAME}"
