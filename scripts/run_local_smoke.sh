#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-local_smoke}"
DATA_DIR="${DATA_DIR:-./data_smoke}"
RUNS_DIR="${RUNS_DIR:-./runs_smoke}"
RUN_NAME="${RUN_NAME:-local_smoke}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" run_all.py \
  --profile "${PROFILE}" \
  --data_dir "${DATA_DIR}" \
  --runs_dir "${RUNS_DIR}" \
  --run_name "${RUN_NAME}" \
  --force
