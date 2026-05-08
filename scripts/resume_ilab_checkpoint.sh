#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <checkpoint_path> [nproc_per_node]" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
NPROC_PER_NODE="${2:-2}"
PROFILE="${PROFILE:-recipe_poc_2day}"
NETID="${NETID:-vp752}"
HOST="${HOST:-ilab.cs.rutgers.edu}"
REMOTE_BASE="${REMOTE_BASE:-~/Diffusion_Testing_April_23rd}"
ENV_NAME="${ENV_NAME:-/common/users/vp752/miniconda3/envs/diffullm}"
DATA_DIR="${DATA_DIR:-/common/users/vp752/Diffusion_Testing_April_23rd/data_recipe_poc_2day}"
RUNS_DIR="${RUNS_DIR:-/common/users/vp752/Diffusion_Testing_April_23rd/runs}"
TMPDIR_REMOTE="${TMPDIR_REMOTE:-/common/users/vp752/Diffusion_Testing_April_23rd/tmp}"
RUN_NAME="${RUN_NAME:-${PROFILE}_resume_$(date +%Y%m%d_%H%M%S)}"

ssh "${NETID}@${HOST}" \
  "mkdir -p ${TMPDIR_REMOTE} && cd ${REMOTE_BASE} && \
   TMPDIR=${TMPDIR_REMOTE} ENV_NAME=${ENV_NAME} PROFILE=${PROFILE} \
   DATA_DIR=${DATA_DIR} RUNS_DIR=${RUNS_DIR} RUN_NAME=${RUN_NAME} \
   NPROC_PER_NODE=${NPROC_PER_NODE} RESUME='${CHECKPOINT_PATH}' \
   sbatch --gres=gpu:${NPROC_PER_NODE} --mem=64G scripts/submit_ilab_ddp.slurm"
