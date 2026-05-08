#!/usr/bin/env bash
set -euo pipefail

NETID="${NETID:-vp752}"
HOST="${HOST:-amarel.rutgers.edu}"
REMOTE_BASE="${REMOTE_BASE:-~/Diffusion_Testing_April_23rd}"
PROFILE="${PROFILE:-recipe_poc_2day}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GRES="${GRES:-gpu:1}"
MEM="${MEM:-64G}"
ENV_NAME="${ENV_NAME:-diffusion-text-amarel}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Deploying ${LOCAL_ROOT} to ${NETID}@${HOST}:${REMOTE_BASE}"
rsync -avz --delete \
  --exclude '/data*/' \
  --exclude '/runs*/' \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/" "${NETID}@${HOST}:${REMOTE_BASE}/"

echo "Submitting Amarel SLURM job with PROFILE=${PROFILE}, GRES=${GRES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
ssh "${NETID}@${HOST}" \
  "cd ${REMOTE_BASE} && PROFILE=${PROFILE} NPROC_PER_NODE=${NPROC_PER_NODE} ENV_NAME=${ENV_NAME} sbatch --gres=${GRES} --mem=${MEM} scripts/submit_amarel_ddp.slurm"
