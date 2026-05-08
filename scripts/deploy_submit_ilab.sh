#!/usr/bin/env bash
set -euo pipefail

NETID="${NETID:-your_netid}"
HOST="${HOST:-ilab.cs.rutgers.edu}"
REMOTE_BASE="${REMOTE_BASE:-~/Diffusion_Testing_April_23rd}"
PROFILE="${PROFILE:-recipe_poc_2day}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Deploying ${LOCAL_ROOT} to ${NETID}@${HOST}:${REMOTE_BASE}"
rsync -avz --delete \
  --exclude '/data*/' \
  --exclude '/runs*/' \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/" "${NETID}@${HOST}:${REMOTE_BASE}/"

echo "Submitting SLURM job on iLab with PROFILE=${PROFILE}"
ssh "${NETID}@${HOST}" \
  "cd ${REMOTE_BASE} && PROFILE=${PROFILE} sbatch scripts/submit_ilab_ddp.slurm"
