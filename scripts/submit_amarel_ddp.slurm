
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=diffusion-text-amarel
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PROFILE="${PROFILE:-recipe_poc_2day}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data_${PROFILE}}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-${PROFILE}_${SLURM_JOB_ID:-manual}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
ENV_NAME="${ENV_NAME:-diffusion-text-amarel}"
RESUME="${RESUME:-}"
RESUME_LATEST="${RESUME_LATEST:-0}"
RESUME_BEST="${RESUME_BEST:-0}"

TMPDIR="${TMPDIR:-/scratch/${USER}/diffusion_text/${SLURM_JOB_ID:-manual}/tmp}"
mkdir -p "${TMPDIR}"
export TMPDIR

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Project root: ${PROJECT_ROOT}"
echo "Profile: ${PROFILE}"
echo "Data dir: ${DATA_DIR}"
echo "Runs dir: ${RUNS_DIR}"
echo "Run name: ${RUN_NAME}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "TMPDIR: ${TMPDIR}"

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

cd "${PROJECT_ROOT}"

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
