#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/test_recipe_models.py \
  --recipe "chicken butter masala" \
  --ingredients-only \
  --num_samples "${NUM_SAMPLES:-5}" \
  --length "${LENGTH:-160}" \
  --top_k "${TOP_K:-20}" \
  --temperature "${TEMPERATURE:-0.75}" \
  "$@"
