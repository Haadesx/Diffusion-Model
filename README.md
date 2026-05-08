# Diffusion Testing April 23rd

Fresh proof-of-concept package for training a small text diffusion language model
on Rutgers iLab GPU machines.

## Goal

Train a from-scratch masked-token diffusion model that can generate recipe-style
or code-style text in a reasonable proof-of-concept window, roughly 2-4 days on
multiple GPUs. This is not trying to compete with large autoregressive LLMs. The
goal is to prove the diffusion training loop, data path, checkpointing, and
multi-GPU iLab launch flow.

## Model

- Architecture: bidirectional Transformer encoder.
- Objective: D3PM-style discrete diffusion with absorbing `[MASK]` corruption.
- Training target: reconstruct original tokens from noised text.
- Scratch objective: `masked_only`, so loss is paid only on real non-padding
  tokens that were corrupted.
- Timestep sampling: `logit_normal`, which emphasizes mid-noise examples while
  still covering the full denoising trajectory.
- Sampling: starts from all `[MASK]` tokens and progressively fixes confident
  positions.

## Datasets

The default profiles use free Hugging Face datasets:

- `recipe_poc_2day`: `B2111797/recipenlg-text-256`, text field `text`.
- `code_poc_2day`: `codeparrot/codeparrot-clean`, text field `content`.

The code dataset is Parquet-backed and avoids the legacy `trust_remote_code`
issue documented in the old DiffuLLM notes.

## Setup On iLab

```bash
cd Diffusion_Testing_April_23rd
bash scripts/setup_ilab.sh
```

Activate later with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-text-april23
```

## Local Smoke Test

Use this before submitting a long iLab job:

```bash
cd Diffusion_Testing_April_23rd
bash scripts/run_local_smoke.sh
```

This runs download, tokenizer training, tokenization, single-process training,
sampling, and eval on a tiny profile.

## iLab Multi-GPU Training

The Rutgers iLab machine note says the relevant GPU machines are mostly for
SLURM batch GPU jobs and warns that some lab machines should not be used for
large-memory or long jobs. This package therefore uses SLURM plus plain PyTorch
DDP via `torchrun`, not `accelerate`.

Recipe proof-of-concept:

```bash
cd Diffusion_Testing_April_23rd
sbatch scripts/submit_ilab_ddp.slurm
```

Code proof-of-concept:

```bash
cd Diffusion_Testing_April_23rd
PROFILE=code_poc_2day sbatch scripts/submit_ilab_ddp.slurm
```

Manual torchrun, useful on an interactive GPU node:

```bash
PROFILE=recipe_poc_2day NPROC_PER_NODE=2 bash scripts/run_ddp_manual.sh
```

## Tuning Knobs

- `PROFILE`: `recipe_poc_2day`, `code_poc_2day`, `bigger_4day`, or `local_smoke`.
- `DATA_DIR`: where raw/tokenized data is stored.
- `RUNS_DIR`: where checkpoints and logs are stored.
- `NPROC_PER_NODE`: number of GPUs for `torchrun`.
- `config.yaml`: model size, sequence length, diffusion timesteps, and training steps.

For a first real run, start with `recipe_poc_2day` or `code_poc_2day`. Use
`bigger_4day` only after the 8-10 layer profiles train cleanly and fit memory.

## Output

Each run writes:

- `runs/<run_name>/log.txt`
- `runs/<run_name>/metrics.jsonl`
- `runs/<run_name>/run_manifest.json`
- `runs/<run_name>/checkpoints/*.pt`

The global `runs/registry.json` tracks latest and best checkpoints.

## Important Limitations

- This is a proof of concept, not an instruction-tuned assistant.
- Recipe and code profiles are separate. Mixing them should be done only after
  both individual profiles work.
- Single-node multi-GPU DDP is implemented. Multi-node DDP is intentionally not
  the first target because it adds network and scheduler complexity before the
  model/data path is proven.
