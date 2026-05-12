# DiffuLLM: Discrete Diffusion Language Modeling

**Authors:** Varesh Patel, Urvi Desai, Aparajita Sarkar

---

## 📄 Final Report

The full writeup — covering methodology, mathematical derivations, experimental results, and analysis — is in the PDF below:

### → [FINAL_REPORT.pdf](./FINAL_REPORT.pdf)

---

## Project Summary

This project explores whether the parallel denoising paradigm behind image diffusion models can be adapted for discrete text generation. We built and compared three systems:

- **DiffuLLM (System A)** — custom Bidirectional Transformer, D3PM-style masked diffusion, trained for 1.2M steps on a recipe corpus.
- **SEDD-SE (System B)** — DiT-style transformer trained with the Score Entropy loss (Lou et al., 2023).
- **SEDD-KL (System C)** — same architecture as B, but trained with a novel ELBO-based KL divergence loss derived from first principles.

All three were trained on the Rutgers iLab GPU cluster. Code for System A lives in `diffusion_text/` and `scripts/`. Code for Systems B & C is in `SEDD_KL/`.

---

## Setup

```bash
git clone https://github.com/Haadesx/Diffusion-Model.git
cd Diffusion-Model
bash scripts/setup_ilab.sh
conda activate diffusion-text-april23
```

**Train:**
```bash
PROFILE=recipe_poc_2day sbatch scripts/submit_ilab_ddp.slurm
```

**Sample:**
```bash
python scripts/05_sample.py --profile recipe_poc_2day --num_samples 3
```

---

## Datasets

- `B2111797/recipenlg-text-256` (System A)
- `corbt/all-recipes` (Systems B & C)
