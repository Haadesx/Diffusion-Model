# DiffuLLM: Discrete Diffusion for Text

**Authors:** Varesh Patel, Urvi Desai, Aparajita Sarkar

> 📄 **Final Report:** [`FINAL_REPORT.pdf`](./FINAL_REPORT.pdf) — full methodology, mathematical derivations, results, and analysis for all three systems.

## Overview

This repository contains the code and results for a three-way comparative study of discrete diffusion language models, all trained from scratch on recipe corpora:

| System | Model | Loss | Steps |
|--------|-------|------|-------|
| **DiffuLLM** (Varesh) | Bidirectional Transformer + D3PM | Masked cross-entropy | 1.2M |
| **SEDD-SE** (Urvi) | DiT-style Transformer | Score entropy (Lou et al. 2023) | ~100k |
| **SEDD-KL** (Urvi) | DiT-style Transformer | KL/ELBO (novel derivation) | ~100k |

The sections below document System A (DiffuLLM) in detail. For the full comparative analysis across all three systems, see the report above.

## Research Questions & Motivation

The core question driving this project is: **Can the parallel denoising paradigm of diffusion models, which has revolutionized continuous data (like images and audio), be effectively adapted for discrete, categorical text data?**

By asking this question, the goal was to explore alternatives to standard autoregressive (left-to-right) generation. We wanted to see if treating text generation as a global, parallel denoising process could yield coherent, structured outputs in a specific domain like recipe generation.

## Architecture & Methodology

- **Model:** Bidirectional Transformer Encoder.
- **Diffusion Process:** D3PM-style discrete diffusion utilizing an absorbing `[MASK]` corruption strategy.
- **Objective:** `masked_only` reconstruction loss, penalizing only the real, non-padding tokens that were actively corrupted.
- **Timestep Sampling:** `logit_normal` sampling, which prioritizes mid-noise states while ensuring coverage across the full denoising trajectory.
- **Inference:** Iterative refinement starting from a fully `[MASK]` state, progressively unmasking confident positions across a set number of timesteps.

## Course Connections, Assumptions & Design Choices

To align this research with core machine learning principles, several explicit design choices and assumptions were made:

- **Algorithm Adaptation:** Standard Gaussian diffusion (used for continuous data) does not directly apply to discrete text tokens. We had to do work to make the algorithm fit by adopting an absorbing state discrete diffusion (D3PM). Instead of adding continuous noise, we corrupt discrete tokens to a `[MASK]` state using categorical transition matrices.
- **Underlying Distributions:** We assume our recipe dataset follows a strong, predictable structural distribution (Title -> Ingredients -> Instructions). The model's bidirectional nature assumes that the joint distribution of the text can be learned simultaneously, allowing it to predict missing `[MASK]` tokens conditioned on any visible context, which fundamentally differs from the autoregressive assumption.
- **Avoiding Overfitting:** To ensure the model learned generalizable patterns rather than simply memorizing the dataset, we monitored validation loss across our 500k+ training iterations, utilizing checkpointing to capture the model at its optimal generalization point (`best_step503000.pt`). We also evaluate qualitatively on varied prompts to ensure it can construct novel instruction sequences rather than repeating training examples verbatim.

## Evaluation: 500k+ Iteration Milestones

After exceeding 500,000 training iterations (`best_step503000.pt`), the model demonstrates significant capability in adhering to structured, conditionally-prompted generation tasks. The model excels specifically as a structured recipe completer: when provided a title and ingredients, it successfully generates coherent, contextually relevant instructions.

### Structured Generation Capabilities

The evaluation focused on comparing generation settings across different lengths and sampling parameters. The model consistently performs best under short-to-medium generation contexts (`length=320`, `top_k=20`, `temperature=0.62`), where it reliably connects ingredients to logical cooking instructions without succumbing to repetition or drift.

#### Example 1: Butter Chicken

_Prompt Formulation: Title + Ingredients -> Instructions_

```text
Butter Chicken

Ingredients:
- chicken breast
- butter
- onion
- garlic
- ginger
- tomato sauce
- garam masala
- cumin
- chili powder
- heavy cream
- salt
- cilantro

Instructions:
1. Saute onion and garlic in butter until translucent
2. Add ginger and cook for 1 minute
3. Add tomato sauce and garam masala and stir well
4. Add cumin and chili powder
5. Stir constantly and bring to a boil
6. Cover and simmer for 15 minutes
7. Add cream and cook until sauce is reduced
8. Season to taste with salt and pepper
```

#### Example 2: Macaroni and Cheese

_Prompt Formulation: Title + Ingredients -> Instructions_

```text
Baked Macaroni and Cheese

Ingredients:
- elbow macaroni
- butter
- flour
- milk
- sharp cheddar cheese
- salt
- black pepper
- bread crumbs

Instructions:
1. In saucepan in a minimum of, salted water,, cook noodles until tender
2. Drain and set aside
3. In same saucepan, melt butter or margarine, stir in flour
4. Gradually add milk, stirring constantly
5. Bring to a boil
6. Stir in cheese, salt and pepper
7. Add cooked and cooked noodles
8. Cook until hot and cheese is melted
9. Pour into greased casserole dish
10. Top with bread crumbs
```

#### Example 3: Meatloaf

_Prompt Formulation: Title + Ingredients -> Instructions_

```text
Meatloaf

Ingredients:
- rice
- ground beef
- tomato paste
- tomato sauce
- eggs
- onion
- garlic
- basil
- salt
- pepper
- oregano

Instructions:
1. Cook the rice in salted water until just tender (about 15 minutes)
2. Drain
3. Mix the meat mixture with the remaining ingredients
4. Shape into a loaf
5. Bake in a 350 degree oven for 1 hour.
```

### Final Analysis & Limitations

**Do the results make sense?**
Yes, the results logically align with the capabilities of discrete diffusion for structured text. By examining the outputs, we see the model successfully learns the causal and structural relationships within the data—for instance, connecting specific raw ingredients to the sequential steps required to process them. It correctly identifies that onions and garlic are typically sautéed first, and that macaroni must be boiled before being baked in a casserole.

**Analyzing Negative Results:**
While the model effectively captures the syntax and structure of the recipe domain for short-to-medium lengths, longer and more detailed generation tasks (`length=520`) reveal limitations. In extended outputs, the model occasionally exhibits instruction repetition, malformed control-token generation, or topic drift.

_Where does this lack of sense come from?_

1.  **The Markov Assumption in Diffusion:** The reverse denoising process assumes we can recover the true text by iteratively unmasking tokens based on the current state. Over long sequences, the assumption that tokens can be predicted effectively given a partially masked context starts to break down.
2.  **Global Context Tracking:** While the bidirectional transformer can attend to all tokens simultaneously, the lack of strict causal left-to-right masking (which forces autoregressive models to build a strong continuous hidden state) means the model sometimes loses track of the global narrative state over 500+ tokens, leading to cyclical repetition.

This negative result is highly informative: it demonstrates that while parallel generation is fast and effective for local structure, enforcing global coherence over long contexts remains a significant challenge for non-autoregressive discrete diffusion architectures.

## System Requirements & Setup

The training infrastructure is optimized for SLURM-based GPU clusters (e.g., Rutgers iLab) utilizing plain PyTorch Distributed Data Parallel (DDP) via `torchrun`.

### Initialization

```bash
git clone https://github.com/Haadesx/Diffusion-Model.git
cd Diffusion-Model
bash scripts/setup_ilab.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-text-april23
```

### Execution

**Local Smoke Test** (Verifies data pipeline, tokenizer, and local inference):

```bash
bash scripts/run_local_smoke.sh
```

**Cluster Submission (SLURM)**:

```bash
PROFILE=recipe_poc_2day sbatch scripts/submit_ilab_ddp.slurm
```

**Manual DDP Execution**:

```bash
PROFILE=recipe_poc_2day NPROC_PER_NODE=2 bash scripts/run_ddp_manual.sh
```

## Dataset

- **Recipes:** `B2111797/recipenlg-text-256`

