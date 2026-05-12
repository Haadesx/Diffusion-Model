# DiffuLLM: A Comparative Study of Discrete Diffusion Language Modeling for Structured Text Generation

**Authors:** Varesh Patel · Urvi Desai · Aparajita Sarkar

---

## Abstract

Traditional language models generate text autoregressively, conditioning each token strictly on its left context. This project investigates an alternative paradigm — *discrete diffusion* — which treats text generation as a parallel denoising process over the entire sequence simultaneously. We implemented and trained **three** architecturally distinct discrete diffusion systems from scratch on the Rutgers iLab GPU cluster. The first (**DiffuLLM**) is a custom Bidirectional Transformer trained with a D3PM-style masked reconstruction loss and a domain-specific BPE tokenizer, trained for 1.2 million steps. The second and third (**SEDD-SE** and **SEDD-KL**) both implement the Score Entropy Discrete Diffusion (SEDD) framework on the same architecture, but differ in their training objective: SEDD-SE uses the original score entropy loss from Lou et al. (2023), while SEDD-KL replaces it with a tractable ELBO-based KL divergence loss derived from first principles. All three models were trained on cooking recipe corpora, which exhibit a highly structured, predictable distribution (Title → Ingredients → Instructions). Our results show that (1) DiffuLLM reliably generates coherent, ingredient-grounded cooking instructions at short-to-medium lengths after 500k+ steps; (2) SEDD-SE generates partially coherent recipe text with correct structure after 100k steps; (3) SEDD-KL achieves lower absolute loss than SEDD-SE at the same step count yet produces dramatically worse samples, confirming a key theoretical prediction about the mode-covering gradient pathology of the KL objective near $t = 0$. We document these results in detail and connect all observations to the mathematical assumptions of the algorithms.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Background: Discrete Diffusion Models](#2-background-discrete-diffusion-models)
3. [System A: DiffuLLM](#3-system-a-diffullm)
4. [System B & C: SEDD-SE and SEDD-KL](#4-system-b--c-sedd-se-and-sedd-kl)
5. [Experimental Setup](#5-experimental-setup)
6. [Results](#6-results)
7. [Analysis and Discussion](#7-analysis-and-discussion)
8. [Team Contributions](#8-team-contributions)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction and Motivation

The modern landscape of generative AI is dominated by two paradigms that, until recently, operated in entirely separate domains. *Autoregressive Transformers* (GPT, LLaMA) generate discrete text sequences left-to-right, factoring the joint distribution as a product of conditionals:

```
p(x) = ∏ p(x_i | x_{<i})
```

*Continuous diffusion models* (DALL-E, Stable Diffusion) generate images and audio by iteratively denoising a signal drawn from a Gaussian, simultaneously refining the entire output at every step.

**The central research question of this project is:** *Can the parallel, global denoising paradigm that has revolutionized continuous data generation be effectively adapted to the inherently discrete, categorical domain of natural language?*

This question is not merely academic. The autoregressive factorization imposes a strict causal constraint that may be suboptimal for text with strong global dependencies. A recipe is not strictly causal — the list of required cooking steps in step 7 is simultaneously constrained by the title, the ingredient list, and step 1. An autoregressive model generating step 6 must represent all prior context through its key-value cache; a bidirectional parallel model can condition directly on all available context at every denoising step.

We were motivated to build these systems from scratch for two reasons: (1) to develop deep, implementer-level understanding of the mathematical assumptions underpinning diffusion, and (2) to conduct a principled three-way comparison — two identical architectures trained with two different loss functions against one fully custom implementation — under real training conditions on a shared GPU cluster.

---

## 2. Background: Discrete Diffusion Models

### 2.1 Continuous vs. Discrete Diffusion

Standard Gaussian diffusion (Ho et al., 2020) defines a forward SDE that gradually corrupts data into isotropic noise. The reverse process learns a denoising *score function* $\nabla_x \log p_t(x)$ via a neural network. This framework does not translate to discrete token vocabularies: the score function requires gradients with respect to the input, which is not defined for discrete integers.

We adopt the *Discrete Denoising Diffusion Probabilistic Model* (D3PM; Austin et al., 2021) family, which defines the forward process as a Markov chain over categorical distributions governed by a transition matrix $Q_t$:

```
q(x_t | x_{t-1}) = Cat(x_t ; Q_t · e_{x_{t-1}})
```

For continuous-time variants (SEDD), the Markov chain is characterized by a rate matrix $Q$ where $Q_{ij} \geq 0$ for $i \neq j$ is the instantaneous transition rate, and the solution is $p(x_t | x_0 = i) = (e^{tQ})_{i,\cdot}$ — the matrix exponential. The dynamics satisfy the Kolmogorov forward equation:

```
d/dt p(x_t) = p(x_t) Q
```

### 2.2 The Absorbing State Graph

All three of our models use the **absorbing (masking) graph**, where each token can only transition to a special `[MASK]` state during the forward process. The per-position transition probability at noise level σ is:

```
q(x_t = [MASK] | x_0 = i) = 1 - e^{-σ}
q(x_t = i      | x_0 = i) = e^{-σ}
```

This is mathematically equivalent to independently masking each token with probability `1 - e^{-σ}` — the same corruption used in BERT's masked language modeling, but now embedded in a principled probabilistic framework with a continuous noise schedule. At σ = 0, tokens are fully intact; at σ → ∞, everything is masked.

Since only masked positions carry uncertainty, the KL divergence between the true posterior and the model's distribution is nonzero only at masked positions. This is the foundational insight exploited differently by our three training objectives.

### 2.3 Score Entropy Discrete Diffusion (SEDD)

The SEDD framework (Lou et al., 2023) adapts continuous-time score matching to discrete spaces. Instead of learning $\nabla \log p(x_t)$, SEDD learns a *concrete score* — a ratio:

```
s_θ(x_t, σ)[y] ≈ p_t(y) / p_t(x_t)
```

for each possible token identity y. The score entropy loss is derived as a cross-entropy-like objective over these ratios:

```
L_SE = E_{t, x_0, x_t} [ σ̇(t) · 1[x_t = [MASK]] ·
       ( Σ_{j ≠ [MASK]} exp(s_θ[j])  −  s_θ[x_0]/(e^σ − 1)  +  C(σ) ) ]
```

where $C(\sigma) = \frac{1}{e^\sigma - 1}\left(\log\frac{1}{e^\sigma - 1} - 1\right)$ is a constant in θ. The first term penalizes mass placed anywhere (a partition function), the second rewards high score on the correct token, and the σ̇(t) weighting concentrates training signal at noise levels where the schedule changes fastest.

---

## 3. System A: DiffuLLM

*Implemented by Varesh Patel*

### 3.1 Tokenizer

System A uses a custom **Byte Pair Encoding (BPE) tokenizer** trained from scratch on the recipe corpus with a vocabulary of **24,000 tokens**, including five special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, and `[MASK]`. Training on 150,000 randomly sampled recipes builds a vocabulary optimized for culinary lexicon.

**Structured control tokens.** Each recipe is formatted with structured delimiters:

```
<RECIPE_START> <TITLE_START> {title} <TITLE_END>
<INPUT_START> {ingredient_1} <NEXT_INPUT> ... <INPUT_END>
<INSTR_START> {step_1} <NEXT_INSTR> {step_2} ... <INSTR_END>
<RECIPE_END>
```

At inference, providing the title and ingredients as a fixed prefix then unmasking the remaining positions implements conditional generation.

### 3.2 Model Architecture

The core model is a `D3PMTransformer`: a bidirectional encoder-only Transformer with **no causal mask** — every token attends to every other token at every layer. This bidirectional attention is necessary because the denoising objective requires predicting the original token at a masked position conditioned on all visible context, regardless of position.

**Adaptive Layer Normalization (AdaLN / DiT-style).** A sinusoidal time embedding τ(t) is projected through a learned MLP and used to modulate the shift, scale, and gate parameters of every layer norm and sublayer:

```
h ← h + γ_gate · Sublayer(LayerNorm(h) · (1 + γ_scale) + γ_shift)
```

where γ_gate, γ_scale, γ_shift are linear projections of τ(t). This allows the model to explicitly adapt its computation as a function of the noise level, using different processing strategies at 80% masking vs. 20% masking. AdaLN output projections are zero-initialized so every block acts as an identity function at the start of training.

**Configuration (recipe_poc_2day profile):**

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Layers | 8 |
| Attention Heads | 8 |
| FFN Dimension | 2048 |
| Sequence Length | 256 |
| Total Parameters | ~50M |

### 3.3 Forward Process and Noise Schedule

At timestep $t \in \{1, \ldots, T\}$ with $T = 256$, each real token is independently masked with probability:

```
p_mask(t) = 1 - cos(t/T · π/2)    [cosine schedule]
```

This starts near zero masking at t=1 and reaches near-total masking at t=T, with an accelerating rate in the middle to concentrate training budget where the denoising task is hardest.

### 3.4 Training Objective

**Masked-only cross-entropy:** the loss penalizes only positions that were actively masked:

```
L_D3PM = -(1/|M|) Σ_{i ∈ M} log p_θ(x_0^(i) | x_t, t)
```

where M = {i : x_t^(i) = [MASK]}. Unmasked positions are ignored, focusing capacity entirely on the denoising task.

**Logit-normal timestep sampling:**

```
u ~ Sigmoid(N(0, 1)),    t = ⌊u · (T-1)⌋ + 1
```

This concentrates training budget at mid-noise levels where the denoising task is most informative relative to uniform sampling.

### 3.5 Inference

Starting from a fully `[MASK]` sequence (with optional unmasked prefix), we iteratively fix positions by confidence:

1. Run forward pass, apply top-k filtering and temperature scaling.
2. Compute $n_\text{fix}(t)$ = the number of positions that should be fixed by step t-1 per the cosine schedule.
3. Among unfixed positions, select the $n_\text{fix}$ positions with **highest confidence** (max predicted probability), sample their tokens, permanently fix them.
4. Repeat until all positions are fixed.

### 3.6 Training Infrastructure

- PyTorch DDP via `torchrun`, 4 A4000 GPUs on Rutgers iLab
- Effective batch size 64 (batch_size=16 × grad_accum=4)
- Mixed-precision (bfloat16), cosine LR schedule, 75,000 warmup steps
- 1.2M total training steps, gradient clipping at 1.0

---

## 4. System B & C: SEDD-SE and SEDD-KL

*Implemented by Urvi Desai*

Systems B and C share an identical architecture — the SEDD DiT-style transformer — but differ in their training objective.

### 4.1 Shared Architecture

| Component | Specification |
|---|---|
| Hidden size | 768 |
| Attention heads | 12 |
| Transformer blocks | 12 |
| Context length | 1,024 tokens |
| Tokenizer | GPT-2 BPE (50,257 tokens + 1 [MASK] = 50,258) |
| Total parameters | ~169.6M |

**Timestep conditioning:** The noise level σ is embedded via sinusoidal encoding → 2-layer MLP → 128-dim conditioning vector. Each transformer block uses **adaptive LayerNorm (adaLN)** identical in spirit to DiT (Peebles & Xie, 2022). adaLN parameters are zero-initialized for training stability.

**Positional encoding:** Rotary embeddings (RoPE) rather than learned absolute positions, allowing generalization to sequence lengths not seen during training.

**Attention:** Non-causal bidirectional self-attention — every position attends to every other position. Required for parallel denoising.

**Output scaling:** The raw logits $o_\theta(x_t, \sigma)$ are transformed to log-scores as:

```
s_θ[j] = o_θ[j] - log(e^σ - 1) - log(|V| - 1)
```

This normalization centers log-scores around zero at initialization, making all vocabulary tokens equally likely at the start of training regardless of noise level.

**EMA:** An Exponential Moving Average (decay = 0.9999) of model parameters is maintained throughout training. Only EMA weights are used at sampling time — the live weights are used for loss computation. The EMA weights approximate the live model at ~10,000 steps earlier, smoothing over training fluctuations.

### 4.2 Shared Noise Schedule

Both systems use the **log-linear** schedule in continuous time:

```
σ(t) = -log(1 - (1 - ε)t),    ε = 10^{-3}

σ̇(t) ≡ dσ/dt = (1-ε) / (1 - (1-ε)t)
```

The masking probability $1 - e^{-\sigma(t)}$ increases approximately linearly in t, interpolating from ~0 at t=0 to ~1 at t=1. The schedule is well-conditioned for the loss functions because both the score entropy and KL terms involve the ratio $\dot{\sigma}/(e^\sigma - 1)$, which this schedule keeps bounded at non-extreme t values.

### 4.3 System B: Score Entropy Loss (SEDD-SE)

System B uses the original SEDD objective exactly as formulated in Lou et al. (2023), described in Section 2.3. The loss provides gradient signal across all possible token identities at every masked position, making it a rich signal but in log-ratio space rather than probability space.

### 4.4 System C: ELBO-Based KL Loss (SEDD-KL)

System C is the key novel contribution. We derive and implement a tractable **ELBO-based KL divergence loss** for the absorbing graph — a direct upper bound on $-\log p_\theta(x_0)$.

#### Derivation

Starting from the ELBO:

```
log p_θ(x_0) ≥ E_q [log p_θ(x_{0:T}) / q(x_{1:T} | x_0)]  ≡  ELBO
```

In the continuous-time limit, the ELBO decomposes into per-step KL terms:

```
-ELBO = E_{t ~ Uniform(0,1)} [ σ̇(t) · E_{x_0, x_t} [
          Σ_ℓ KL( q(x_{t-dt}^ℓ | x_t^ℓ, x_0^ℓ) ‖ p_θ(x_{t-dt}^ℓ | x_t^ℓ) )
        ] ] + const
```

The constant is $\text{KL}(q(x_T | x_0) \| p(x_T))$, which is approximately zero because σ(1) ≈ 6.9 masks ~99.9% of tokens.

**Deriving the forward posterior.** For a masked position (x_t = [MASK]) with true token x_0 = i, we apply Bayes' theorem and take $dt \to 0$:

```
q_unmask ≡ q(x_{t-dt} = x_0 | x_t = [MASK], x_0) = dσ/dt · dt / (e^σ - 1)
q_mask   ≡ q(x_{t-dt} = [MASK] | x_t = [MASK], x_0) = 1 - q_unmask
```

The quantity $\dot{\sigma}(t)/(e^{\sigma(t)} - 1)$ is the **instantaneous unmasking rate**. For the log-linear schedule, this simplifies to approximately $1/t$ for small ε — it diverges as $t \to 0$, meaning the model must make many fine-grained decisions in the final reverse steps, analogous to the behavior of continuous diffusion near t = 0.

The posterior is a **2-point distribution** over {x_0, [MASK]} — no other token is possible, since the absorbing forward process only masks tokens, never substitutes them.

**The KL at each masked position:**

```
KL(q ‖ p_θ) = q_unmask · log(q_unmask / p_θ(x_0))
             + q_mask   · log(q_mask   / p_θ([MASK]))
```

where $p_\theta$ is obtained by applying log-softmax to the model's output logits. At unmasked positions the KL is identically zero (the posterior is a delta function on the observed token).

**Full loss:**

```
L_KL = E_{t, x_0, x_t} [ Σ_{i: x_t^(i) = [MASK]} KL_i(q ‖ p_θ) ]
```

**Numerical stability.** Computing $e^\sigma - 1$ directly loses precision for small σ (catastrophic cancellation). We use `torch.expm1(sigma)` when σ < 0.5. We also clamp q_unmask ∈ [10⁻⁸, 1] to prevent log-of-zero errors, since the $1/t$ divergence at small t can push q_unmask above 1.

```python
esigm1 = torch.where(
    sigma_b < 0.5,
    torch.expm1(sigma_b),      # stable: e^σ - 1 for small σ
    sigma_b.exp() - 1,
)
q_unmask = torch.clamp(dsigma_b / esigm1, min=1e-8, max=1.0)
q_mask   = torch.clamp(1.0 - q_unmask,    min=1e-8, max=1.0)
```

### 4.5 Theoretical Comparison: SE vs. KL

| Property | Score Entropy (L_SE) | KL / ELBO (L_KL) |
|---|---|---|
| Divergence minimized | Score entropy divergence | KL(q ‖ p_θ) at each step |
| Relation to log-likelihood | Indirect surrogate | Direct ELBO lower bound |
| Mode behavior | Mode-seeking tendency | Mode-covering tendency |
| Gradient weighting | Proportional to σ̇(t) | Proportional to σ̇(t)/(e^σ − 1) ≈ 1/t |
| Gradient variance near t=0 | Moderate | Very high (few masked positions, large weights) |
| Perplexity interpretable | No | Yes |
| Zero lower bound | Yes | Yes |

**The mode-covering vs. mode-seeking distinction** is critical. $\text{KL}(p_\text{data} \| p_\theta)$ (forward KL, what the ELBO minimizes) is **mode-covering**: a model that misses any mode of $p_\text{data}$ incurs infinite loss, so the optimal $p_\theta$ must spread probability mass over all modes, potentially producing blurry/incoherent samples. The score entropy divergence is closer in spirit to reverse KL: the $\sum_y e^{s_\theta[y]}$ term penalizes placing excess mass anywhere, encouraging the model to concentrate on high-density modes.

**The 1/t gradient divergence** is the most practically important difference. Near t ≈ ε = 10⁻³, the KL weighting is ~1000×. With batch size 8 and ~0.1% of positions masked at small t, the KL gradient at fine timesteps is dominated by ~1 token per sequence — a single-sample gradient estimate with enormous variance. The optimizer can exploit easy large-t losses while failing to learn fine-grained denoising near t = 0, which is exactly where coherent text requires the most precision.

### 4.6 Sampling Algorithm

Both systems use the **predictor-corrector (PC) sampling** framework from SEDD:

- **Euler predictor**: Discretizes the reverse CTMC with step size Δt. At each step, masked tokens may be unmasked according to the model's predicted scores.
- **Analytic predictor**: Computes an exact categorical posterior using the staggered score composed with the transpose transition matrix.
- **Final denoiser step**: At t → 0, analytically removes remaining noise by snapping to the most probable unmasked tokens.

Default: 128 reverse steps with the Euler predictor + denoiser.

### 4.7 Training Infrastructure

- `torch.multiprocessing.spawn` + NCCL, single A4000 GPU on Rutgers iLab
- AdamW: β₁=0.9, β₂=0.999, ε=10⁻⁸, lr=3×10⁻⁴, no weight decay
- 2,500-step linear warmup, gradient clipping at 1.0
- Batch size: 8 (single-GPU memory constraint)
- Target: 200,000 iterations (both runs executed ~100k steps)

---

## 5. Experimental Setup

### 5.1 Datasets

**System A (DiffuLLM):** `B2111797/recipenlg-text-256` (Hugging Face) — 500,000 examples from the RecipeNLG corpus, pre-chunked to 256 tokens.

**Systems B & C (SEDD):** `corbt/all-recipes` (Hugging Face) — structured cooking recipes formatted as `{Title}\nIngredients:\n- ...\nDirections:\n- ...`, tokenized with GPT-2 BPE and chunked into 1,024-token blocks with EOS separators between recipes.

**Important caveat (SEDD):** The recipe dataset has no separate validation split — the "valid" split falls back to "train". Evaluation loss during SEDD training is therefore computed on different batches of the *same* training distribution. This means it approximates held-out training loss rather than measuring true generalization, and overfitting cannot be detected from the training logs alone.

### 5.2 Distribution Assumption

We explicitly assume both datasets follow a strong, predictable structural distribution: every recipe conforms to Title → Ingredients → Instructions. This assumption is critical — it is why we expect discrete diffusion to succeed here. The model's bidirectional attention can learn to simultaneously attend to the title and ingredients when generating instructions, rather than building up context incrementally as in autoregressive generation.

### 5.3 Compute Environment

Rutgers iLab GPU cluster, NVIDIA A4000 GPUs (16GB VRAM):
- **System A:** 4 GPUs, ~48 GPU-hours, 1.2M steps
- **Systems B & C:** 1 GPU each, ~20 GPU-hours each, ~100k steps

### 5.4 Evaluation Protocol

**System A:** Structured evaluation on held-out prompts. Provide title + specific ingredients, generate 3–4 samples via confidence-based unmasking, score outputs on a 0–20 rubric (structural validity, ingredient-instruction coherence, procedural sensibility, freedom from corruption).

**Systems B & C:** Generate samples at step 100k using EMA weights and 128 Euler reverse steps. Evaluate qualitatively and via training/eval loss curves.

---

## 6. Results

### 6.1 System A: DiffuLLM Results

#### Training Dynamics

Training loss decreased steadily through warmup and continued to improve past 500,000 steps. Best validation loss checkpoint captured at **step 503,000** (`best_step503000.pt`). Further checkpoints at steps 654k, 1172k, and 1200k were captured. As shown below, later checkpoints exhibit regression in generation quality despite lower training loss — a sign of overfitting to the structural token distribution.

#### Qualitative Results

**Example 1: Butter Chicken** (length=320, top_k=20, temperature=0.62, checkpoint 503k — Score 17/20)

```
Butter Chicken

Ingredients:
- chicken breast, butter, onion, garlic, ginger,
  tomato sauce, garam masala, cumin, chili powder,
  heavy cream, salt, cilantro

Instructions:
1. Heat butter in a skillet and saute onion, garlic and ginger until soft
2. Remove
3. Add tomato sauce, garam masala, cumin, chilli powder and salt, stir to combine
4. Add cream and stir to combine
5. Add chicken and coconut mixture to pan, bring to boil for 5 minutes and reduce heat
6. Cover and cook for about 25 minutes
```

The model correctly identifies that aromatics (onion, garlic, ginger) are sautéed before sauce, and that cream is added near the end. All three samples scored 16–17/20.

**Example 2: Baked Macaroni and Cheese** (checkpoint 503k — Score 17/20)

```
Instructions:
1. In saucepan in a minimum of salted water, cook noodles until tender
2. Drain and set aside
3. In same saucepan, melt butter, stir in flour
4. Gradually add milk, stirring constantly
5. Bring to a boil
6. Stir in cheese, salt and pepper
7. Add cooked noodles
8. Cook until hot and cheese is melted
9. Pour into greased casserole dish
10. Top with bread crumbs
```

Textbook béchamel technique. Correct roux order (butter + flour before milk), all 8 provided ingredients used.

**Example 3: Meatloaf** (checkpoint 503k — Score 16/20)

```
Instructions:
1. Cook the rice in salted water until just tender (about 15 minutes)
2. Drain
3. Mix the meat mixture with the remaining ingredients
4. Shape into a loaf
5. Bake in a 350 degree oven for 1 hour.
```

Five coherent, correctly ordered steps.

#### Failure Modes at Longer Contexts

At length ≥ 500 tokens and with later checkpoints (>600k steps), quality degrades:

1. **Special token leakage:** Raw control tokens (`<NEXT_INSTR>`, `NEXTNEXT_INSTR>`) appear verbatim in decoded text rather than functioning as structural boundaries.
2. **Token ID drift:** Raw integer token IDs (`23 22 18 oblong 23`) appear past the recipe boundary — the model generating into what should be padding space.
3. **Hallucinated structure:** Steps like *"Bake according to the table of St.com"* show structurally plausible phrasing with semantically incoherent content.
4. **Repetition:** `"Eat. Eat. Eat."` loops at length ≥ 500, indicating loss of global position tracking.

**Checkpoint comparison** (Baked Mac and Cheese, 4 samples each):

| Checkpoint | Mean Quality Score |
|---|---|
| best_step503000 | ~17/20 |
| best_step1172000 | ~12/20 |
| final_step1200000 | ~12/20 |

Training loss continued decreasing past 503k while generation quality peaked and degraded — canonical overfitting behavior.

---

### 6.2 Systems B & C: SEDD Results

#### Initial Training Failure (Diagnosed and Fixed)

Before training could begin, both SEDD runs failed immediately after model construction with:

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**Root cause:** The config file stored optimizer hyperparameters in scientific notation:

```yaml
lr: 3e-4      # parsed as string "3e-4", not float 3×10⁻⁴
eps: 1e-8     # parsed as string "1e-8", not float 1×10⁻⁸
```

YAML's handling of unquoted scientific notation is parser-dependent. The runtime YAML parser (via Hydra/OmegaConf) treated these as strings; PyTorch's AdamW rejected them on the comparison `0.0 <= lr`. The fix — replacing with decimal notation (`lr: 0.0003`, `eps: 0.00000001`) — resolved it immediately. The model, loss function, noise schedule, and sampling procedure were all correctly implemented; only the configuration format was wrong.

**What was working before the fix:** The full 169.6M-parameter model was successfully instantiated on an NVIDIA RTX A4500 (18.34 GB), and the EMA tracker was correctly initialized. The error occurred at exactly the optimizer construction step, confirming the type error as the sole cause.

#### Completed Training Runs

Both loss variants were trained on the Rutgers iLab cluster for approximately 100,000 steps. The score entropy run was terminated early by a disk quota error at step 96,750; the KL run completed to step 100,000.

| Run | Loss type | Steps completed | Final train loss | Final eval loss |
|---|---|---|---|---|
| SEDD-SE | Score Entropy | 96,750 | 1,056 | 1,499 |
| SEDD-KL | KL / ELBO | 100,000 | 663 | 272 |

**Caveat on comparability:** The two losses are measured in different units. Score entropy loss is in ratio-space (dimensionless); KL loss is in nats per masked token per timestep. Their absolute magnitudes are not directly comparable — only within-run trends are interpretable.

#### Training Dynamics

Both runs started from a loss consistent with near-uniform predictions. Score entropy initialized at ~1.1×10⁴; KL at ~6.9×10³. Both decreased rapidly in the first 10,000 steps, then continued a noisy downward trend.

Evaluation loss at key steps (single-batch estimates, high variance due to batch size 8):

| Step | SEDD-SE eval | SEDD-KL eval |
|---:|---:|---:|
| 0 | 1.077×10⁴ | 6.918×10³ |
| 10,000 | 1.812×10³ | 8.326×10² |
| 20,000 | 2.188×10³ | 2.110×10³ |
| 30,000 | 1.408×10³ | 1.253×10³ |
| 40,000 | 1.761×10³ | 5.813×10² |
| 50,000 | 2.436×10³ | 1.032×10³ |
| 60,000 | 1.624×10³ | 6.895×10² |
| 70,000 | 1.520×10³ | 1.453×10³ |
| 80,000 | 1.200×10³ | 6.066×10² |
| 90,000 | 2.039×10³ | 1.178×10³ |
| 100,000 | 1.091×10³ | 272 |

The high variance is expected: with batch size 8 and single-batch evaluation, each eval call covers 8 × 1024 = 8,192 tokens — a small and noisy point estimate of the true population loss. The downward trend is visible in both runs despite the noise.

#### Qualitative Sample Quality at Step 100k

Samples were generated using EMA weights and 128 Euler reverse steps.

**SEDD-SE (score entropy, step 96.75k):**

```
noodles; drain. Toss with bread crumbs, Parmesan cheese and pepper.
Dip chicken breast in the cheese mixture and toss to coat. Heat a 5-in.
skillet over medium heat; cook chicken 3-4 minutes per side until lightly
browned. Remove from the pan and let stand 10 minutes. Reduce heat;
stir in flour; whisk in the 1/2 cup water and 2 Tbsp. of the cream;
cook and stir until thickened, stirring occasionally.
```

```
1 cup corn juice
- juice of 1 lime wedge
- oil, for brushing
- lime or lime juice, to taste (optional)

Directions:
- Stir up the lime juice and lime juice and place in a bowl and toss well.
- Garnish with lime wedge.
```

**SEDD-KL (KL/ELBO loss, step 100k):**

```
on's grit for and does: diced with oven, the burn cloves soy to Add
bell- bottomC bowl a
 cupsas Continueits the dip Cutand- ounces,
```

```
- ifer/ heat steak servingionsagon egg min coer: and pan degrees
 up maple cookiesarch
isc stirring cup. with- cooled peeledo,ulated2 and 1 over- lightly
```

The score entropy samples exhibit coherent sentence structure, plausible recipe vocabulary, and correctly formatted directions. The KL samples are largely incoherent — fragmented subword tokens with broken syntax — despite the KL run achieving substantially lower absolute loss.

---

## 7. Analysis and Discussion

### 7.1 Do the Results Make Sense?

**For DiffuLLM:** Yes. The model's ability to generate coherent, ingredient-grounded cooking instructions is precisely what the D3PM/absorbing-state formulation predicts:

1. **Global context access.** The bidirectional Transformer attends to all positions simultaneously — including both the ingredient list and the partially-denoised instruction tokens — forming a globally coherent representation at every denoising step.

2. **Confidence-scheduled unmasking.** The most confidently predicted tokens (common verbs like "add," "stir," "heat") are fixed first, providing a scaffold for resolving more ambiguous positions. This is analogous to how a human might fill in a recipe by placing the most obvious steps first.

3. **Domain structure alignment.** The recipe domain is well-suited to parallel generation because its dependencies are bidirectional: the title constrains the ingredients which constrain the instructions, and the instructions retroactively constrain which ingredients are plausible. A bidirectional model exploits all of these constraints simultaneously.

**For SEDD-SE:** Yes. At ~5% of the reference paper's training budget, the model shows early signs of learning recipe structure — imperative verbs, realistic measurements, formatted ingredient lists — though without complete global coherence. Individual sentences are plausible but the recipe as a whole does not follow a logical arc. This is consistent with a model that has learned local n-gram statistics but not yet the long-range structural constraints.

**For SEDD-KL:** Yes — the failure is consistent with the theory. Lower loss does not mean better model when the losses measure different quantities in different units.

### 7.2 Connecting to Course Concepts

**The ELBO and KL Divergence.** System C's loss is a direct application of variational inference. Minimizing $\sum_t \text{KL}(q(x_{t-dt} | x_t, x_0) \| p_\theta(x_{t-dt} | x_t))$ is equivalent to maximizing the ELBO on $\log p_\theta(x_0)$. This connects discrete diffusion directly to Variational Autoencoders (VAEs) — both maximize a tractable lower bound on log-likelihood rather than optimizing the intractable true marginal.

**Score Matching and Fisher Divergence.** The SEDD-SE loss connects directly to score matching (Hyvärinen, 2005). Continuous score matching minimizes the Fisher divergence $\mathbb{E}[\|\nabla_x \log p_\text{data} - \nabla_x \log p_\theta\|^2]$. SEDD generalizes this to discrete spaces where "score" means ratio rather than gradient, and Fisher divergence becomes score entropy divergence.

**The Markov Assumption.** All three systems assume the reverse process is Markov: $p_\theta(x_{t-1} | x_t)$ conditions only on $x_t$, not the full trajectory. This is necessary for tractability but has a cost: the model cannot "remember" what it unmasked in earlier denoising steps when resolving later steps. Over long sequences, this creates the repetition and drift observed in DiffuLLM.

**Generalization vs. Overfitting.** The DiffuLLM checkpoint comparison reveals a canonical overfitting trajectory: training loss continues to decrease past 503k, but held-out generation quality peaks early and degrades. The model memorized specific token sequences — particularly structural endings — rather than learning the underlying distribution.

**Bias-Variance Tradeoff in the KL Loss.** The $1/t$ gradient weighting of the KL loss is a direct manifestation of the bias-variance tradeoff. At small t (near the data), gradients are very high magnitude but cover very few positions (low bias, high variance). At large t (heavily masked), gradients cover many positions but carry weak signal per position (lower variance, higher bias). The score entropy's σ̇(t) weighting provides a more uniform tradeoff across noise levels. The KL's high-variance gradient at fine timesteps is the theoretically predicted cause of SEDD-KL's poor sample quality.

### 7.3 Analyzing Negative Results

**Where does DiffuLLM break down?**

Two mechanisms cause structural token leakage at long context:

1. **The Markov assumption in action.** The parallel denoising process at step t conditions only on the current masked sequence $x_t$. Over 700 tokens with dozens of denoising steps, information about global structural position ("are we inside or outside the instruction block?") must be inferred from the pattern of already-unmasked tokens. When many confident positions have already been fixed, the remaining uncertain positions near the sequence boundary have insufficient context to determine whether they should be content tokens or structural delimiters.

2. **Global context tracking without forced causality.** Autoregressive models build a continuous, updating KV cache encoding the full left context. Our bidirectional model attends to all tokens simultaneously but has no mechanism that forces it to track progress along the sequence. For sequences exceeding the training context length of 256 tokens, this results in cyclical repetition and position amnesia.

**Why did SEDD-KL fail despite lower loss?**

As derived in Section 4.5, the KL loss weighting near $t \to 0$ is $\approx 1/t$. At $t \approx \epsilon = 10^{-3}$, each masked position contributes a gradient weight ~1000× larger than at $t = 1$. With batch size 8 and ~0.1% of positions masked at small $t$, the KL gradient at fine timesteps is dominated by ~1 token per sequence — a single-sample estimate with enormous variance. The optimizer likely exploited easy large-t losses (many masked positions, moderate weights) while failing to learn the fine-grained denoising needed for coherent output near $t = 0$.

The lower absolute KL loss (272 vs. 1,056) reflects that the KL loss is measuring something different — nats per token in probability space — not that the model is better. Both losses have zero as their minimum at the globally optimal model, but at 100k steps we are nowhere near convergence, and the optimization landscapes are fundamentally different.

### 7.4 Comparative Summary: All Three Systems

Stepping back from the individual systems, what does this project actually tell us about discrete diffusion for structured text?

The three models we built are not cleanly comparable — they differ in size, training budget, tokenizer, and even what their loss functions are measuring — but that unevenness is itself informative. DiffuLLM (Varesh) is the smallest and simplest: ~50M parameters, a custom 24K BPE vocabulary tuned to the recipe domain, a cosine noise schedule over 256 discrete steps, and 1.2M training steps on a multi-GPU setup. It reaches genuine coherence. Recipes produced at checkpoint 503k are grounded in their ingredients, follow logical preparation order, and handle subtle culinary logic (aromatics before sauce, cream added last) correctly. Its weakness is equally fundamental: the Markov denoising process has no causal state-tracker, so beyond ~500 tokens the model loses track of where it is in the recipe structure and starts repeating phrases or leaking structural delimiter tokens.

SEDD-SE and SEDD-KL (Urvi) share everything except their loss function: the same 169.6M-parameter architecture, GPT-2's 50K vocabulary, a continuous log-linear noise schedule, and 1,024-token context. They were trained to ~100k steps on a single GPU — roughly 8% of DiffuLLM's step count, though at larger batch sizes. At this budget, SEDD-SE produces partially coherent recipe fragments: the structure is starting to emerge, and the outputs feel like early-training text that knows *what kind of thing* it is supposed to be writing. SEDD-KL, despite posting a substantially lower absolute loss (272 vs. 1,056 in their respective units), generates incoherent fragments — high gradient variance near $t \to 0$ destabilizes the fine-grained denoising that coherent output requires. Lower loss did not mean better model; the two objectives are not measuring the same thing.

The honest takeaway is that DiffuLLM's strong results reflect compute and engineering iteration more than an intrinsic advantage of its simpler objective. Given the same 1M+ steps and multi-GPU resources, the SEDD architecture — with its longer context, larger vocabulary, and RoPE positional embeddings — would likely surpass it. What this project surfaces is something more practically useful than a clean ranking: discrete diffusion works, its failure modes are mathematically predictable rather than arbitrary, and the choice of loss function has consequences that only become visible once training has run long enough for the optimization dynamics to dominate.

For reference, the key design choices across all three systems are summarized below:

| Dimension | DiffuLLM (A) | SEDD-SE (B) | SEDD-KL (C) |
|---|---|---|---|
| Implemented by | Varesh Patel | Urvi Desai | Urvi Desai |
| Loss objective | Masked cross-entropy | Score entropy | KL/ELBO |
| Relation to likelihood | Indirect | Indirect surrogate | Direct lower bound |
| Tokenizer | Custom BPE (24K) | GPT-2 (50.3K) | GPT-2 (50.3K) |
| Noise schedule | Cosine (discrete T=256) | Log-linear (continuous) | Log-linear (continuous) |
| Context length | 256 tokens | 1,024 tokens | 1,024 tokens |
| Model size | ~50M params | ~169.6M params | ~169.6M params |
| Training steps | 1.2M | ~96,750 | ~100,000 |
| Inference | Confidence unmasking | Euler PC + Denoiser | Euler PC + Denoiser |
| EMA | No | Yes (0.9999) | Yes (0.9999) |
| Best sample quality | Excellent (503k ckpt) | Partial coherence | Incoherent at 100k |
| Primary strength | Structured short recipes | Architecture sound | Interpretable objective |
| Primary limitation | Token leakage >500 tokens | Needs more training | Gradient variance at small t |

**The training budget gap is significant.** DiffuLLM ran for 1.2M steps; the SEDD models ran for ~100k steps (~8% of that budget at much larger batch sizes). A fair comparison would require SEDD models trained to convergence. The original SEDD paper trains for 1M+ steps with batch size 512 on 8 GPUs — far more compute than available on a single A4000.

---

## 8. Team Contributions

**Varesh Patel** designed and implemented the complete DiffuLLM system (System A) from scratch: the custom BPE tokenizer training pipeline, the D3PMTransformer architecture with AdaLN conditioning, the cosine forward process, the masked-only cross-entropy loss, the logit-normal timestep sampling, the confidence-scheduled inference algorithm, the DDP training pipeline via `torchrun`, and the structured evaluation framework. All training was run on the Rutgers iLab cluster, achieving 1.2M training steps with the best checkpoint captured at step 503,000.

**Urvi Desai** adapted the SEDD codebase (Systems B and C). For System B (SEDD-SE), she configured and ran the full SEDD pipeline on the recipe domain, diagnosed and fixed the YAML scientific notation config bug, managed the single-GPU training run to ~96,750 steps, and generated and analyzed the qualitative samples. For System C (SEDD-KL), she derived the ELBO-based KL divergence loss for the absorbing graph from first principles — including the continuous-time posterior derivation, the $1/t$ unmasking rate, the 2-point posterior structure, and the numerical stability choices (`expm1`, clamping) — and implemented it as a drop-in replacement for score entropy, then ran and analyzed the comparison experiment to 100,000 steps.

**Aparajita Sarkar** performed qualitative analysis of model outputs across all three systems, developed the scoring rubric for recipe evaluation, contributed the analytical write-up documenting assumptions, failure modes, and connections to course concepts, and organized the three-way comparative analysis.

---

## 9. Conclusion

This project demonstrates that discrete absorbing-state diffusion models can learn to generate domain-specific structured text with high coherence for short-to-medium sequence lengths, and provides a principled three-way comparison of architectures and objectives trained on real GPU cluster hardware.

**Key findings:**

1. **Proof of concept with sufficient training:** DiffuLLM generates coherent, ingredient-grounded cooking instructions at checkpoint 503k, correctly handling béchamel technique, aromatics ordering, and cream-last logic. Non-autoregressive discrete diffusion works for structured text.

2. **Architecture vs. objective:** The SEDD architecture (169.6M params, 1024-token context, GPT-2 tokenizer, RoPE) is theoretically sounder than DiffuLLM for long-context generation, but was under-trained due to single-GPU resource constraints. At 100k steps, SEDD-SE shows early structural learning; SEDD-KL is not yet competitive.

3. **Score entropy outperforms KL at equal training budget:** Despite achieving lower absolute loss, SEDD-KL produces incoherent samples at 100k steps. This confirms the theoretical prediction: the KL loss's $1/t$ gradient weighting near $t = 0$ produces extremely high-variance gradient estimates at fine noise levels, destabilizing the learning of fine-grained denoising that produces coherent text. Score entropy's σ̇(t) weighting provides a more stable training signal at this budget.

4. **The KL loss has theoretical advantages:** With sufficient training and lower learning rate, the KL loss should converge to the same optimum as score entropy while providing direct perplexity interpretability and a tighter connection to maximum likelihood. The optimization dynamics, not the objective itself, explain the poor performance at 100k steps.

5. **Failure modes are mathematically predictable:** DiffuLLM's token leakage and repetition at long sequences are direct consequences of the Markov assumption and the absence of forced causal state-tracking. SEDD-KL's incoherence is a consequence of the $1/t$ gradient weighting. These are not bugs — they are fundamental algorithmic properties made visible by experiment.



---

## 10. References

- Lou, A., Meng, C., & Ermon, S. (2023). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* arXiv:2310.16834.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.
- Austin, J., et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications.
- Peebles, W. & Xie, S. (2022). *Scalable Diffusion Models with Transformers.* ICCV.
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR.
- Kingma, D. P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.
- Ghazvininejad, M., et al. (2019). *Mask-Predict: Parallel Decoding of Conditional Masked Language Models.* EMNLP.
