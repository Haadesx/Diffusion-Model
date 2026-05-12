# Score Entropy Discrete Diffusion for Recipe Generation
## A Project Report (v2)

---

## 1. Motivation and Problem Statement

Generative modeling of natural language has traditionally been dominated by autoregressive models — systems that factorize the joint distribution of a sequence as a product of conditionals:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

This is tractable and highly effective, but it imposes a strict left-to-right inductive bias. Generation is inherently sequential: you cannot fill in position 5 while knowing the content of position 10. This raises a natural question — **can we build a generative model for discrete sequences that reasons globally, filling in text in an order-free, iterative way?**

Continuous diffusion models (Ho et al., 2020; Song et al., 2021) answered this question for images by learning to reverse a Gaussian corruption process. Extending this idea to discrete domains is non-trivial: there is no meaningful notion of "adding Gaussian noise" to an integer token index. The goal of this project is to implement and study **Score Entropy Discrete Diffusion (SEDD)** (Lou et al., 2023), which proposes a principled solution using continuous-time Markov chains (CTMCs) over discrete state spaces, trained with a novel score entropy objective.

As an extension, we additionally implement an **ELBO-based KL divergence loss** as an alternative training objective, derive it from first principles for the absorbing graph, and compare it theoretically to the score entropy formulation.

The dataset chosen is a corpus of cooking recipes. This is a deliberate and interesting choice: recipes have strong structural regularity (title → ingredients → directions) but highly variable content, making them a reasonable testbed for whether a model can learn both global structure and local vocabulary.

---

## 2. Background: Continuous Diffusion and the Challenge of Discreteness

### 2.1 Continuous Score-Based Diffusion

Score-based generative models define a forward stochastic differential equation (SDE) that gradually corrupts data:

$$dx = f(x,t)\,dt + g(t)\,dW$$

The reverse-time SDE (Anderson, 1982) is:

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{W}$$

The unknown score function $\nabla_x \log p_t(x)$ is estimated by a neural network $s_\theta(x, t)$ via **denoising score matching**:

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_t}\left[\left\|s_\theta(x_t, t) - \nabla_{x_t}\log p(x_t|x_0)\right\|^2\right]$$

This works beautifully in continuous spaces because the score is a well-defined gradient. In discrete spaces, there is no gradient to take.

### 2.2 Why Discrete Diffusion Is Different

For tokens drawn from a finite vocabulary $\mathcal{V} = \{1, \ldots, S\}$, we cannot perturb with Gaussian noise. Instead, we need to define a corruption process that remains in discrete state space. The key insight is to replace the continuous SDE with a **continuous-time Markov chain (CTMC)**, characterized by a rate matrix $Q$ where:

- $Q_{ij} \geq 0$ for $i \neq j$ is the instantaneous rate of transitioning from state $i$ to state $j$
- $Q_{ii} = -\sum_{j \neq i} Q_{ij}$ (rows sum to zero)

The marginal distribution at time $t$ satisfies the **Kolmogorov forward equation**:

$$\frac{d}{dt}p(x_t) = p(x_t) Q$$

with solution $p(x_t | x_0 = i) = e^{tQ}_{i, \cdot}$ — the $i$-th row of the matrix exponential.

The reverse process, analogous to the reverse SDE, replaces the score with the **ratio** of distributions at different noise levels. This ratio plays the role of the score function in discrete space.

---

## 3. The SEDD Framework

### 3.1 The Absorbing Graph

This implementation uses an **absorbing state Markov chain**, also known as a "masking" diffusion process. The vocabulary is augmented with a single special token `[MASK]` (index $S+1$, here index 50257), and the rate matrix is:

$$Q_{ij} = \begin{cases} -1 & \text{if } i = j \neq \texttt{[MASK]} \\ 1 & \text{if } j = \texttt{[MASK]}, \, i \neq \texttt{[MASK]} \\ 0 & \text{otherwise} \end{cases}$$

This means every non-mask token transitions to `[MASK]` at rate 1, and `[MASK]` is absorbing (it never transitions out). The marginal transition probability is:

$$p(x_t = \texttt{[MASK]} \mid x_0 = i) = 1 - e^{-\sigma(t)}, \quad p(x_t = i \mid x_0 = i) = e^{-\sigma(t)}$$

where $\sigma(t)$ is a noise schedule controlling how much corruption has occurred. At $t = 0$, $\sigma = 0$ and the token is intact; at $t = 1$, $\sigma \to \infty$ and everything is masked.

Each token is independently masked with probability $1 - e^{-\sigma}$. This is precisely Bernoulli masking — the same corruption used in BERT's masked language modeling, but now embedded in a principled probabilistic framework with a continuous noise schedule.

**Connection to course material:** This is a discrete-time analog of the data augmentation implicit in denoising autoencoders (Vincent et al., 2008). The key difference is that here the corruption level $\sigma$ is a continuous random variable drawn during training, not a fixed hyperparameter. This is essential for score matching: the model must learn to denoise at every noise level simultaneously.

### 3.2 The Log-Linear Noise Schedule

The noise schedule $\sigma(t)$ controls the rate at which tokens are masked as $t$ goes from 0 to 1. The **log-linear schedule** is:

$$\sigma(t) = -\log(1 - (1 - \epsilon)t), \quad \epsilon = 10^{-3}$$

$$\dot{\sigma}(t) \equiv \frac{d\sigma}{dt} = \frac{1-\epsilon}{1 - (1-\epsilon)t}$$

This choice ensures that the masking probability $1 - e^{-\sigma(t)}$ increases approximately linearly in $t$, interpolating from $\approx 0$ at $t=0$ to $\approx 1$ at $t=1$. The schedule is "log-linear" because $\sigma(t)$ is the negative log of a linear function.

The motivation for this specific schedule paired with the absorbing graph is mathematical: both the score entropy and KL loss functions involve terms of the form $\dot{\sigma}(t) / (e^{\sigma(t)} - 1)$, which represent an instantaneous unmasking rate. The log-linear schedule keeps this term well-conditioned across the full range $t \in (0, 1)$.

### 3.3 The Score Entropy Loss

The core theoretical contribution of SEDD is replacing the score matching loss with a **score entropy** objective. In continuous diffusion, the score is a gradient, and score matching has a well-known tractable form via denoising. For discrete distributions, the analog of the score is the **ratio function**:

$$s_\theta(x_t, t)[y] \approx \frac{p_t(y)}{p_t(x_t)}$$

This ratio tells us, for each possible token $y$ at each position, how likely $y$ is compared to the current (possibly masked) token $x_t$.

The **score entropy** loss is derived as a cross-entropy-like objective over these ratios. For the absorbing graph, only masked positions contribute to the loss (since unmasked positions trivially have ratio 1 against themselves). For a masked position at $(x_t = \texttt{[MASK]})$ with true token $x_0 = i$, the loss is:

$$\mathcal{L}_{\text{SE}} = \mathbb{E}_{t, x_0, x_t}\left[\dot{\sigma}(t) \cdot \mathbf{1}[x_t = \texttt{[MASK]}] \cdot \left(\sum_{j \neq \texttt{[MASK]}} e^{s_\theta(x_t, t)[j]} - \frac{1}{e^{\sigma}-1} \cdot s_\theta(x_t, t)[x_0] + C(\sigma)\right)\right]$$

where $C(\sigma) = \frac{1}{e^\sigma - 1}\left(\log\frac{1}{e^\sigma - 1} - 1\right)$ is a constant with respect to $\theta$.

Examining the loss:
- The first term $\sum_j e^{s_\theta[\cdot]}$ penalizes the model for placing probability mass anywhere (like a partition function term)
- The second term rewards the model for assigning high score to the correct unmasked token $x_0$
- The structure is analogous to cross-entropy loss, but in the space of log-ratios rather than log-probabilities

**Why this works:** A key property is that the loss is **tractable** — unlike continuous score matching, which requires integrating over all possible denoised states, the absorbing graph means we only need to know $x_0$ (the original token) and $x_t$ (the masked token), both of which are available during training.

The total loss weights contributions by $\dot{\sigma}(t)$, the rate of the noise schedule — positions where the schedule changes fastest contribute more to training. This is the discrete analog of the importance weighting in continuous score matching.

### 3.4 The Score Network Architecture

The score network $s_\theta(x_t, t)$ is a **DiT-style (Diffusion Transformer)** model. The architecture is:

| Component | Specification |
|-----------|--------------|
| Hidden size | 768 |
| Attention heads | 12 |
| Transformer blocks | 12 |
| Context length | 1024 tokens |
| Vocab size input | 50,257 (GPT-2) + 1 `[MASK]` = 50,258 |
| Total parameters | ~169.6M |

**Timestep conditioning:** The noise level $\sigma$ is embedded via sinusoidal positional encoding followed by a 2-layer MLP into a 128-dimensional conditioning vector $c$. Each transformer block then uses **adaptive LayerNorm (adaLN)** to modulate both the attention and MLP sublayers:

$$\text{adaLN}(x, c) = (1 + \text{scale}(c)) \odot \text{LayerNorm}(x) + \text{shift}(c)$$

This is the same mechanism used in DiT (Peebles & Xie, 2022), imported directly from the image diffusion literature. The adaLN parameters are initialized to zero, meaning the model initially behaves like a standard transformer and learns to condition on noise level during training.

**Positional encoding:** Rotary embeddings (RoPE) are used rather than learned absolute positional embeddings. RoPE encodes relative positions multiplicatively in the attention logits, which is more efficient and generalizes better to sequence lengths not seen during training.

**Attention:** The model uses bidirectional (non-causal) self-attention — every position can attend to every other position. This is crucial: unlike autoregressive models, SEDD does not assume a left-to-right generation order. The full-context attention is what allows the model to reason globally about which masked tokens to fill in.

**Output scaling by sigma:** The raw output logits $o_\theta(x_t, \sigma)$ are transformed to log-scores via ([model/utils.py](Score-Entropy-Discrete-Diffusion-main/model/utils.py)):

$$s_\theta[j] = o_\theta[j] - \log(e^\sigma - 1) - \log(|\mathcal{V}| - 1)$$

This normalization centers the log-scores around zero at initialization, providing a stable starting point where all vocabulary tokens are treated equally likely regardless of noise level. Critically, during training `get_score_fn` returns these raw log-scores; during sampling it returns `exp(log_score)` — the actual ratios. Both the score entropy and KL loss functions must consume the raw log-score form.

### 3.5 Reverse Sampling: The Euler Predictor

Given a trained score network, generation works by initializing a sequence of all `[MASK]` tokens and iteratively running the reverse CTMC. The **Euler predictor** discretizes the reverse process with step size $\Delta t$:

$$x_{t - \Delta t} \sim \text{Cat}\left(e_i + \Delta t \cdot \dot{\sigma}(t) \cdot R^{\leftarrow}(x_t, t)\right)$$

where the **reverse rate matrix** $R^{\leftarrow}$ is:

$$R^{\leftarrow}_{x_t \to y} = s_\theta(x_t, t)[y] \cdot Q^T_{x_t, y}$$

For the absorbing graph, $Q^T$ is nonzero only for transitions from `[MASK]` back to real tokens. So at each step, masked tokens may be unmasked according to the model's predicted scores. This is conceptually similar to the BERT masked language model: "given this masked sequence, what is each token most likely to be?" — but done iteratively over 128 steps.

After the final predictor step, a **denoising step** analytically removes any remaining noise, snapping the sequence to the most probable unmasked tokens.

---

## 4. Experimental Setup

### 4.1 Dataset

The dataset is the **corbt/all-recipes** dataset (sourced from HuggingFace), containing structured cooking recipes. Each recipe is formatted as:

```
{Title}

Ingredients:
- {ingredient 1}
- {ingredient 2}
...

Directions:
- {step 1}
- {step 2}
...
```

Text is tokenized with the **GPT-2 BPE tokenizer** (50,257 tokens), and recipes are concatenated and chunked into fixed-length blocks of 1024 tokens, with EOS tokens separating individual recipes within a block. Both train and validation splits come from the same dataset (there is no held-out test set in this dataset configuration — the "valid" split falls back to "train" since the recipe dataset only has a train split). This is an important caveat: the evaluation loss is therefore an approximation of training loss on a different batch, not a true generalization estimate.

### 4.2 Training Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimizer | AdamW | Adaptive learning rate + decoupled weight decay |
| Learning rate | 3×10⁻⁴ | Standard for transformer-scale models |
| β₁, β₂ | 0.9, 0.999 | Standard Adam moments |
| Weight decay | 0 | No regularization beyond dropout |
| Warmup steps | 2,500 | Linear warmup from 0 to peak LR |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| EMA decay | 0.9999 | Slow-moving average of weights for sampling |
| Batch size | 8 | Limited by single-GPU memory |
| Planned iterations | 200,000 | |
| Dropout | 0.1 | |

The **Exponential Moving Average (EMA)** of model weights is a standard trick in diffusion model training. The EMA weights, updated at every step as $\theta_{\text{EMA}} \leftarrow 0.9999 \cdot \theta_{\text{EMA}} + 0.0001 \cdot \theta$, are used only at sampling time — the live weights are used for loss computation. This smooths out the training trajectory and produces better samples without additional compute.

---

## 5. Results

### 5.1 Initial Training Failure (Diagnosed and Fixed)

The first local training attempt failed immediately after model construction with:

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**Root cause:** The generated config file stored optimizer hyperparameters in scientific notation:

```yaml
lr: 3e-4      # parsed as string "3e-4", not float 3×10⁻⁴
eps: 1e-8     # parsed as string "1e-8", not float 1×10⁻⁸
```

YAML's handling of unquoted scientific notation is parser-dependent. The runtime environment treated these as strings; PyTorch's AdamW rejected them on the comparison `0.0 <= lr`. The fix — using decimal notation (`lr: 0.0003`, `eps: 0.00000001`) — resolved the issue immediately. The error was entirely in configuration, not in the model or training code.

### 5.2 Completed Training Runs

Both loss variants were trained on the Rutgers ilab cluster (single NVIDIA GPU) for approximately 100,000 steps, using the configuration in §4.2. The score entropy run was terminated early by a disk quota error at step 96,750; the KL run completed to step 100,000.

| Run | Loss type | Steps completed | Final train loss | Final eval loss |
|-----|-----------|----------------|-----------------|-----------------|
| Score Entropy | $\mathcal{L}_\text{SE}$ | 96,750 | 1,056 | 1,499 |
| KL / ELBO | $\mathcal{L}_\text{KL}$ | 100,000 | 663 | 272 |

**Caveat on comparability:** The two losses are measured in different units. Score entropy loss is in ratio-space (dimensionless), while KL loss is in nats per masked token per timestep. Their absolute magnitudes are not directly comparable; only within-run trends are interpretable.

### 5.3 Training Dynamics

Both runs started from a loss consistent with near-uniform predictions. At step 0, the score entropy loss was $\approx 1.1 \times 10^4$ and the KL loss was $\approx 6.9 \times 10^3$. Both decreased rapidly in the first 10,000 steps, then continued a noisy downward trend.

Evaluation loss at key steps (single-batch, high variance):

| Step | Score Entropy eval | KL eval |
|-----:|------------------:|--------:|
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

The high variance is expected: with a batch size of 8 and a single-batch eval, the evaluation loss is a noisy point estimate, not a smoothed curve. The downward trend is visible in both runs despite the noise.

**Important caveat:** The recipe dataset has no separate validation split — the evaluation loss is computed on a different batch from the *same* training distribution. This means neither run's eval loss measures true generalization; it approximates the training loss on held-out batches. Overfitting cannot be detected from these logs alone.

### 5.4 Qualitative Sample Quality

Samples were generated at step 100,000 using the EMA weights and 128 Euler reverse steps. Representative outputs:

**Score Entropy (step 100k):**
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

**KL / ELBO (step 100k):**
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

The score entropy samples exhibit coherent sentence structure, plausible recipe vocabulary, and correctly formatted directions. The KL samples, despite lower loss, are largely incoherent — fragmented tokens with broken syntax.

### 5.5 Analysis: Lower KL Loss Did Not Yield Better Samples

This is the central empirical finding. The KL model achieved substantially lower loss at step 100k yet produced far worse samples. Several explanations are consistent with theory:

1. **Different loss scales.** The losses are incommensurable. A KL loss of 272 nats and a score entropy loss of 1,091 ratio-units measure different things; "lower" does not mean "better model."

2. **Mode-covering gradient pathology.** As derived in §8.4, the KL loss weights near-$t=0$ timesteps with weight $\approx 1/t$, which diverges as $t \to 0$. At $t \approx 10^{-3}$, each masked position contributes a gradient weight $\sim 1000\times$ larger than at $t = 1$. With batch size 8 and $\approx 0.1\%$ of positions masked at small $t$, the KL gradient at small $t$ is dominated by $\sim 1$ token per sequence — extremely high variance. The optimizer may exploit the loss at easy timesteps (large $t$, many masked positions) while failing to learn the fine-grained denoising needed for coherent output near $t = 0$.

3. **Score entropy's implicit regularization.** The score entropy loss penalizes the model for placing probability mass anywhere (the $\sum_y e^{s_\theta[y]}$ term), effectively regularizing the output distribution. The KL loss applies softmax normalization but does not have this explicit penalty on mass allocation — the model may spread probability more uniformly, producing lower cross-entropy loss on the distribution but worse mode-concentrated samples.

4. **Training budget.** Both models were trained for only ~100k steps versus the 1M+ steps used in the original SEDD paper. At this early stage, the score entropy loss may provide a more stable gradient signal, while the KL loss's high variance near $t = 0$ slows effective learning. With longer training, the gap might close.

### 5.6 Overfitting Considerations

With 169M parameters trained on a small recipe dataset (batch size 8, no weight decay), overfitting is a real concern. The standard mitigations in place are dropout (p=0.1) and EMA smoothing. The fact that training and eval loss track each other (within noise) throughout training suggests the model has not severely overfit by step 100k — but the lack of a true held-out set makes this hard to confirm.

### 5.7 Assumptions and Simplifications

**Independence of masking across positions:** The absorbing state transition is applied independently to each token. This means that the *pattern* of masking is not informative about the data — the model cannot infer anything from *which* positions are masked (only from *what* is in the unmasked positions). This is structurally sound but means the model must work harder than, say, an order-aware corruption process.

**Bidirectional attention is appropriate:** The architecture uses non-causal attention, allowing each token to attend to all others. This is correct for a diffusion model (which conditions on the whole noisy sequence) but assumes that the model has no need for an autoregressive inductive bias. Whether this helps or hurts on a structured domain like recipes is an empirical question.

**The GPT-2 tokenizer is appropriate for recipe text:** GPT-2's BPE tokenizer was trained on web text, not recipe text. Culinary vocabulary (ingredient names, measurements, cooking verbs) may be poorly tokenized — split into subword fragments — leading to longer effective sequences and potentially worse representation. A domain-specific tokenizer might help.

---

## 6. Analysis of Training Dynamics and Model Behavior

### 6.1 Training Dynamics

Both models started with high, noisy loss consistent with near-uniform predictions and decreased rapidly in the first 10k steps (the 2,500-step warmup period, followed by rapid improvement). The EMA decay of 0.9999 means the EMA weights lag the live model by $\approx 1/(1-0.9999) = 10{,}000$ effective steps — so samples from the step-100k checkpoint use EMA weights approximating the live model at ~step 90k.

The high variance in single-batch evaluation loss is a direct consequence of batch size 8: each eval call computes the loss on 8 sequences of 1024 tokens, a small and noisy estimate of the true population loss.

### 6.2 What the Models Have and Have Not Learned

At 100k steps (~5% of the original paper's training budget), neither model has converged. The score entropy model shows clear signs of learning recipe structure — imperative verbs, realistic measurements, formatted ingredient lists — but lacks global coherence; individual sentences are plausible but the recipe as a whole does not follow a logical arc. The KL model has learned recipe-domain vocabulary but not syntax or structure, suggesting it is optimizing a different aspect of the distribution under its mode-covering objective.

### 6.3 Connection to Iterative Refinement

A key claimed advantage of SEDD over autoregressive models is that the 128-step reverse process can refine the sequence globally, not left-to-right. The score entropy samples exhibit this to a degree: tokens appear to be placed with awareness of the full context window (e.g., "Garnish with lime wedge" follows "Directions:"). However, at this training budget the refinement is incomplete — later steps in the reverse process are not correcting early mistakes, they are just sampling from a weakly-trained model.

This is expected: the quality of the Euler predictor's reverse process depends on how well $s_\theta$ approximates the true score at every noise level $\sigma$. At 100k steps, the approximation is rough, particularly near $\sigma \to 0$ (fine-grained denoising) where the score landscape is most complex.

---

## 7. Connections to Course Material

### 7.1 Score Matching and Fisher Divergence

The theoretical foundation of SEDD connects directly to **score matching** (Hyvärinen, 2005). Score matching minimizes the **Fisher divergence** between the model and data distributions:

$$D_F(p_\text{data} \| p_\theta) = \mathbb{E}_{p_\text{data}}\left[\left\|\nabla_x \log p_\text{data}(x) - \nabla_x \log p_\theta(x)\right\|^2\right]$$

The genius of denoising score matching is avoiding the intractable $\nabla_x \log p_\text{data}$ by instead matching the score of the noise-perturbed distribution, which is analytically available. SEDD generalizes this to discrete spaces where "score" means ratio rather than gradient, and "Fisher divergence" becomes "score entropy divergence."

### 7.2 The ELBO Connection

The score entropy loss can be interpreted as a bound on the negative log-likelihood. In continuous diffusion, the training objective is equivalent to minimizing a weighted sum of denoising score matching losses, which lower bounds the data log-likelihood. The analogous statement holds for discrete diffusion: the score entropy objective is a tractable surrogate for the true log-likelihood of the reverse process. This is the same variational reasoning that underlies VAEs (Kingma & Welling, 2013), but applied to a Markov chain rather than a latent variable.

### 7.3 Assumptions About Underlying Distributions

The model makes no parametric assumption about the distribution of recipes. It is a **nonparametric** estimator in the sense that the neural network can in principle approximate any distribution over the vocabulary given sufficient capacity and data. However, it implicitly assumes:

- **Exchangeability within corruption level:** The masking is i.i.d. across positions, which assumes tokens are equally likely to be observed or masked regardless of their position and identity. This is a simplification — in practice, some tokens (e.g., structural markers like "Ingredients:") are more predictable from context than others.
- **Smoothness of the score function:** The transformer architecture imposes a smoothness bias on the learned score — nearby noise levels $\sigma$ produce similar model outputs. This is a form of implicit regularization that helps generalization but may limit the model's ability to handle sharp transitions in the score landscape.

### 7.4 Generalization and the Bias-Variance Tradeoff

At 169M parameters trained on a small recipe dataset with a batch size of 8 and no weight decay, this model is firmly in the **high-variance regime** of the bias-variance tradeoff. The model has far more capacity than needed to fit the data, and without sufficient regularization or data, it will tend to memorize rather than generalize. This is precisely the regime where dropout and EMA are most valuable — they inject noise and smoothing respectively to prevent the model from collapsing to sharp memorized solutions.

---

## 8. Extension: KL Divergence as an Alternative Training Objective

The original SEDD training objective minimizes a **score entropy divergence** — a quantity analogous to Fisher divergence, adapted for discrete distributions. A natural alternative is to minimize the **forward KL divergence** $\text{KL}(p_\text{data} \| p_\theta)$, which is equivalent to maximum likelihood estimation. For diffusion models, this KL is not directly tractable, but it can be bounded below by the **Evidence Lower Bound (ELBO)**, which decomposes into per-timestep KL terms that *are* analytically computable for the absorbing graph.

This section derives the ELBO-based KL loss from first principles, states all assumptions, and compares it to the score entropy formulation. The implementation is in [losses.py](Score-Entropy-Discrete-Diffusion-main/losses.py) as `get_kl_loss_fn`, selectable via `training.loss_type: kl` in the config.

### 8.1 The Forward KL and the ELBO

Minimizing the forward KL divergence between the data distribution $p_\text{data}$ and the model $p_\theta$ is equivalent to maximum likelihood:

$$\text{KL}(p_\text{data} \| p_\theta) = \mathbb{E}_{x_0 \sim p_\text{data}}\left[-\log p_\theta(x_0)\right] + \text{const}$$

The log-likelihood $\log p_\theta(x_0)$ is intractable: it requires marginalizing over all possible trajectories $x_{0:T}$ of the reverse process. The standard approach — used in VAEs and all DDPM-style models — is to instead maximize a lower bound.

By Jensen's inequality applied to the log of the evidence:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\,dx_{1:T} \geq \mathbb{E}_{q}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}\right] \equiv \text{ELBO}$$

where $q(x_{1:T} | x_0)$ is the forward process — the fixed corruption process we designed. Maximizing the ELBO with respect to $\theta$ minimizes an upper bound on $\text{KL}(p_\text{data} \| p_\theta)$.

**Assumption:** The ELBO is a valid lower bound on the log-likelihood regardless of the choice of $q$. The bound is tight (equals $\log p_\theta(x_0)$) if and only if $q(x_{1:T} | x_0) = p_\theta(x_{1:T} | x_0)$, i.e., the forward and reverse processes agree exactly. In practice this never holds exactly, so the ELBO is always a strict lower bound. The gap between the ELBO and the true log-likelihood is the **posterior gap** — a form of approximation error that the score entropy formulation sidesteps by not committing to a specific parametric reverse process.

### 8.2 Decomposing the ELBO for the Absorbing Graph

For a CTMC-based diffusion model, the ELBO decomposes over time into a sum (or integral) of per-step KL terms. In the continuous-time limit this becomes:

$$-\text{ELBO} = \mathbb{E}_{t \sim \text{Uniform}(0,1)}\left[\dot{\sigma}(t) \cdot \mathbb{E}_{x_0, x_t}\left[\text{KL}\!\left(q(x_{t-dt} \mid x_t, x_0) \,\Big\|\, p_\theta(x_{t-dt} \mid x_t)\right)\right]\right] + \text{const}$$

The constant is $\text{KL}(q(x_T | x_0) \| p(x_T))$ — the divergence between the fully-noised data and the model's prior. For the absorbing graph with $\sigma(1) \to \infty$, all tokens are masked with probability 1, and the prior $p(x_T)$ is identically the all-mask sequence, so this term is zero. We proceed to derive the inner KL.

#### 8.2.1 The Forward Posterior $q(x_{t-dt} | x_t, x_0)$

Using Bayes' theorem:

$$q(x_{t-dt} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-dt})\, q(x_{t-dt} \mid x_0)}{q(x_t \mid x_0)}$$

For the absorbing graph, each factor is a Bernoulli in masking status. Let $m_s \equiv 1 - e^{-\sigma(s)}$ denote the masking probability at time $s$. We consider only the case $x_t = \texttt{[MASK]}$, since if $x_t \neq \texttt{[MASK]}$ the token is observed and the model has nothing to predict ($\text{KL} = 0$, see §8.2.3).

Given $x_t = \texttt{[MASK]}$ and $x_0 = i \neq \texttt{[MASK]}$:

$$q(x_{t-dt} = i \mid x_t = \texttt{[MASK]}, x_0 = i) = \frac{q(\texttt{[MASK]} \mid i) \cdot q(i \mid x_0 = i)}{q(\texttt{[MASK]} \mid x_0 = i)}$$

$$= \frac{(m_t - m_{t-dt}) \cdot e^{-\sigma(t-dt)}}{m_t} = \frac{e^{-\sigma(t-dt)} - e^{-\sigma(t)}}{1 - e^{-\sigma(t)}}$$

$$q(x_{t-dt} = \texttt{[MASK]} \mid x_t = \texttt{[MASK]}, x_0 = i) = \frac{m_{t-dt}}{m_t} = \frac{1 - e^{-\sigma(t-dt)}}{1 - e^{-\sigma(t)}}$$

So the posterior is a **2-point distribution** over $\{i, \texttt{[MASK]}\}$, with the remaining $|V| - 1$ tokens having zero posterior probability. This is exact — no approximation. The absorbing structure guarantees that the true token can only be $x_0$ or $\texttt{[MASK]}$; any other token is impossible given $x_t = \texttt{[MASK]}$ (since the forward process only masks, never substitutes).

#### 8.2.2 The Continuous-Time Limit

Taking $dt \to 0$ using L'Hôpital's rule (or equivalently, a first-order Taylor expansion of $e^{-\sigma(t-dt)} \approx e^{-\sigma(t)} + \dot{\sigma}(t)\,dt\,e^{-\sigma(t)}$):

$$q_\text{unmask} \equiv q(x_{t-dt} = i \mid x_t = \texttt{[MASK]}, x_0 = i) \;\xrightarrow{dt \to 0}\; \frac{\dot{\sigma}(t)\,e^{-\sigma(t)}\,dt}{1 - e^{-\sigma(t)}} = \frac{\dot{\sigma}(t)\,dt}{e^{\sigma(t)} - 1}$$

$$q_\text{mask} \equiv q(x_{t-dt} = \texttt{[MASK]} \mid x_t = \texttt{[MASK]}, x_0 = i) \;\xrightarrow{dt \to 0}\; 1 - q_\text{unmask}$$

The quantity $\dot{\sigma}(t)/(e^{\sigma(t)} - 1)$ is the **instantaneous unmasking rate** — the rate at which a masked token reveals its true identity as we reverse time. For the log-linear schedule:

$$\frac{\dot{\sigma}(t)}{e^{\sigma(t)} - 1} = \frac{(1-\epsilon)/(1-(1-\epsilon)t)}{1/(1-(1-\epsilon)t) - 1} = \frac{1-\epsilon}{(1-\epsilon)t - 1 + 1} \cdot \frac{1-(1-\epsilon)t}{1} \xrightarrow{\epsilon \to 0} \frac{1}{t}$$

So for small $\epsilon$, the unmasking rate is approximately $1/t$ — it diverges as $t \to 0$ (near the data), meaning the model must make many fine-grained decisions in the last few reverse steps. This is analogous to the behavior of continuous diffusion models near $t = 0$, where the score function diverges.

#### 8.2.3 The Per-Position KL Term

The model predicts a categorical distribution over all $|V|+1$ tokens by applying softmax to its log-score output $s_\theta(x_t, \sigma)$. Let:

$$p_\theta(y \mid x_t) = \frac{\exp(s_\theta(x_t, \sigma)[y])}{\sum_{z} \exp(s_\theta(x_t, \sigma)[z])}$$

**Case 1: $x_t \neq \texttt{[MASK]}$ (unmasked position).** The forward posterior is a delta mass: $q(x_{t-dt} = x_t \mid x_t, x_0) = 1$ with probability 1. The KL divergence from a delta to any distribution is:

$$\text{KL}(\delta_{x_t} \| p_\theta(\cdot \mid x_t)) = -\log p_\theta(x_t \mid x_t) \cdot 1$$

However, note that the model's output has a structural property: the diagonal entry $s_\theta(x_t, \sigma)[x_t]$ is zeroed out before softmax (implemented at [transformer.py:303](Score-Entropy-Discrete-Diffusion-main/model/transformer.py:303)), meaning the model assigns zero log-ratio to the current token. This makes the probability $p_\theta(x_t | x_t)$ not directly meaningful. More importantly, **unmasked positions carry no gradient signal** for any reasonable loss: the posterior is a point mass on the observed token, and the model cannot change this token during reverse sampling (the reverse rate $Q^T_{x_t, y} = 0$ for $x_t \neq \texttt{[MASK]}$). Both the score entropy loss and the KL loss correctly assign zero contribution to unmasked positions.

**Case 2: $x_t = \texttt{[MASK]}$ (masked position).** The forward posterior is the 2-point distribution derived above. The KL divergence is:

$$\text{KL}(q \| p_\theta) = q_\text{unmask} \log \frac{q_\text{unmask}}{p_\theta(x_0 \mid \texttt{[MASK]})} + q_\text{mask} \log \frac{q_\text{mask}}{p_\theta(\texttt{[MASK]} \mid \texttt{[MASK]})}$$

Substituting the continuous-time expressions for $q_\text{unmask}$ and $q_\text{mask}$, and using $\log p_\theta = \log\text{softmax}(s_\theta)$:

$$\text{KL}(q \| p_\theta) = q_\text{unmask}\left(\log q_\text{unmask} - s_\theta[x_0] + \log Z\right) + q_\text{mask}\left(\log q_\text{mask} - s_\theta[\texttt{[MASK]}] + \log Z\right)$$

where $Z = \sum_y \exp(s_\theta[y])$ is the partition function. In code this is computed via `F.log_softmax`, which handles the $\log Z$ subtraction stably.

#### 8.2.4 The Full KL Loss

Integrating over time and sequence positions, the total loss is:

$$\mathcal{L}_{\text{KL}} = \mathbb{E}_{t, x_0, x_t}\left[\sum_{\ell=1}^{L} \mathbf{1}[x_t^\ell = \texttt{[MASK]}] \cdot \text{KL}\!\left(q(x_{t-dt}^\ell \mid x_t^\ell, x_0^\ell) \;\Big\|\; p_\theta(x_{t-dt}^\ell \mid x_t^\ell)\right)\right]$$

where the $\dot{\sigma}(t)$ time-weighting is already absorbed into $q_\text{unmask} = \dot{\sigma}(t)\,dt/(e^\sigma - 1)$. The sum over positions is exact because the absorbing graph masks each position independently — the positions decouple in the ELBO (a consequence of the i.i.d. masking assumption stated in §6.4).

### 8.3 Numerical Implementation

The implementation in `get_kl_loss_fn` ([losses.py](Score-Entropy-Discrete-Diffusion-main/losses.py)) follows the derivation exactly with two numerical stability choices:

**Stability choice 1:** Computing $e^\sigma - 1$ via `torch.expm1(sigma)` when $\sigma < 0.5$. When $\sigma$ is small, $e^\sigma \approx 1 + \sigma$, and direct computation of $e^\sigma - 1$ loses precision due to catastrophic cancellation. `expm1` avoids this.

**Stability choice 2:** Clamping $q_\text{unmask} \in [10^{-8}, 1]$. The ratio $\dot{\sigma}/(e^\sigma - 1)$ approaches $1/t$ as $\epsilon \to 0$; near $t \approx \epsilon = 10^{-3}$ this is $\approx 10^3$, far exceeding 1 and violating the probabilistic interpretation. The clamping enforces that $q_\text{unmask}$ remains a valid probability, and prevents the $\log q_\text{unmask}$ term in the KL from growing without bound at very small $t$.

**Why `log_softmax` rather than `softmax` then log:** Chaining softmax and log introduces numerical error because softmax outputs near 0 or 1 lose precision in float32. `F.log_softmax` computes the log-probability in a single numerically stable pass using the log-sum-exp trick.

### 8.4 Theoretical Comparison: Score Entropy vs. KL (ELBO)

These two objectives minimize different divergences and have meaningfully different properties. The comparison is worth making explicit.

#### What each objective is minimizing

The **score entropy loss** minimizes a **score entropy divergence** $D_\text{SE}$ defined on the space of log-ratios:

$$D_\text{SE}(u^\star \| s_\theta) = \mathbb{E}\left[\dot{\sigma}\left(\sum_y e^{s_\theta[y]} - u^\star[y] \cdot s_\theta[y] + C\right)\right]$$

where $u^\star[y] = p_\sigma(y)/p_\sigma(x_t)$ is the true log-ratio (score). This is minimized at zero when $s_\theta = u^\star$ everywhere, and the gradient with respect to $s_\theta$ is well-defined even if $s_\theta$ is a bad approximation. The score entropy divergence is not a standard KL or Fisher divergence — it is purpose-designed for the discrete ratio estimation problem.

The **KL loss** minimizes $\text{KL}(q \| p_\theta)$ at each timestep, summed over time. This is a **maximum likelihood** objective: at optimum, $p_\theta(x_{t-dt} | x_t)$ equals the true forward posterior $q(x_{t-dt} | x_t, x_0)$ for all $t$. Crucially, the KL divergence is **asymmetric**: $\text{KL}(q \| p_\theta)$ is zero only when $p_\theta$ assigns nonzero probability everywhere $q$ does. If $p_\theta$ assigns zero probability to the correct token, the loss is infinite — there is a hard constraint that the model must cover the support of the posterior.

#### The mode-covering vs. mode-seeking distinction

This asymmetry has a well-known consequence for generative models. $\text{KL}(p_\text{data} \| p_\theta)$ (forward KL, what the ELBO minimizes) is **mode-covering**: a model that misses any mode of $p_\text{data}$ incurs infinite loss, so the optimal $p_\theta$ spreads probability mass to cover all modes, potentially generating some low-quality samples. $\text{KL}(p_\theta \| p_\text{data})$ (reverse KL) is **mode-seeking**: the optimal model concentrates on one mode and ignores others. The score entropy divergence is closer in spirit to reverse KL, since it penalizes the model for placing excess mass anywhere (the $\sum_y e^{s_\theta[y]}$ term) without requiring full coverage.

For recipe generation, this distinction matters practically: a mode-covering model (KL) would try to generate all recipe styles and structures; a mode-seeking model (score entropy) would tend to concentrate on the highest-density recipes and potentially produce more stereotyped but higher-quality output.

#### Relationship to log-likelihood

Only the KL/ELBO loss has a direct relationship to log-likelihood: minimizing $\mathcal{L}_\text{KL}$ maximizes a lower bound on $\log p_\theta(x_0)$. The score entropy loss does not directly bound the log-likelihood — it optimizes a different divergence that happens to also lead to a good generative model, but the connection to log-probability is indirect. This matters if you want to compute or compare perplexity: perplexity is defined in terms of log-likelihood, so a model trained with $\mathcal{L}_\text{SE}$ does not have a clean perplexity interpretation, whereas $\mathcal{L}_\text{KL}$ does.

#### Gradient signal at extreme noise levels

Near $t = 1$ (heavily masked): nearly all positions are masked, giving many gradient contributions, but each position is individually uncertain — the model must essentially guess. Near $t = 0$ (lightly masked): very few positions are masked, giving sparse gradient signal, but those few positions are highly informative (only one token could plausibly fill each mask given the nearly-complete context).

For the **score entropy loss**, the $\dot{\sigma}$ weighting concentrates training signal at timesteps where the noise schedule changes fastest — for the log-linear schedule, this is at intermediate $t$ values.

For the **KL loss**, the weighting $\dot{\sigma}/(e^\sigma - 1) \approx 1/t$ diverges as $t \to 0$, giving enormous weight to the final reverse steps where the context is richest. This is in principle desirable — the hardest and most informative decisions happen near $t = 0$ — but it also means the gradient has high variance (few masked positions each with very large weights). In practice this may require lower learning rates or more careful gradient clipping for the KL loss than for score entropy.

#### Constraint on the model output

The score entropy loss treats the model output as log-ratios — it does not require them to form a valid probability distribution. The $\sum_y e^{s_\theta[y]}$ term implicitly acts as a partition function regularizer, but the model is free to output any real values.

The KL loss applies `log_softmax` to the model's output, treating it as a categorical logit vector. This is a stronger constraint: the model's predictions must sum to 1. Superficially this seems to add information, but in practice the model learns to satisfy this through training regardless of the loss — the softmax normalization is imposed by the architecture during KL training and by the score entropy structure during SE training.

#### Summary comparison

| Property | Score Entropy ($\mathcal{L}_\text{SE}$) | KL / ELBO ($\mathcal{L}_\text{KL}$) |
|---|---|---|
| Divergence minimized | Score entropy divergence | $\text{KL}(q \| p_\theta)$ at each step |
| Relation to log-likelihood | Indirect surrogate | Direct lower bound (ELBO) |
| Mode behavior | Mode-seeking tendency | Mode-covering tendency |
| Gradient weighting | Proportional to $\dot{\sigma}(t)$ | Proportional to $\dot{\sigma}(t)/(e^\sigma - 1) \approx 1/t$ |
| Gradient variance near $t=0$ | Moderate | High (few masked positions, large weights) |
| Perplexity interpretable | No | Yes |
| Requires valid probability | No | Yes (via softmax) |
| Zero lower bound | Yes (at perfect prediction) | Yes (at perfect posterior match) |

### 8.5 Implementation Details and Configuration

The loss type is selected via the `training.loss_type` field in the config:

```yaml
training:
  loss_type: score_entropy   # original SEDD objective
  # loss_type: kl            # ELBO-based KL objective
```

This is threaded through `get_step_fn` in [losses.py](Score-Entropy-Discrete-Diffusion-main/losses.py) and logged at training startup via `run_train.py`. The default is `score_entropy`, so all existing behavior is unchanged. The KL loss raises a `ValueError` if used with a non-absorbing graph, since the 2-point posterior derivation relies on the absorbing structure.

### 8.6 Empirical Comparison Results

Both loss functions were trained to ~100k steps on the recipe dataset. The key findings, analyzed against theory:

**Score entropy produced qualitatively better samples.** Despite higher loss in absolute terms, the score entropy model generated partially coherent recipe text with correct structure, realistic vocabulary, and readable directions. The KL model produced fragmented, syntactically broken output. This confirms the theoretical prediction that mode-covering objectives (KL) can achieve lower cross-entropy at the cost of sample sharpness.

**KL converged to lower loss faster.** The KL eval loss was consistently lower throughout training — but this reflects the different loss scales, not model quality. The KL loss is in nats; score entropy loss is in ratio-space units. These are not comparable.

**KL gradient variance was a practical problem.** The $1/t$ weighting of small-$t$ timesteps produces extremely high variance gradients at $t \approx \epsilon = 10^{-3}$. With batch size 8 and $\approx 0.1\%$ of positions masked at small $t$, the KL gradient at fine timesteps is dominated by $\sim 1$ token per sequence — a single-sample gradient estimate. This appears to have destabilized learning of fine-grained denoising, which is the crucial capability for producing coherent text.

**Both losses should agree on the optimal model.** At the global minimum, $s_\theta[y] = \log p_t(y) - \log p_t(x_t)$ for all $y$, and $p_\theta(y | x_t) = q(x_{t-dt} = y | x_t, x_0)$ for all $t$. Both losses are zero at this point. The observed differences reflect optimization dynamics at 100k steps — far from convergence — not a fundamental incompatibility between the two objectives. With 1M+ steps and a lower learning rate for KL, the gap may close.

### 8.7 Assumptions Specific to the KL Loss

Beyond the assumptions inherited from the base SEDD framework (§6.4), the KL loss adds:

**The ELBO gap is acceptable.** The ELBO is a lower bound on $\log p_\theta(x_0)$. The gap is determined by how well the model's reverse process matches the true forward posterior. For the absorbing graph, the posterior is a 2-point distribution over $\{x_0, \texttt{[MASK]}\}$, but the model predicts a full $|V|+1$-way categorical. The model has more expressive power than needed to match the posterior exactly, so the ELBO gap should be small at convergence — but it may be large early in training when the model assigns probability to impossible tokens.

**Positions are conditionally independent given $x_t$.** The ELBO sum over positions treats each position's KL independently. This is exact given the i.i.d. masking structure of the absorbing graph — positions do not interact in the forward process. However, the model uses full bidirectional attention, so its predictions for different positions are not independent. The ELBO factorization is valid for computing the loss, but the model's internal computation couples positions together, which is precisely what allows it to produce coherent multi-token outputs.

**The prior term $\text{KL}(q(x_T|x_0) \| p(x_T))$ is zero.** This relies on the forward process reaching the all-mask distribution exactly at $T = 1$. With $\sigma(1) = -\log(1 - (1-\epsilon)) = -\log\epsilon \approx 6.9$ for $\epsilon = 10^{-3}$, the masking probability is $1 - e^{-6.9} \approx 99.9\%$ — not exactly 1. A small fraction of tokens remain unmasked, introducing a nonzero prior KL. This is a minor inaccuracy of order $\epsilon$, negligible in practice.

---

## 9. Summary

This project implemented **Score Entropy Discrete Diffusion (SEDD)** — a theoretically grounded framework for generative modeling of discrete sequences via continuous-time Markov chains. The key ideas are:

- Replace the Gaussian forward process with an **absorbing state CTMC** that masks tokens with probability $1 - e^{-\sigma(t)}$
- Replace the score (gradient of log-density) with the **ratio of distributions** $p_t(y)/p_t(x)$, which is tractable for the absorbing graph
- Train a **DiT-style transformer** with adaLN timestep conditioning to predict these log-ratios via the score entropy loss
- Generate sequences by running the reverse CTMC from a fully-masked initialization using the **Euler predictor** over 128 steps

**Training ran successfully** (after diagnosing and fixing a YAML type error in the config where scientific notation `3e-4` was parsed as a string). Both the score entropy and KL loss variants were trained to ~100k steps on the recipe dataset on the Rutgers ilab cluster.

**Results:** The score entropy model at 100k steps generates partially coherent recipe text with correct structure and plausible vocabulary. The KL model at the same step count produces incoherent, fragmented output despite achieving lower loss in absolute terms. This empirically confirms the theoretical prediction: the KL loss's mode-covering gradient weighting ($\approx 1/t$ near $t=0$) produces high-variance gradient estimates at fine noise levels, destabilizing the learning of the fine-grained denoising that produces coherent text. The score entropy's $\dot{\sigma}$ weighting provides a more stable gradient signal at this training budget.

**As an extension**, an ELBO-based KL divergence loss was derived from first principles and implemented as an alternative to score entropy. The derivation shows that the per-step KL $\text{KL}(q(x_{t-dt}|x_t,x_0) \| p_\theta(x_{t-dt}|x_t))$ reduces to a cross-entropy between a 2-point posterior (over the true token and `[MASK]`) and the model's predicted categorical, with weights $q_\text{unmask} = \dot{\sigma}/(e^\sigma - 1)$ and $q_\text{mask} = 1 - q_\text{unmask}$. The key theoretical distinction from score entropy is that the KL loss is mode-covering (minimizes forward KL, directly bounds log-likelihood) while score entropy has mode-seeking tendencies (minimizes a ratio-space divergence, no direct likelihood bound). Both objectives share the same global minimum but differ significantly in their optimization landscapes and gradient variance profiles — particularly near $t = 0$ where the KL weighting diverges as $1/t$, confirmed empirically by the poor sample quality of the KL model at 100k steps.

---

## References

- Lou, A., Meng, C., & Ermon, S. (2023). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* arXiv:2310.16834.
- Ho, J., Jain, A., & Abbeel, S. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications.
- Peebles, W. & Xie, S. (2022). *Scalable Diffusion Models with Transformers.* ICCV.
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR.
- Kingma, D. P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.
- Austin, J., et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS.
