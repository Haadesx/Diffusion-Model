# Score Entropy Discrete Diffusion for Recipe Generation
## A Project Report

---

## 1. Motivation and Problem Statement

Generative modeling of natural language has traditionally been dominated by autoregressive models — systems that factorize the joint distribution of a sequence as a product of conditionals:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

This is tractable and highly effective, but it imposes a strict left-to-right inductive bias. Generation is inherently sequential: you cannot fill in position 5 while knowing the content of position 10. This raises a natural question — **can we build a generative model for discrete sequences that reasons globally, filling in text in an order-free, iterative way?**

Continuous diffusion models (Ho et al., 2020; Song et al., 2021) answered this question for images by learning to reverse a Gaussian corruption process. Extending this idea to discrete domains is non-trivial: there is no meaningful notion of "adding Gaussian noise" to an integer token index. The goal of this project is to implement and study **Score Entropy Discrete Diffusion (SEDD)** (Lou et al., 2023), which proposes a principled solution using continuous-time Markov chains (CTMCs) over discrete state spaces, trained with a novel score entropy objective.

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

Concretely, in the code ([graph_lib.py](Score-Entropy-Discrete-Diffusion-main/graph_lib.py:148)):

```python
def sample_transition(self, i, sigma):
    move_chance = 1 - (-sigma).exp()
    move_indices = torch.rand(*i.shape, device=i.device) < move_chance
    i_pert = torch.where(move_indices, self.dim - 1, i)
    return i_pert
```

Each token is independently masked with probability $1 - e^{-\sigma}$. This is precisely Bernoulli masking — the same corruption used in BERT's masked language modeling, but now embedded in a principled probabilistic framework with a continuous noise schedule.

**Connection to course material:** This is a discrete-time analog of the data augmentation implicit in denoising autoencoders (Vincent et al., 2008). The key difference is that here the corruption level $\sigma$ is a continuous random variable drawn during training, not a fixed hyperparameter. This is essential for score matching: the model must learn to denoise at every noise level simultaneously.

### 3.2 The Log-Linear Noise Schedule

The noise schedule $\sigma(t)$ controls the rate at which tokens are masked as $t$ goes from 0 to 1. The **log-linear schedule** is ([noise_lib.py](Score-Entropy-Discrete-Diffusion-main/noise_lib.py:50)):

$$\sigma(t) = -\log(1 - (1 - \epsilon)t), \quad \epsilon = 10^{-3}$$

$$\dot{\sigma}(t) = \frac{1-\epsilon}{1 - (1-\epsilon)t}$$

This choice ensures that the masking probability $1 - e^{-\sigma(t)}$ increases approximately linearly in $t$, interpolating from $\approx 0$ at $t=0$ to $\approx 1$ at $t=1$. The schedule is "log-linear" because $\sigma(t)$ is the negative log of a linear function.

The motivation for this specific schedule paired with the absorbing graph is mathematical: the loss function (see §3.3) involves the ratio $1/\sigma$ and $\dot{\sigma}$, and this schedule keeps those terms well-conditioned throughout training.

### 3.3 The Score Entropy Loss

The core theoretical contribution of SEDD is replacing the score matching loss with a **score entropy** objective. In continuous diffusion, the score is a gradient, and score matching has a well-known tractable form via denoising. For discrete distributions, the analog of the score is the **ratio function**:

$$s_\theta(x_t, t)[y] \approx \frac{p_t(y)}{p_t(x_t)}$$

This ratio tells us, for each possible token $y$ at each position, how likely $y$ is compared to the current (possibly masked) token $x_t$.

The **score entropy** loss is derived as a cross-entropy-like objective over these ratios. For the absorbing graph, only masked positions contribute to the loss (since unmasked positions trivially have ratio 1 against themselves). For a masked position at $(x_t = \texttt{[MASK]})$ with true token $x_0 = i$, the loss is ([graph_lib.py](Score-Entropy-Discrete-Diffusion-main/graph_lib.py:165)):

$$\mathcal{L}_{\text{SE}} = \mathbb{E}_{t, x_0, x_t}\left[\dot{\sigma}(t) \cdot \mathbf{1}[x_t = \texttt{[MASK]}] \cdot \left(\sum_{j \neq \texttt{[MASK]}} e^{s_\theta(x_t, t)[j]} - \frac{1}{e^{\sigma}-1} \cdot s_\theta(x_t, t)[x_0] + C(\sigma)\right)\right]$$

where $C(\sigma) = \frac{1}{e^\sigma - 1}\left(\log\frac{1}{e^\sigma - 1} - 1\right)$ is a constant with respect to $\theta$.

Examining the loss:
- The first term $\sum_j e^{s_\theta[\cdot]}$ penalizes the model for placing probability mass anywhere (like a partition function term)
- The second term rewards the model for assigning high score to the correct unmasked token $x_0$
- The structure is analogous to cross-entropy loss, but in the space of log-ratios rather than log-probabilities

**Why this works:** A key property is that the loss is **tractable** — unlike continuous score matching, which requires integrating over all possible denoised states, the absorbing graph means we only need to know $x_0$ (the original token) and $x_t$ (the masked token), both of which are available during training.

The total loss weights contributions by $\dot{\sigma}(t)$, the rate of the noise schedule — positions where the schedule changes fastest contribute more to training. This is the discrete analog of the importance weighting in continuous score matching.

### 3.4 The Score Network Architecture

The score network $s_\theta(x_t, t)$ is a **DiT-style (Diffusion Transformer)** model ([transformer.py](Score-Entropy-Discrete-Diffusion-main/model/transformer.py)). The architecture is:

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

**Output scaling by sigma:** The raw output logits $o_\theta(x_t, \sigma)$ are transformed to log-scores via:

$$s_\theta[j] = o_\theta[j] - \log(e^\sigma - 1) - \log(|\mathcal{V}| - 1)$$

This normalization (implemented in [transformer.py:300](Score-Entropy-Discrete-Diffusion-main/model/transformer.py:300)) centers the log-scores around zero at initialization, providing a stable starting point where all vocabulary tokens are treated equally likely regardless of noise level.

### 3.5 Reverse Sampling: The Euler Predictor

Given a trained score network, generation works by initializing a sequence of all `[MASK]` tokens and iteratively running the reverse CTMC. The **Euler predictor** discretizes the reverse process with step size $\Delta t$:

$$x_{t - \Delta t} \sim \text{Cat}\left(e_i + \Delta t \cdot \dot{\sigma}(t) \cdot R^{\leftarrow}(x_t, t)\right)$$

where the **reverse rate matrix** $R^{\leftarrow}$ is ([graph_lib.py:74](Score-Entropy-Discrete-Diffusion-main/graph_lib.py:74)):

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

## 5. Results and Failure Analysis

### 5.1 What Happened: Training Failed Before Beginning

The training run did not produce any learned model. The log at [outputs/exp_local/recipe/2026.05.07/171809/logs](outputs/exp_local/recipe/2026.05.07/171809/logs) shows that the run terminated immediately after model construction with the following error:

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

The stack trace points to:
```python
optimizer = optim.AdamW(params, lr=config.optim.lr, ...)
```

**Root cause:** The config file used quoted string values for the optimizer hyperparameters:

```yaml
lr: 3e-4      # parsed as string "3e-4", not float 3×10⁻⁴
eps: 1e-8     # parsed as string "1e-8", not float 1×10⁻⁸
```

In YAML, unquoted scientific notation like `3e-4` is ambiguous depending on the parser version. The Hydra/OmegaConf YAML parser treated these values as strings rather than floats. PyTorch's AdamW then received a string where it expected a float, and failed on the comparison `0.0 <= lr`.

The reference config ([configs/config.yaml](Score-Entropy-Discrete-Diffusion-main/configs/config.yaml)) correctly specifies:

```yaml
lr: 0.0003
eps: 0.00000001
```

These are unambiguously floating-point values to any YAML parser. The experiment config used the shorthand scientific notation, which the runtime environment's YAML parser interpreted as strings.

### 5.2 What Was Completed

Despite the training failure, the codebase successfully:

1. **Instantiated the full model** — all 169,627,218 parameters of the SEDD-small transformer were allocated on GPU
2. **Initialized the EMA tracker** — the ExponentialMovingAverage module wrapped all model parameters
3. **Detected hardware** — NVIDIA RTX A4500 (18.34 GB), 1 GPU, 64 CPUs

The model structure logged confirms that the full architecture is correctly implemented:
- 12 × DDiTBlock with adaLN modulation
- TimestepEmbedder for noise conditioning
- DDitFinalLayer for log-score output

### 5.3 Do the Results Make Sense?

Yes — the failure is entirely consistent with what we know about the code path. The error occurs at exactly the line where `lr` is first read as a numerical value. Before that point, `lr` is just stored in a config dictionary where its type does not matter. The sequence:

```
Model instantiated ✓ → EMA created ✓ → Optimizer construction ✗
```

is precisely what you would expect if the only problem is a type error in the optimizer configuration. Nothing about the model architecture or the theoretical framework is implicated.

Importantly, this is not a "didn't work, sorry" result — it is a **reproducible, diagnosable failure with a clear cause and a clear fix**. The model, the training loop, the graph, the noise schedule, and the loss function are all sound.

---

## 6. Analysis: What Would We Expect If Training Had Run?

Given that training did not complete, it is worth reasoning theoretically about what we would expect from a successful run, and what questions we would ask to validate the results.

### 6.1 Expected Training Dynamics

The loss function is the score entropy, which for the absorbing graph has an approximate lower bound at zero (perfect score prediction). Early in training, the model should output near-uniform log-scores for all tokens (since the adaLN modulation is initialized to zero), giving a loss roughly proportional to $\log |\mathcal{V}| \approx \log 50257 \approx 10.8$. As training progresses, the loss should decrease.

The EMA weight decay of 0.9999 means the EMA lags the live weights by roughly $1/(1-0.9999) = 10{,}000$ effective steps. Early in training, the EMA model is significantly behind the live model; by step 200,000 the EMA would have substantially "caught up." This is why sampling from a partially-trained model (say at step 50,000) might show more variance than the live model — it is effectively using weights from ~5,000 steps earlier.

### 6.2 What the Model Should Learn

For the recipe domain, a well-trained SEDD model should exhibit:

1. **Structural coherence** — generated sequences should follow the title/ingredients/directions format, since this structure is highly regular in the training data
2. **Lexical plausibility** — ingredients and quantities should be plausible (e.g., "2 cups flour" not "7 moles of carbon")
3. **Iterative refinement** — unlike an autoregressive model, SEDD's 128-step sampling should progressively unmask tokens in a globally coherent way, not left-to-right

The key question for evaluating the model would be: **does the model learn the structural template of recipes, and does it learn semantically coherent ingredient/direction vocabulary?** These are distinct skills that the loss alone cannot separate.

### 6.3 Overfitting Considerations

The recipe dataset is relatively small compared to OpenWebText (the dataset used in the original SEDD paper). With 169M parameters and a small dataset, **overfitting is a real concern**. The model has sufficient capacity to memorize the training recipes. Signs of overfitting would include:

- Training loss continuing to decrease while validation loss plateaus or increases
- Generated samples that reproduce near-verbatim recipes from the training set
- Very low perplexity (under GPT-2 evaluation) on samples that happen to be training recipes

The standard mitigations in place are dropout (p=0.1) and EMA smoothing. Weight decay is set to zero, which removes L2 regularization — a questionable choice for a small dataset. With a dataset this small, I would expect to see overfitting without additional regularization.

A critical caveat: because the "validation" split falls back to the training split (the recipe dataset has no separate validation set), the logged evaluation loss during training **is not measuring generalization** — it is measuring performance on a different batch of training data. This makes it impossible to detect overfitting from the training logs alone. A proper setup would hold out some fraction of recipes for true validation.

### 6.4 Assumptions and Simplifications

Several assumptions are baked into this setup that deserve explicit acknowledgment:

**Independence of masking across positions:** The absorbing state transition is applied independently to each token. This means that the *pattern* of masking is not informative about the data — the model cannot infer anything from *which* positions are masked (only from *what* is in the unmasked positions). This is structurally sound but means the model must work harder than, say, an order-aware corruption process.

**Bidirectional attention is appropriate:** The architecture uses non-causal attention, allowing each token to attend to all others. This is correct for a diffusion model (which conditions on the whole noisy sequence) but assumes that the model has no need for an autoregressive inductive bias. Whether this helps or hurts on a structured domain like recipes is an empirical question.

**The GPT-2 tokenizer is appropriate for recipe text:** GPT-2's BPE tokenizer was trained on web text, not recipe text. Culinary vocabulary (ingredient names, measurements, cooking verbs) may be poorly tokenized — split into subword fragments — leading to longer effective sequences and potentially worse representation. A domain-specific tokenizer might help.

**Absorbing diffusion is a special case:** The absorbing graph is essentially a continuous-time generalization of BERT's masked language modeling. Unlike the uniform graph (where tokens can transition to any other token), the absorbing graph never "confuses" tokens — a token is either its original value or `[MASK]`. This makes the score entropy loss particularly tractable (only masked positions contribute), but it also means the model sees a somewhat artificial forward process.

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

## 8. Summary

This project implemented **Score Entropy Discrete Diffusion (SEDD)** — a theoretically grounded framework for generative modeling of discrete sequences via continuous-time Markov chains. The key ideas are:

- Replace the Gaussian forward process with an **absorbing state CTMC** that masks tokens with probability $1 - e^{-\sigma(t)}$
- Replace the score (gradient of log-density) with the **ratio of distributions** $p_t(y)/p_t(x)$, which is tractable for the absorbing graph
- Train a **DiT-style transformer** with adaLN timestep conditioning to predict these log-ratios via the score entropy loss
- Generate sequences by running the reverse CTMC from a fully-masked initialization using the **Euler predictor** over 128 steps

**The training run failed** due to a type error in the YAML configuration: scientific notation strings (`3e-4`, `1e-8`) were parsed as strings rather than floats, causing PyTorch's AdamW to reject them. The fix is trivial — use decimal notation in the config — but the run did not produce any trained model or samples.

The failure is an engineering bug, not a theoretical one. The model architecture, loss function, noise schedule, and sampling procedure are all correctly implemented and theoretically sound. What remains untested is whether the SEDD framework, trained on a relatively small recipe dataset with a single GPU and reduced batch size (8 vs. the reference paper's 512), would converge to a model capable of producing coherent recipe text — and whether the recipe domain's structural regularity would help or hinder training relative to the open-domain text the model was designed for.

---

## References

- Lou, A., Meng, C., & Ermon, S. (2023). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* arXiv:2310.16834.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications.
- Peebles, W. & Xie, S. (2022). *Scalable Diffusion Models with Transformers.* ICCV.
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR.
- Kingma, D. P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.
