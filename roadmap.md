# Research Roadmap: Iterative Code Refinement via Diffusion LLMs

> Synthesized from GPT Deep Research (14 pages) + Gemini Deep Research (12 pages)

---

## The Publishable Gap (Both Reviews Agree)

**Nobody has done a controlled, compute-matched comparison of diffusion iterative refinement vs AR self-refinement for code generation.**

- **CDLM** (arXiv Dec 2025) introduced the *concept* and a Code Revision Benchmark (CRB), but only showed diffusion self-correction works — **not** that it beats AR self-refinement
- **DiffuCoder** (ICLR 2026) analyzed diffusion decoding patterns for code but focused on RL post-training, **not** head-to-head vs AR
- **ReMDM** (NeurIPS 2025) proved remasking enables inference-time compute scaling, but evaluated on general text, **not code**

### Our Paper's Thesis
> *"A 0.5B diffusion model with N refinement passes produces higher pass@k, fewer syntax errors, and more localized edits than the same 0.5B AR model with N self-refinement passes, under matched compute budgets."*

---

## Key Literature (Ordered by Relevance)

| Paper | Venue | Why It Matters |
|---|---|---|
| **CDLM** (Zhang et al.) | arXiv 2512.15596 | Defines corrective diffusion + CRB benchmark |
| **ReMDM** (Wang et al.) | NeurIPS 2025 | Principled remasking sampler for iterative refinement |
| **DiffuCoder** (Gong et al.) | ICLR 2026 | Code diffusion analysis + diffusion-native RL |
| **Dream-Coder 7B** (Xie et al.) | arXiv 2509.01142 | Emergent sketch-first code generation patterns |
| **Stable-DiffCoder** (ByteDance) | arXiv 2601.15892 | Proves diffusion CPT beats AR on code benchmarks |
| **MDLM** (Sahoo et al.) | NeurIPS 2024 | Foundation: Rao-Blackwellized masked diffusion |
| **AnCoder** (AnchorTree) | arXiv 2602.17688 | AST-guided diffusion scheduling (+9% syntax validity) |
| **Fast-dLLM** (Wu et al.) | ICLR 2026 | KV cache + parallel decoding acceleration |
| **EB-Sampler** (Ben-Hamu et al.) | arXiv 2505.24857 | Entropy-bounded adaptive unmasking |
| **Self-Refine** (Madaan et al.) | NeurIPS 2023 | AR self-refinement baseline |
| **Reflexion** (Shinn et al.) | NeurIPS 2023 | AR agentic refinement with compiler feedback |

---

## Experimental Design (Feasible on iLab A4000)

### Models (Same Base → Two Heads)
- **Shared base**: `Qwen/Qwen2.5-Coder-0.5B`
- **AR Baseline**: Fine-tune with standard causal LM objective on code
- **DLM Experimental**: Convert to MDLM using dLLM framework, train with masked diffusion objective on same code data

### Refinement Protocol
| Pass | AR Self-Refinement | Diffusion Refinement |
|---|---|---|
| 0 | Generate once | Generate once (full denoising) |
| 1 | Append error + regenerate fully | Remask low-confidence tokens → re-denoise |
| 2 | Append error + regenerate fully | Remask low-confidence tokens → re-denoise |
| 3 | Append error + regenerate fully | Remask low-confidence tokens → re-denoise |

### Critical Fairness Constraint
Report quality as a function of **wall-clock time AND NFEs (number of forward evaluations)**, not just pass@k. Diffusion refinement is cheaper per pass (only re-denoises masked positions), while AR refinement regenerates everything.

### Benchmarks
- **HumanEval+ / MBPP+** (EvalPlus) — functional correctness with expanded tests
- **CRB** (from CDLM) — code revision / error localization
- **CodeXGLUE Code Refinement** — bug fixing task

### Metrics Per Refinement Pass
1. **Syntax Error Reduction Rate (SERR)** — parsing errors per pass
2. **Test Pass Trajectory** — fraction of unit tests passed after each pass
3. **Edit Locality** — % tokens changed per pass (DLM should be surgical)
4. **AST Edit Distance** — structural code changes per pass
5. **Confidence Calibration** — does low confidence correlate with actual bugs?
6. **Wall-clock + NFEs** — compute-matched fairness

---

## 3-Phase Implementation Plan for iLab

### Phase 1: Diffusion Adaptation (~30 GPU-hours)
- Convert `Qwen2.5-Coder-0.5B` to MDLM using dLLM (we already have this pipeline!)
- Train on `bigcode/the-stack-smol` for 40k steps with LoRA
- **Already in progress** via `submit_train_coder.sh`

### Phase 2: CDLM-Inspired Post-Training (~10 GPU-hours)
- Add a correction-oriented training stage (CDLM's key insight)
- Corrupt code samples with type-preserving mutations → train model to identify and fix corrupted tokens
- This calibrates the model's confidence so low-confidence = actual error

### Phase 3: Evaluation (~5 GPU-hours)
- Run both AR and DLM on HumanEval+/MBPP+/CRB
- Log per-token confidence at each refinement pass
- Generate refinement trajectory plots (the paper's key figures)

### Total Compute: ~45 A4000 GPU-hours (< 2 days on iLab)

---

## What Makes This Paper Novel

1. **First controlled comparison** of diffusion refinement vs AR self-refinement for code under matched compute
2. **Trajectory analysis** — nobody has published per-pass metrics showing how refinement improves (or degrades) code across iterations
3. **Practical scale** — proving the thesis at 0.5B with LoRA on a single GPU makes it reproducible by any university lab
4. **CDLM gap** — CDLM showed diffusion *can* self-correct, but didn't compare against AR self-refinement baselines (Self-Refine, Reflexion)
