# DiffuLLM: Discrete Diffusion Language Modeling

**Authors:** Varesh Patel, Urvi Desai, Aparajita Sarkar

**Final Report:** [📄 FINAL_REPORT.pdf](./FINAL_REPORT.pdf)

---

A discrete diffusion language model trained from scratch on structured recipe text. The model treats generation as a parallel denoising process over masked token sequences rather than autoregressive left-to-right decoding.

Three systems are compared in the final report:
- **DiffuLLM** — custom bidirectional transformer with D3PM masked reconstruction loss
- **SEDD-SE** — score entropy discrete diffusion (Lou et al., 2023)
- **SEDD-KL** — same architecture, ELBO-based KL divergence loss

All training was run on the Rutgers iLab GPU cluster.
