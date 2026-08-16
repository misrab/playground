# 03 — Yeast sequences (VAE)

**Question:** can we learn a latent space of yeast promoters (or a protein family) and sample new ones?

**Data:** *S. cerevisiae* promoters, or one well-studied yeast protein family (UniProt).

**Plan:**
1. Small VAE on one-hot (or tokenized) sequences
2. Plots: latent 2D, reconstruction quality, a few sampled sequences
3. Diffusion only if the VAE is done and still interesting

**Stack:** Colab, PyTorch. Optional JAX if already comfortable.

**Out of scope:** protein-structure diffusion, starting this before 01 and 02 ship.
