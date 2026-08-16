# 01 — E. coli promoters (CNN)

**Question:** can a model tell promoter DNA from random / non-promoter sequence?

**Data:** RegulonDB or a small RefSeq *E. coli* promoter set. One-hot encode DNA.

**Plan:**
1. Tiny 1D CNN from scratch (PyTorch)
2. Same task with a small pretrained DNA model (HF) if useful
3. Plots: confusion matrix, sequence logo of learned motifs / saliency

**Stack:** Colab, PyTorch. JAX rewrite of the vanilla CNN only if this ships.

**Out of scope:** images, transformers, JAX-first.
