# 01 — Yeast cells (2D CNN)

**Question:** where in the yeast cell is this fluorescent protein? (localization / morphology)

**Data:** DeepLoc yeast GFP images (Kraus et al.) — *S. cerevisiae*, ~12 localization classes. Non-human, real microscopy, sized for Colab.

Fallback if that download is painful: DIBaS (33 bacterial species, brightfield). Same plan, prokaryote instead of fungi.

**Plan:**
1. Small 2D CNN from scratch (PyTorch)
2. Same task, pretrained ResNet (timm or HF)
3. Plots: confusion matrix, Grad-CAM on cells, a few failure cases

**Stack:** Colab, PyTorch. JAX rewrite of the vanilla CNN only if this ships.

**Out of scope:** DNA/1D, PlantVillage, human Cell Painting, transformers.
