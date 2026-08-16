# 01 — Yeast cells (2D CNN)

**Question:** where in the yeast cell is this fluorescent protein? (localization / morphology)

**Data:** DeepLoc — *S. cerevisiae* GFP localization, 19 classes, 2-channel hdf5 (`Chong_*.hdf5`).
Download: `http://spidey.ccbr.utoronto.ca/~okraus/DeepLoc_full_datasets.zip`

Docs (no dataset card; paper + code are the schema):
- Kraus et al. 2017, *MSB* — https://www.embopress.org/doi/full/10.15252/msb.20177551
- Repo — https://github.com/okraus/DeepLoc
- Source screen: Chong et al. 2015, *Cell* — https://doi.org/10.1016/j.cell.2015.04.051

Fallback: DIBaS (33 bacterial species, brightfield).

**Plan:**
1. Small 2D CNN from scratch (PyTorch)
2. Same task, pretrained ResNet (timm or HF)
3. Plots: confusion matrix, Grad-CAM on cells, a few failure cases

**Stack:** Colab, PyTorch. JAX rewrite of the vanilla CNN only if this ships.

**Out of scope:** DNA/1D, PlantVillage, human Cell Painting, transformers.
