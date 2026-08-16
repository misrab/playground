# 02 — Bacterial / fungal proteins (Transformer)

**Question:** can sequence alone predict protein family or localization (non-human)?

**Data:** UniProt, filter bacteria or fungi. Keep it small.

**Plan:**
1. Tiny transformer from scratch (PyTorch, or JAX if 01 went well)
2. Fine-tune ESM2 on the same labels
3. Plots: UMAP of embeddings, attention over sequence

**Stack:** Colab, HuggingFace `transformers` + ESM2 (PyTorch).

**Out of scope:** structure prediction, human proteins, huge fine-tunes.
