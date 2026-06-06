# FAQ

Frequently asked questions about GARAGE.

---

## General

### What is GARAGE?

**GARAGE** (Graph-Attentive Rare-cell-Aware single-cell data Generation) is a deep learning framework that generates synthetic scRNA-seq data with explicit preservation of rare cell populations. It combines a Graph Attention Network (GAT) for intelligent cell selection with a Generative Adversarial Network (GAN) for synthesis.

### What Python version do I need?

- **Core pipeline:** Python 3.12.5
- **Benchmarking reference baselines (TF1.11):** Python 3.7.12
- **Data validation notebook:** Python 3.9.21 (compatible with 3.12.5 via `data_validation.py`)

### Do I need a GPU?

A GPU is recommended for CBMC (7,895 cells) and Muraro (2,126 cells). The code auto-detects CUDA. Yan and Pollen train in minutes on CPU.

### Can I use GARAGE on my own dataset?

Yes. See [Preparing Your Data](prepare_your_data) for exact CSV format requirements and how to register your dataset in `config.py`.

---

## Generation

### How long does training take?

| Dataset | Cells | GPU time (approx.) | CPU time (approx.) |
|---------|-------|-------------------|-------------------|
| Yan | 124 | 1 min | 2 min |
| Pollen | 301 | 2 min | 5 min |
| Muraro | 2,126 | 10 min | 30 min |
| CBMC | 7,895 | 20 min | 1 hour |

Times are for default settings (GAT: 7501 epochs, GAN: 20001 iterations).

### My GAN losses are oscillating. Is that normal?

Some oscillation is normal. If loss swings are large (> 2.0) and persistent, try:
- Reducing the learning rates (`g_lr`, `d_lr` in `config.py`).
- Increasing `nd_steps` (discriminator updates per generator step).

At convergence, D_loss should be approximately $\log(2) \approx 0.69$.

### What does the leakage fraction ($\lambda$) do?

$\lambda$ controls how many real seed cells (selected by the GAT) are mixed into the generator's input. Higher $\lambda$ → more guidance from real data → more stable training but potentially less novelty. Default: 0.2.

---

## Validation

### My ARI is very low. What should I do?

Ordered by likelihood:
1. Try a different feature selection method (`--method pca` instead of `--method cv2`).
2. Increase the number of selected genes (`--n_genes 200` or 500).
3. Check that the Leiden resolution sweep covers an appropriate range for your dataset.
4. Re-train GARAGE with a higher `leakage_fraction`.
5. Check that the generated data is not pure noise (via UMAP plot).

### What's a "good" ARI for scRNA-seq data?

This depends on the dataset. As a rough guide:
- **Yan** (124 cells, 6 types): ARI > 0.60 is good.
- **Pollen** (301 cells, 11 types): ARI > 0.45 is good.
- **CBMC** (7,895 cells, 13 types): ARI > 0.55 is good.
- **Muraro** (2,126 cells, 10 types): ARI > 0.50 is good.

### Why is my macro-F1 much lower than ARI?

This usually means at least one rare cell type is being lost by the generator. Increase `priority_weight` in `config.py` and ensure `rare_threshold` is set correctly.

### Can I compare different models?

Yes. Run the full benchmark suite (see [Benchmarking Tutorial](tutorials/04_benchmark_against_sota)) and use `analysis/build_summary_tables.py` to produce comparison tables.

---

## Troubleshooting

### CUDA out of memory

- Reduce `leakage_fraction` (fewer real cells per batch).
- Run on CPU: set `DEVICE = "cpu"` at the top of `garage.py`.
- For CBMC, consider subsetting to fewer cells.

### "No module named 'torch_geometric'"

```bash
pip install torch_geometric
```

Or see the exact PyG installation command for your PyTorch/CUDA version at [pyg.org](https://pyg.org/whl/).

### Generated CSV is empty or all zeros

The generator may have collapsed to producing zeros. Check:
- Training logs: is G_loss diverging?
- Try decreasing learning rates.
- Try training for fewer iterations.

### Feature selection returns 0 genes

This can happen if the generated data is all zeros or has zero variance. Run the UMAP plot first to check if generated data is plausible.

---

## Citing GARAGE

```bibtex
@software{garage2025,
  author    = {Ganguly, Ritwik and others},
  title     = {GARAGE: Graph-Attentive Rare-cell-Aware single-cell RNA-seq Data Generation},
  year      = {2025},
  publisher = {bioRxiv},
  doi       = {10.1101/2025.09.28.679012},
  url       = {https://github.com/RitwikGanguly/GARAGE}
}
```

See also the [Citation](citation) page.

---

## Getting Help

- **Documentation:** [garage-docs.readthedocs.io](https://garage-docs.readthedocs.io/)
- **GitHub Issues:** [github.com/RitwikGanguly/GARAGE/issues](https://github.com/RitwikGanguly/GARAGE/issues)
- **BioRxiv:** [10.1101/2025.09.28.679012](https://doi.org/10.1101/2025.09.28.679012)
