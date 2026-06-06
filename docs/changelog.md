# Changelog

All notable changes to the GARAGE project.

---

## [1.0.0] — 2025-09-28 (bioRxiv preprint)

### Added

- Two-stage GARAGE pipeline: GAT-based cell selection + GAT-seeded GAN generation.
- Built-in support for 4 scRNA-seq datasets: Yan, Pollen, CBMC, Muraro.
- Wasserstein distance computation via Optimal Transport.
- Data validation module: CV², Fano, and PCA loading feature selection; Leiden clustering; ARI/NMI/macro-F1 reporting; UMAP visualisation.
- 5 general-purpose SOTA baselines (PyTorch): GAN, WGAN, F-GAN, VAE, LSH-GAN.
- 5 scRNA-seq-specific baselines (PyTorch): scGAN, scVAE, scDiffusion, GAN-ROS, VAE-ROS.
- Biological validation: GAT attention ↔ marker-gene enrichment, rare-cell positive rate analysis.
- Held-out rare-cell utility experiment with Random Forest classifier.
- Ablation studies: leakage fraction sweep (0.0–0.3) and multi-seed synthesis (5 seeds).
- Analysis suite: distribution metrics, clustering evaluation, aggregate losses, summary tables.
- ReadTheDocs documentation (Sphinx + MyST Parser + sphinx-rtd-theme).
- `config.py` — single source of truth for paths and hyper-parameters.
- `CITATION.cff` — citation metadata for GitHub/Zotero integration.
