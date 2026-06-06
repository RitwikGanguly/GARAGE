# Summary

GARAGE (**G**raph-**A**ttentive **RA**re-cell aware single-cell data **GE**neration) is a deep learning framework for generating high-fidelity synthetic single-cell RNA-seq (scRNA-seq) data.

Traditional Generative Adversarial Networks (GANs) often struggle with the high-dimensional and sparse nature of scRNA-seq data, leading to training instability and a failure to reproduce rare but biologically important cell populations. GARAGE overcomes these challenges with a unique two-stage architecture that intelligently guides the generative process.

---

## Workflow

The GARAGE framework uses a two-stage process to generate realistic synthetic cells, with a special focus on preserving rare cell types.

<p align="center">
  <img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/garage_workflow.jpg" alt="GARAGE" width="680"/>
</p>

*A high-level overview of the GARAGE framework.*

1.  **Stage 1: GAT-based Cell Selection:**
    A Graph Attention Network (GAT) is trained on a cell-cell KNN graph. By leveraging its attention mechanism, the GAT identifies a core set of "archetypal" or high-importance cells that are most influential in defining the data's structure and cell-type identities.

2.  **Stage 2: GAT-Seeded GAN Generation:**
    Instead of receiving only random noise, the GAN's generator is fed a **hybrid input batch** — a mixture of random vectors and the high-priority "seed" cells selected by the GAT. This "attention-guided leakage" anchors the generator to known, biologically realistic states, stabilising training and ensuring all cell types are represented.

---

## Key Features

*   **GAT-Informed Seeding:** Moves beyond random sampling to intelligently select the most representative cells to guide generation.
*   **Enhanced Rare Cell Generation:** The framework is explicitly designed to better capture and generate samples for rare and underrepresented cell populations.
*   **Improved Stability & Convergence:** The seeded-generation process significantly stabilises GAN training, reduces mode collapse, and accelerates convergence.
*   **High-Fidelity Synthetic Data:** Produces synthetic datasets ideal for data augmentation, methods benchmarking, and privacy-preserving data sharing.

---

## Repository Structure

```
.
├── data_generation/                Core GARAGE pipeline
│   ├── garage.py                   Full GARAGE: GAT subsampling + GAN generation
│   └── wasserstein_distance.py     Wasserstein distance (real vs. generated)
│
├── data_validation/                Quality evaluation
│   ├── feature_selection.py        Python implementation (CV², Fano, PCA loading)
│   ├── feature_selection.R         Original R implementation (reference)
│   ├── data_validation.py          Clustering (Leiden) → ARI / NMI / macro-F1 / UMAP
│   └── data_vaidation_garage.ipynb Original validation notebook (reference)
│
├── benchmarking/
│   ├── sota/                       General-purpose baselines (PyTorch)
│   │   ├── gan.py, wgan.py, fgan.py, vae.py, lsh_gan.py
│   │   └── *_tf1.py               Original TF1.11 reference implementations
│   └── scrna_seq_specific/        scRNA-seq-specific baselines (PyTorch)
│       ├── scgan.py, scvae.py, scdiffusion.py
│       └── gan_ros.py, vae_ros.py
│
├── biological_analysis/            Rare-cell biology experiments
│   ├── biological_validation.py    GAT attention ↔ marker-gene enrichment
│   ├── rare_cell_utility.py        Held-out rare-cell classification utility
│   └── marker_gene_clustering.py   Marker-gene-based clustering evaluation
│
├── ablation_study/                 Sensitivity experiments
│   ├── leakage_ablation.py         GAT leakage fraction 0.0–0.3
│   └── multi_seed_synthesis.py     Multi-seed generation (5 seeds × 4 datasets)
│
├── analysis/                       Post-processing and plotting
│   ├── distribution_metrics.py     MMD + Sliced Wasserstein Distance
│   ├── clustering_evaluation.py    Feature-selection + clustering across seeds
│   ├── aggregate_losses.py         Aggregate GAN loss records
│   ├── build_summary_tables.py     Mean±std summary tables
│   └── plot_wasserstein_vs_leakage.py  WD vs leakage figure
│
├── data/                           Input data directory
│   ├── cell_types/                 Cell-type label files (*.csv)
│   └── expression_matrix/          Gene expression matrices (*.csv)
│
├── results/                        Output directory (generated CSVs, figures)
├── docs/                           ReadTheDocs documentation source
├── img/                            Images used in README and docs
├── config.py                       Shared paths and hyper-parameter constants
├── CITATION.cff                    Citation metadata
├── LICENSE                         MIT License
├── requirements_garage.txt         Core dependencies (Python 3.12.5)
└── requirements_benchmarking.txt   Benchmarking dependencies (Python 3.7.12)
```

**Note:** GARAGE pipeline requires Python 3.12.5; benchmarking reference baselines use Python 3.7.12.

---

## Citation

If you use **GARAGE** in your research, please cite:

> Ganguly, R., et al.  "GARAGE: A Graph-Attentive GAN for Rare Cell-Aware Single-Cell RNA-seq Data Generation."  *bioRxiv*, 2025.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
