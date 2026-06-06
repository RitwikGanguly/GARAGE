# Motivation

## Why Generate Synthetic scRNA‑seq Data?

Single-cell RNA sequencing (scRNA-seq) has revolutionised biology by measuring gene expression in thousands of individual cells simultaneously. However, real scRNA-seq data comes with inherent limitations:

| Challenge | Impact |
|-----------|--------|
| **Rare cell types** | Populations like circulating tumour cells or tissue-resident stem cells appear in tiny numbers. Downstream analyses are underpowered. |
| **Data paucity** | Experiments are expensive and time-consuming. Small datasets limit the statistical power of clustering, differential expression, and machine learning models. |
| **Patient privacy** | Sharing real patient scRNA-seq data raises ethical and regulatory concerns (GDPR, HIPAA). |
| **Benchmarking** | Developing and testing new algorithms requires diverse, controlled datasets with known ground truth. |
| **Dropout & noise** | Zero-inflated expression, amplification noise, and batch effects obscure true biological signal. |

Synthetic data generation addresses all five challenges simultaneously — **if** the synthetic data faithfully reproduces the biological properties of real data.

---

## The GARAGE Approach

GARAGE was designed around a specific insight: **standard GANs fail on scRNA-seq data because they lose rare cell populations.** Mode collapse — where a GAN produces only a few "average" cell types repeatedly — is fatal for biological applications.

GARAGE solves this with a two-stage architecture:

1. **Stage 1:** A Graph Attention Network (GAT) identifies the most representative cells, with **explicit priority weighting for rare cell types.**
2. **Stage 2:** A GAN generates synthetic cells from a **hybrid input** that mixes random noise with GAT-selected seed cells.

The result: synthetic data that preserves **both abundant and rare cell types**, with distributional fidelity validated by Wasserstein distance, MMD, and clustering metrics.

---

## Where GARAGE Fits in the Literature

| Method | Approach | Rare-cell aware? | scRNA-seq optimised? |
|--------|----------|------------------|---------------------|
| `scGAN` (Marouf et al.) | WGAN-GP with deep generator | No | Yes |
| `scVAE` (Lopez et al.) | $\beta$-VAE with NB likelihood | No | Yes |
| `scDiffusion` (Luo et al.) | DDPM on log-normalised counts | No | Yes |
| `LSH-GAN` | Random KNN subsample + GAN | Partially | No |
| **GARAGE** | GAT-seeded GAN with priority weighting | **Yes** | **Yes** |

GARAGE is the first framework to **explicitly couple a graph attention model with a GAN** for the purpose of rare-cell preservation.

---

## Who Should Use GARAGE?

- **Bioinformaticians** wanting to augment small or imbalanced scRNA-seq datasets.
- **Methods developers** needing realistic benchmark datasets with ground-truth cell type labels.
- **Clinicians** exploring privacy-preserving data sharing for multi-centre studies.
- **Researchers** studying rare cell populations (cancer stem cells, circulating tumour cells, tissue-resident immune cells).

---

## Next Steps

- [GARAGE Architecture](garage_architecture) — detailed technical description of the two-stage pipeline.
- [Quickstart](quickstart) — your first synthetic dataset in 5 minutes.
- [End-to-End Tutorial](tutorials/01_end_to_end) — complete walkthrough.
