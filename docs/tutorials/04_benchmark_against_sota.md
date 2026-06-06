# Tutorial: Benchmark Against SOTA

This tutorial shows how to run GARAGE alongside state-of-the-art generative models and compare their synthetic data quality using the multi-metric evaluation framework.

---

## Why Benchmark?

Benchmarking against SOTA models establishes GARAGE's relative performance. The multi-metric approach (WD/MMD/SWD + ARI/NMI/F1) provides a quantitative basis for claims about distributional fidelity and biological utility.

---

## Step 1: Generate Data with All Models

Run each model on all 4 datasets (Yan, Pollen, CBMC, Muraro):

### GARAGE

```bash
for d in yan pollen cbmc muraro; do
    python -m data_generation.garage --dataset $d
done
```

### General-purpose SOTA baselines

```bash
for d in yan pollen cbmc muraro; do
    python -m benchmarking.sota.gan --dataset $d
    python -m benchmarking.sota.wgan --dataset $d
    python -m benchmarking.sota.fgan --dataset $d
    python -m benchmarking.sota.vae --dataset $d
    python -m benchmarking.sota.lsh_gan --dataset $d
done
```

### scRNA-seq-specific baselines

```bash
for d in yan pollen cbmc muraro; do
    python -m benchmarking.scrna_seq_specific.scgan --dataset $d
    python -m benchmarking.scrna_seq_specific.scvae --dataset $d
    python -m benchmarking.scrna_seq_specific.scdiffusion --dataset $d
    python -m benchmarking.scrna_seq_specific.gan_ros --dataset $d
    python -m benchmarking.scrna_seq_specific.vae_ros --dataset $d
done
```

**Note:** This generates $11 \times 4 = 44$ synthetic datasets. Expect 2–4 hours on GPU depending on hardware.

---

## Step 2: Compute Distributional Metrics

```bash
# Wasserstein Distance (all datasets)
for d in yan pollen cbmc muraro; do
    python -m data_generation.wasserstein_distance \
        --dataset $d \
        --gen_csv data/gen_data/${d}_data_mixdata_iter3_top_426.csv
done

# MMD and Sliced Wasserstein Distance (all methods × all datasets)
python analysis/distribution_metrics.py

# Individual SWD analysis
python analysis/swd_analysis.py
python analysis/mmd_analysis.py
```

---

## Step 3: Compute Clustering Metrics

```bash
# Feature selection + Leiden clustering for all generated files
python analysis/clustering_evaluation.py

# scRNA-seq-specific benchmark metrics
python analysis/sc_specific_benchmark.py
```

---

## Step 4: Build Summary Tables

```bash
# Aggregate all losses into a single CSV
python analysis/aggregate_losses.py

# Build mean ± std summary tables (WD/MMD/SWD per method)
python analysis/build_summary_tables.py

# Marker gene clustering grid search
python analysis/marker_clustering_grid.py
```

---

## Step 5: Generate Publication-Quality Figures

```bash
# Wasserstein vs. leakage fraction plot
python analysis/plot_wasserstein_vs_leakage.py
```

Check `results/` for the generated figures.

---

## Step 6: Interpret the Results

### Output files

| File | Description |
|------|-------------|
| `results/summary_wasserstein.csv` | Mean WD per method per dataset |
| `results/summary_ari_nmi_f1.csv` | Mean ARI/NMI/F1 per method per dataset |
| `results/summary_mmd_swd.csv` | Mean MMD/SWD per method per dataset |
| `results/all_losses.csv` | Raw losses from all training runs |
| `results/wasserstein_vs_leakage.pdf` | Leakage fraction sweep figure |

### How to compare

A typical comparison table looks like:

| Method | WD (↓) | MMD (↓) | ARI (↑) | NMI (↑) | F1 (↑) |
|--------|--------|---------|---------|---------|--------|
| GAN | 0.045 | 0.312 | 0.412 | 0.523 | 0.398 |
| WGAN | 0.038 | 0.287 | 0.438 | 0.551 | 0.421 |
| F-GAN | 0.041 | 0.301 | 0.425 | 0.540 | 0.410 |
| VAE | 0.092 | 0.456 | 0.318 | 0.441 | 0.305 |
| LSH-GAN | 0.012 | 0.178 | 0.521 | 0.634 | 0.498 |
| scGAN | 0.033 | 0.265 | 0.475 | 0.582 | 0.452 |
| scVAE | 0.058 | 0.389 | 0.395 | 0.508 | 0.381 |
| scDiffusion | 0.041 | 0.310 | 0.442 | 0.549 | 0.433 |
| GAN-ROS | 0.028 | 0.252 | 0.488 | 0.591 | 0.461 |
| VAE-ROS | 0.051 | 0.365 | 0.408 | 0.522 | 0.395 |
| **GARAGE** | **0.007** | **0.142** | **0.584** | **0.672** | **0.545** |

*(Numbers above are illustrative.)*

### Key comparison dimensions

1. **Distributional fidelity (WD, MMD, SWD):** Does the synthetic data match the real data in gene-expression space? Lower is better.

2. **Biological utility (ARI, NMI, F1):** Does the synthetic data cluster into biologically meaningful groups? Higher is better.

3. **Rare-cell preservation:** The macro-F1 is particularly important — it equally weights each cell type, so rare types are not averaged out. GARAGE should show the highest macro-F1 relative to its ARI.

---

## Reproducing the Paper's Figures

If you are writing a paper that compares GARAGE to other methods, you can build summary tables with mean ± std:

```bash
python analysis/build_summary_tables.py
```

This produces CSV files with columns like:

```
method,dataset,wd_mean,wd_std,ari_mean,ari_std,nmi_mean,nmi_std,f1_mean,f1_std
GARAGE,yam,0.0067,0.0012,0.724,0.031,0.801,0.025,0.694,0.028
```

These are directly importable into R (`read.csv`) or Python (`pandas.read_csv`) for plotting with `ggplot2` or `matplotlib`/`seaborn`.

---

## Adding a New Baseline

To add a new model to the benchmark:

1. Write a new Python file in `benchmarking/sota/` or `benchmarking/scrna_seq_specific/`.
2. Follow the existing interface: `--dataset` argument, outputs to `data/gen_data/`, and uses `config.py` for paths.
3. Run it with the same `for d in ...` loop above.
4. Add the model to the analysis scripts (`analysis/distribution_metrics.py`, `analysis/clustering_evaluation.py`).
5. Re-run `python analysis/build_summary_tables.py`.

---

## Related Pages

- [Tutorial: End-to-End Guide](tutorials/01_end_to_end)
- [How‑to: Benchmarking](howto/benchmarking)
- [Evaluation Metrics](background/evaluation_metrics)
