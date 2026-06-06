# Analysis

Post-processing scripts for aggregating results and generating publication figures.

## Files

| File | Description |
|---|---|
| `distribution_metrics.py` | MMD (Maximum Mean Discrepancy) and Sliced Wasserstein Distance. |
| `clustering_evaluation.py` | Feature selection + Leiden clustering across multiple seeds → raw results CSV. |
| `sc_specific_benchmark.py` | ARI/NMI/F1 benchmarking for scRNA-seq-specific baselines (scGAN, scVAE, scDiffusion, GAN+ROS, VAE+ROS). |
| `aggregate_losses.py` | Aggregates per-iteration GAN losses into compact summary statistics. |
| `build_summary_tables.py` | Creates publication-ready mean±std summary tables from raw clustering results. |
| `marker_clustering_grid.py` | Grid-search version of the marker-gene clustering evaluation. |
| `plot_wasserstein_vs_leakage.py` | Generates Wasserstein distance vs leakage plots. |
| `mmd_analysis.py` | Standalone MMD analysis. |
| `swd_analysis.py` | Standalone Sliced Wasserstein analysis. |
