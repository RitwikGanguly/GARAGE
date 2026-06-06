# How-to: Run Ablation Studies

A recipe for the two sensitivity analyses: leakage fraction sweep and multi-seed reproducibility.

## Goal

Understand how GARAGE's performance changes with different leakage fractions and whether results are reproducible across random seeds.

## Prerequisites

- GARAGE environment installed.
- Ablation scripts in `ablation_study/`.

## Steps

### 1. Leakage fraction ablation

```bash
python ablation_study/leakage_ablation.py
```

This runs GARAGE on all 4 datasets with $\lambda \in \{0.0, 0.1, 0.2, 0.3\}$ and logs losses to `results/rev6_losses.csv`.

### 2. Multi-seed synthesis

```bash
python ablation_study/multi_seed_synthesis.py
```

This runs GARAGE with 5 different random seeds across all 4 datasets and saves the generated data with seed-specific filenames.

### 3. Generate the WD vs. leakage figure

```bash
python analysis/plot_wasserstein_vs_leakage.py
```

Saves `results/wasserstein_vs_leakage.pdf` — a plot showing how WD changes with $\lambda$.

## Interpreting the Results

- **Leakage = 0.0:** Standard GAN (no seeding). Expect higher WD and lower ARI.
- **Leakage = 0.2:** GARAGE default. Should show best or near-best metrics.
- **Leakage ≥ 0.3:** Diminishing returns — too much real data in the batch and the generator simply copies.

The WD vs. leakage plot should show a U-shape or steady decline, confirming that $\lambda = 0.2$ is a reasonable default.

## Related

- [Tutorial: End-to-End Guide](tutorials/01_end_to_end)
- [GARAGE Architecture](background/garage_architecture)
