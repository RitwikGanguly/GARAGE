# Ablation Study

Sensitivity experiments that validate key design decisions in GARAGE.

## Files

| File | Description |
|---|---|
| `leakage_ablation.py` | Varies the GAT leakage fraction (0.0–0.3) and records GAN training losses to assess how GAT seeding stabilizes training. |
| `multi_seed_synthesis.py` | Generates synthetic data using 5 random seeds across 5 methods × 4 datasets to assess robustness and reproducibility. |
