# How-to: Run GARAGE

A recipe for running the full GARAGE pipeline on any dataset.

## Goal

Generate synthetic scRNA-seq data using the GARAGE two-stage pipeline (GAT subsampling → GAN generation).

## Prerequisites

- [Installation](installation) completed.
- At least one dataset configured in `config.py` → `DATASET_CONFIG`.

## Steps

### 1. Basic invocation

```bash
python -m data_generation.garage --dataset muraro
```

Replace `muraro` with `yan`, `pollen`, or `cbmc` for the built-in datasets, or your own registered dataset name.

### 2. Customise hyper-parameters

Edit `config.py` → `GARAGE_DEFAULTS` before running:

```python
GARAGE_DEFAULTS = {
    "gan_total_iters": 30001,  # Train longer
    "leakage_fraction": 0.3,   # More GAT seeds
    "g_lr": 0.0001,            # Slower generator
    "d_lr": 0.0002,            # Slower discriminator
    # ... other params unchanged
}
```

### 3. Monitor training

Console output shows loss every 1000 iterations:

```
iter 0:    D_loss=1.231, G_loss=0.823
iter 1000: D_loss=0.752, G_loss=1.412
iter 5000: D_loss=0.611, G_loss=1.658
...
```

- **G_loss increasing slightly** is expected early on as the generator learns.
- **D_loss oscillating** (> 1.0 swings) indicates training instability — reduce learning rates.
- **G_loss diverging** (→ ∞) — reduce `d_lr` or increase `nd_steps`.

### 4. Running on multiple datasets

```bash
for d in yan pollen cbmc muraro; do
    python -m data_generation.garage --dataset $d
done
```

## Output

Generated data is saved to `data/gen_data/<dataset>_data_mixdata_iter3_top_426.csv`.

Training losses are saved to `results/losses_<dataset>.csv`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM | Reduce batch size/leakage fraction, or set `DEVICE = "cpu"` in `garage.py`. |
| Fast GAT convergence (loss = 0 after 100 iters) | GAT may be overfitting — reduce `gat_epochs`. |
| Slow GAN convergence | Increase `gan_total_iters` or adjust learning rates. |
| "File exists" error on output | Delete the existing `data/gen_data/` file or rename it. |

## Related

- [GARAGE Architecture](background/garage_architecture) — how the two-stage pipeline works.
- [Quickstart](quickstart) — first-time setup.
- [Feature Selection](howto/feature_selection) — next step after generation.
