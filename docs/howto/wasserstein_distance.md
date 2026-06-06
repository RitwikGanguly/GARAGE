# How-to: Compute Wasserstein Distance

A recipe for measuring distributional similarity between real and generated scRNA-seq data.

## Goal

Compute the Wasserstein (Earth Mover's) Distance between the real expression matrix and a generated data CSV.

## Prerequisites

- Generated data CSV (GARAGE or any baseline).
- The corresponding real expression matrix registered in `config.py`.

## Steps

### 1. Single dataset

```bash
python -m data_generation.wasserstein_distance \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv
```

### 2. Batch across multiple datasets

```bash
for d in yan pollen cbmc muraro; do
    python -m data_generation.wasserstein_distance \
        --dataset $d \
        --gen_csv data/gen_data/${d}_data_mixdata_iter3_top_426.csv
done
```

### 3. Using the Python API

```python
from data_generation.wasserstein_distance import compute_wasserstein
import pandas as pd

real = pd.read_csv("data/expression_matrix/muraro_expression_matrix.csv")
gen = pd.read_csv("data/gen_data/muraro_data_mixdata_iter3_top_426.csv")

wd = compute_wasserstein(real.values, gen.values)
print(f"Wasserstein distance: {wd:.6f}")
```

## Interpreting the Output

| WD Range | Interpretation |
|----------|---------------|
| < 0.005 | Near-perfect match (e.g., CBMC with GARAGE) |
| 0.005–0.02 | Excellent match |
| 0.02–0.10 | Good match |
| 0.10–0.50 | Moderate — distribution has significant divergence |
| > 0.50 | Poor — generated data is far from real |

**Note:** WD depends on dataset scale and normalisation. Always normalise to sum-to-1 distributions (handled by `wasserstein_distance.py` automatically).

## Related

- [Evaluation Metrics](background/evaluation_metrics) — WD, MMD, SWD and how they compare.
- [Distribution Metrics Analysis](analysis/distribution_metrics.py) — batch processing for all methods.
