# Wasserstein Distance

## Overview

Wasserstein distance (Earth Mover's Distance) measures the minimum "cost" to transform one probability distribution into another. In data generation models like GARAGE, it quantifies the level of similarity between the real scRNA-seq data distribution and the generated synthetic data distribution.

**Intuition:** Given two distributions (real data $P$ and generated data $Q$), the Wasserstein distance is the minimum amount of "mass" that must be moved — weighted by distance — to transform $P$ into $Q$.

### Advantages

- Works well for distributions that may not perfectly overlap.
- Captures spatial/geometric relationships in the data.
- Provides a single scalar that can be compared across datasets and models.

### Limitations

- Computationally expensive for large datasets (exact solution via Optimal Transport is $O(n^3)$).
- Approximate solutions (Sinkhorn) reduce cost but introduce a regularisation bias.
- Should be complemented with clustering-based metrics (ARI, NMI, F1).

---

## Wasserstein Distance in GARAGE

After generating synthetic cell samples with GARAGE, the Wasserstein distance is computed between the generated and real data distributions. A small Wasserstein distance (e.g., $\ll 0.1$ for normalised data) indicates that the synthetic data closely matches the real data.

### Running the Computation

```bash
# Example: compute WD between real CBMC and generated output
python -m data_generation.wasserstein_distance \
    --dataset cbmc \
    --gen_csv data/gen_data/cbmc_data_mixdata_iter3_top_426.csv
```

The script:
1. Loads the real expression matrix (via `config.py`).
2. Loads the generated data CSV.
3. Normalises both matrices to sum-to-1 probability distributions.
4. Computes the pairwise Euclidean distance matrix.
5. Runs Optimal Transport (`ot.emd2`) to compute the exact Earth Mover's Distance.

### Implementation Reference

See `data_generation/wasserstein_distance.py` for the full implementation. Key snippet:

```python
import ot
from scipy.spatial.distance import cdist

unifs1 = real_data / len(real_data)
unifs2 = gen_data / len(gen_data)
dist_mat = cdist(unifs1, unifs2, metric='euclidean')
emd_dist = ot.emd2(
    np.ones(len(real_data)) / len(real_data),
    np.ones(len(gen_data)) / len(gen_data),
    dist_mat,
    numItermax=100000
)
```

### What Good Scores Look Like

| Dataset | Cells | Typical WD | Interpretation |
|---------|-------|------------|----------------|
| Yan     | 124   | < 0.01     | Excellent match on small datasets |
| Pollen  | 301   | < 0.02     | Tight fit |
| CBMC    | 7,895 | < 0.005    | Near-perfect on large datasets |
| Muraro  | 2,126 | < 0.01     | Strong match |

### Improving Wasserstein Distance

If your WD is high:
1. Train the GAN for more iterations (increase `gan_total_iters` in `config.py`).
2. Increase the leakage fraction $\lambda$ (e.g., from 0.2 to 0.3).
3. Adjust the generator/discriminator learning rates.
4. Check that the generated output includes all cell types (no mode collapse).

---

## Related Metrics

Wasserstein distance is one of several distributional metrics available in GARAGE:

- **Maximum Mean Discrepancy (MMD):** Kernel-based; see `analysis/distribution_metrics.py`.
- **Sliced Wasserstein Distance (SWD):** 1D projections for speed; see `analysis/swd_analysis.py`.

For clustering-based validation (ARI, NMI, macro-F1, UMAP), see [Clustering Validation](howto/clustering_validation).
