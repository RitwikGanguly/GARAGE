# How-to: Clustering Validation

A recipe for measuring the biological utility of generated scRNA-seq data via clustering quality.

## Goal

Report ARI, NMI, and macro-F1 scores by comparing Leiden clusters of the generated data to ground-truth cell type labels.

## Prerequisites

- Generated data CSV.
- Feature selection completed (see ./feature_selection).

## Steps

### 1. Run the complete validation

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method cv2 \
    --n_genes 100 \
    --plot_umap
```

This performs feature selection, PCA, KNN graph construction, Leiden clustering (resolution sweep), and ARI/NMI/F1 reporting — all in one command.

### 2. Examine the UMAP plots

The `--plot_umap` flag saves UMAP visualisations to `results/`:

- `results/<dataset>_<method>_umap_real.pdf` — real data coloured by Leiden cluster.
- `results/<dataset>_<method>_umap_gen.pdf` — generated data coloured by Leiden cluster.

Compare these qualitatively: clusters in the generated plot should mirror clusters in the real plot in shape, separation, and relative position.

### 3. Tune the resolution sweep

The default resolution ranges are in `data_validation/data_validation.py`:

```python
RESOLUTION_RANGES = {
    "yan": np.arange(0.80, 1.61, 0.01),
    "pollen": np.arange(0.10, 3.01, 0.01),
    "cbmc": np.arange(0.20, 0.81, 0.01),
    "muraro": np.arange(0.10, 3.01, 0.01),
}
```

For your own dataset, experiment with this range. The best resolution is the one that maximises ARI against ground truth.

### 4. Interpret the results

```bash
cat results/muraro_cv2_ari_nmi_f1.csv
```

Typical columns:

```
dataset,method,n_genes,best_resolution,ari,nmi,f1
muraro,cv2,100,1.42,0.6734,0.7519,0.6108
```

**Interpretation thresholds (scRNA-seq data):**

| ARI Range | Interpretation |
|-----------|----------------|
| > 0.70 | Excellent — the generated data's clusters match ground truth nearly as well as real data. |
| 0.50–0.70 | Good — meaningful biological signal captured. |
| 0.30–0.50 | Fair — some signal, but cluster boundaries are fuzzy. |
| < 0.30 | Poor — generated data lacks the structure to resolve cell types. |

## Using the Python API

```python
from data_validation.data_validation import validate_generated_data
import pandas as pd

gen_df = pd.read_csv("data/gen_data/muraro_data_mixdata_iter3_top_426.csv")
metrics = validate_generated_data(
    dataset="muraro",
    gen_df=gen_df,
    method="cv2",
    n_genes=100,
    plot_umap=True
)
print(metrics)  # {"ari": 0.6734, "nmi": 0.7519, "f1": 0.6108, "resolution": 1.42}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| ARI = 0.0 | Feature selection failed to pick informative genes. | Try `--method pca` or increase `--n_genes`. |
| Leiden returns 1 cluster | Resolution too low. | Widen the sweep range (e.g., start at 0.01). |
| UMAP is a blob | Generated data lacks structure. | Re-train GARAGE with higher `leakage_fraction`. |
| NMI > ARI | NMI is less sensitive to cluster size imbalance — normal for datasets with rare types. | Expected. |
| macro-F1 much lower than ARI | Rare cell types are poorly represented in the generated data. | Increase `priority_weight` in `config.py`. |

## Related

- ./feature_selection — the prerequisite step.
- [Evaluation Metrics](background/evaluation_metrics) — detailed metric definitions.
- [Interpret Outputs](howto/interpret_outputs) — putting metrics together.
