# How-to: Feature Selection

A recipe for selecting the most informative genes from your generated data.

## Goal

Identify the top $k$ genes that capture the most biological variability in generated scRNA-seq data, for downstream clustering evaluation.

## Prerequisites

- Generated data CSV from GARAGE or a baseline model.
- The corresponding real expression matrix and cell-type labels (via `config.py`).

## Steps

### 1. CV² selection (recommended default)

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method cv2 \
    --n_genes 100
```

CV² = $\sigma^2 / \mu^2$ — selects genes with high relative variability.

### 2. Fano index selection

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method fano \
    --n_genes 100
```

Fano = $\sigma^2 / \mu$ — similar to CV² but with different mean scaling.

### 3. PCA loading selection

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method pca \
    --n_genes 100
```

Ranks genes by their aggregate loading on the first 3 principal components.

### 4. Using the Python API directly

```python
from data_validation.data_validation import cv2_selection, fano_selection, pca_loading_selection
import pandas as pd

gen_data = pd.read_csv("data/gen_data/muraro_data_mixdata_iter3_top_426.csv").values
top_genes = cv2_selection(gen_data, n_genes=100)
```

## Choosing the right method

| Method | Best for | Weakness |
|--------|----------|----------|
| **CV²** | General purpose; good for log-normalised data. | Sensitive to low-mean genes (division by small μ). |
| **Fano** | Count-based or raw-normalised data. | Favours high-mean genes. |
| **PCA loading** | Captures global variance structure. | Sensitive to batch effects and outliers. |

**Rule of thumb:** Start with CV² for 100 genes. If ARI is unexpectedly low, try PCA loading as a fallback.

## Output

The `--gen_csv` command prints selected gene indices and the top metrics (ARI, NMI, F1). It also saves results to `results/<dataset>_<method>_ari_nmi_f1.csv`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CV² returns all zeros | Zero-variance genes in the generated data. Check that the GAN produced diverse output. |
| PCA loading selects the same 3 genes | The first 3 PCs explain nearly all variance (common for simple datasets). Try Fano. |
| All methods give similar ARI ~ 0.5 | This may be the best the model can do — see [Interpret Outputs](howto/interpret_outputs). |

## Related

- [Clustering Validation](howto/clustering_validation) — the next step after feature selection.
- [Evaluation Metrics](background/evaluation_metrics) — detailed explanation of each method.
