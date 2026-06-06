# Data Validation

Quality evaluation of generated scRNA-seq data via feature selection, Leiden clustering, and distributional metrics.

## Files

| File | Description |
|---|---|
| `feature_selection.py` | Python port of the R feature selection functions: CV² (normalised dispersion), Fano factor (variance/mean), and PCA loading. |
| `feature_selection.R` | Original R implementation (for reference / reviewer verification). |
| `data_validation.py` | Feature selection → cluster real data with Leiden → report ARI / NMI / macro-F1 / UMAP. |
| `data_vaidation_garage.ipynb` | Original Jupyter notebook (for reference). |

## Quick Start

```bash
python -m data_validation.feature_selection --method cv2 \
    --gen_csv <path_to_gen_csv> --real_csv <path_to_real_csv>

python -m data_validation.data_validation --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv
```
