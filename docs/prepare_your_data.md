# Preparing Your Data

GARAGE can generate synthetic cells from **any scRNA-seq dataset** — not just the four built-in ones. This guide explains how to format and register your own data.

---

## Required Input Files

You need exactly **two** CSV/TSV files:

| File | Format | Example |
|------|--------|---------|
| **Expression matrix** | Rows = cells, columns = genes. Values = log-normalised or scaled counts. | `my_data_counts.csv` |
| **Cell-type labels** | One column with the cell type for each cell. Same row order as the expression matrix. | `my_data_labels.csv` |

---

## Expression Matrix Format

The expression matrix should be a **cells × genes** CSV file. Values are typically:

- **Log-normalised counts** (e.g., `log2(CPM + 1)` from Seurat or Scanpy).
- **Scaled counts** (e.g., from `sc.pp.scale`).
- **Raw normalised counts** (the GAN will re-scale via Sigmoid).

### Option A: No header (cells × genes, no gene names)

```text
0.0,1.2,0.0,3.4,0.0,...
0.0,0.0,5.6,0.0,2.1,...
...
```

This is the format for **Yan** and **Pollen** datasets.

### Option B: With header (rows = cells, columns = gene names)

```text
cell_id,ENSG000001,RPL13A,GAPDH,...
cell_1,0.0,1.2,0.0,...
cell_2,0.0,0.0,5.6,...
...
```

This is the format for **CBMC** and **Muraro** datasets.

The critical setting in `config.py` is:

| Parameter | `yan`/`pollen` | `cbmc`/`muraro` |
|-----------|---------------|-----------------|
| `header` | `None` | `0` |
| `transpose` | `True` (files are genes × cells) | `False` (files are cells × genes) |
| `index_col` | — | `0` (first column is cell ID) |

---

## Cell-Type Labels Format

A single column of cell-type identifiers, one per cell:

```text
cell_type
B cell
T cell CD4+
NK cell
...
```

**Important:** The row order must match the expression matrix. If your expression matrix has 300 rows (cells), your labels file must have 300 entries.

---

## Registering Your Dataset

Add a new entry to `DATASET_CONFIG` in `config.py`:

```python
DATASET_CONFIG = {
    # ... existing entries ...

    "my_dataset": {
        "expression_file": "my_data_counts.csv",
        "label_file": "my_data_labels.csv",
        "header": 0,              # or None if no header
        "index_col": 0,           # or None
        "transpose": False,       # True if the file is genes × cells
        "label_header": 0,        # or None if labels file has no header
        "label_col": "cell_type", # or 0 if using column index
        "rare_threshold": 50,     # types with < 50 cells are "rare"
        "iter_map": [0, 1, 2, 3, 4, 5],
    },
}
```

---

## Setting the Rare Threshold

The `rare_threshold` parameter is critical — it defines which cell types GARAGE treats as "rare" and gives priority attention. A good rule of thumb:

| Total cells ($n$) | Recommended threshold |
|-------------------|----------------------|
| < 500 | $n / 10$ |
| 500 – 5,000 | 50 – 200 |
| > 5,000 | 200 – 500 |

Run this quick check in Python to see which types would be flagged:

```python
import pandas as pd
labels = pd.read_csv("my_data_labels.csv", header=0)
counts = labels["cell_type"].value_counts()
print(counts[counts < 50])  # types with fewer than 50 cells
```

---

## Placing Your Files

Put the expression matrix and labels files in:

```
GARAGE/
├── data/
│   ├── cell_types/
│   │   └── my_data_labels.csv       ← your labels file
│   └── expression_matrix/
│       └── my_data_counts.csv       ← your expression matrix
```

The paths are relative to `config.py`'s `DATA_DIR`:

```python
DATA_DIR = os.path.join(REPO_ROOT, "data")
CELL_TYPES_DIR = os.path.join(DATA_DIR, "cell_types")
EXPRESSION_DIR = os.path.join(DATA_DIR, "expression_matrix")
```

---

## Running GARAGE on Your Data

Once your dataset is registered and files are in place:

```bash
python -m data_generation.garage --dataset my_dataset
python -m data_validation.data_validation \
    --dataset my_dataset \
    --gen_csv data/gen_data/my_dataset_data_mixdata_iter3_top_426.csv \
    --method cv2
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError` | Expression or labels file not found. | Check that files exist in the expected subdirectory. |
| Shape mismatch | Expression matrix and labels have different row counts. | Verify that both files correspond to the same cells in the same order. |
| `ValueError` on `transpose` | Wrong transpose setting. | Try toggling `transpose` in `DATASET_CONFIG`. |
| GAT loss NaN | Zero-expression cells or genes. | Filter out cells/genes with zero total expression. |
| ARI = 0 | Feature selection selected uninformative genes. | Try `--method pca` or increase `n_genes`. |
| CUDA OOM | Dataset too large for GPU. | Reduce `leakage_fraction` or run on CPU (set `DEVICE = "cpu"` in `config.py`). |

---

## Example: Adding the CBMC Dataset from Scratch

For reference, here's how the CBMC dataset entry looks in `config.py`:

```python
"cbmc": {
    "expression_file": "cbmc_rna_scaled.csv",
    "label_file": "cell_type_cbmc.csv",
    "header": 0,
    "index_col": 0,
    "transpose": True,
    "label_header": 0,
    "label_col": "x",
    "rare_threshold": 200,
    "iter_map": [0, 1, 2, 3, 4, 5],
},
```

This handles a genes × cells matrix (hence `transpose: True`) with a header row and index column, cell type labels with a header and the label column named `"x"`.
