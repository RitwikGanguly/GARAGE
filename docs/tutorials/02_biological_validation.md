# Tutorial: Biological Validation

This tutorial demonstrates how to validate that GARAGE-generated synthetic data preserves biologically meaningful properties — specifically, that rare cell type marker genes are enriched in the cells the GAT selects as "important."

**Dataset:** CBMC (bone marrow mononuclear cells, 7,895 cells, 13 cell types).

**Prerequisite:** Run GARAGE on CBMC first (`python -m data_generation.garage --dataset cbmc`).

---

## What Biological Validation Measures

GARAGE's biological validation asks two questions:

1. **Are GAT-selected cells enriched for rare cell types?**  
   If the priority-weight mechanism works, high-attention cells should contain a disproportionately large share of rare cell types.

2. **Do high-attention cells express known marker genes for rare cell types?**  
   This confirms that the GAT's attention mechanism captures biologically meaningful signal, not just statistical artefacts.

---

## Step 1: Run the Main Biological Validation

```bash
python biological_analysis/biological_validation.py
```

### What happens

1. **Loads CBMC data** — expression matrix (7,895 × 2,000) and cell-type labels (13 types).

2. **Trains the GAT classifier** — same architecture as `garage.py`:
   - 2 GATConv layers (heads: 8, 1).
   - Priority weight = 2.0 on rare cell types.
   - 7,501 training iterations.

3. **Extracts attention weights** from the second GAT layer:
   - Per-cell attention scores are saved to `results/attention_weights.csv`.
   - Cells are split into **HIGH** (top 20 %) and **LOW** (bottom 20 %) attention groups.

4. **Cell-type enrichment analysis:**
   - For each cell type, computes the observed/expected ratio in the HIGH-attention group.
   - Fisher's exact test: is enrichment significantly non-random?

5. **Marker-gene expression analysis:**
   - Defines a curated list of rare-cell marker genes (e.g., CD34 for haematopoietic stem cells).
   - Computes mean expression of each marker in HIGH vs. LOW vs. ALL cells.
   - Reports $\log_2$ fold change and Wilcoxon rank-sum p-value.

6. **Positive-rate analysis (cell-type level):**
   - For each cell type and its marker genes, computes the fraction of cells expressing any marker.
   - Compares this positive rate in HIGH vs. ALL cells.
   - Fisher's exact test for significance.

---

## Step 2: Interpret the Results

### Output files

| File | Contents |
|------|----------|
| `results/attention_weights.csv` | Per-cell attention scores (sorted highest to lowest). |
| `results/biological_validation.csv` | Expression of all marker genes in HIGH vs. LOW vs. ALL, with log₂FC and p-values. |
| `results/positive_rate_per_celltype.csv` | Per-cell-type marker positive rate in HIGH vs. ALL, with Fisher p-values. |

### Console output (interpretation guide)

The script prints a self-interpreting summary. Key patterns to look for:

**Good signs:**

```
✓ HIGH-attention group is enriched for CD34+ HSCs (observed/expected = 3.2, p < 1e-6)
✓ CD34 expression: 2.4-fold higher in HIGH vs. LOW (Wilcoxon p < 1e-10)
✓ 68 % of HIGH cells are positive for their type's marker genes vs. 32 % in ALL
```

**Warning signs:**

```
✗ No significant enrichment of rare cell types in the HIGH group
✗ Marker gene expression is not different between HIGH and LOW (p > 0.05)
```

If you see warning signs, try:
- Increasing `priority_weight` in `config.py` (e.g., from 2.0 to 4.0).
- Increasing `leakage_fraction` (e.g., from 0.2 to 0.3).
- Checking that `rare_threshold` in `DATASET_CONFIG['cbmc']` is set correctly (should capture the types you want to study).

---

## Step 3: Marker Gene Clustering

```bash
python biological_analysis/marker_gene_clustering.py
```

This script performs a grid search over clustering parameters (feature selection method × number of genes × resolution) and reports ARI/NMI/F1 for each combination — specifically focused on whether marker genes for rare cell types improve clustering quality.

---

## Step 4: Held-out Rare Cell Utility

```bash
python biological_analysis/rare_cell_utility.py
```

This experiment quantifies how much GARAGE-generated data helps a downstream classifier detect rare cell types. See [Tutorial: Rare Cell Experiment](tutorials/03_rare_cell_experiment) for a detailed walkthrough.

---

## Biological Interpretation for Papers

When writing up results, the following narrative is typical:

> "Biological validation confirmed that cells selected by GARAGE's GAT attention mechanism are significantly enriched for rare haematopoietic cell types (CD34+ HSCs: 3.2× enrichment, Fisher p < 1e-6; erythroblasts: 2.1× enrichment, p < 1e-4). Known marker genes for these rare populations — including KLF1, GATA1, and HBG2 — were expressed at 2–5× higher levels in the high-attention group compared to the low-attention group. The per-cell-type marker positive rate was 68 % in the high-attention subset vs. 32 % in the full dataset (p < 1e-4), confirming that the GAT's priority-weight mechanism selects cells with genuine biological relevance."

---

## Customising for Your Own Dataset

To run biological validation on your own data:

1. Edit `biological_analysis/biological_validation.py`:
   - Change the dataset loading to use your expression matrix and labels.
   - Define marker genes for your rare cell types.
2. Edit `biological_analysis/marker_gene_clustering.py` similarly.

A future version of GARAGE will accept a `--marker_genes` JSON or CSV argument for ad-hoc marker gene lists.
