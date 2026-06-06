# How-to: Biological Validation

A recipe for verifying that GARAGE preserves biological signal for rare cell types.

## Goal

Test whether GAT-high-attention cells are enriched for rare cell types and express known marker genes.

## Prerequisites

- GARAGE run completed on CBMC (or a dataset with defined marker genes).
- `biological_analysis/biological_validation.py` — ships with the repo.

## Steps

### 1. Run the main validation

```bash
python biological_analysis/biological_validation.py
```

### 2. Run marker gene clustering

```bash
python biological_analysis/marker_gene_clustering.py
```

### 3. Examine the output files

```bash
# Check attention weights
head -20 results/attention_weights.csv

# Check marker gene expression
cat results/biological_validation.csv | column -s, -t | head

# Check per-cell-type positive rates
cat results/positive_rate_per_celltype.csv | column -s, -t | head
```

### 4. Key metrics to verify

From the console output or CSV files:

| Metric | Good sign | Warning sign |
|--------|-----------|--------------|
| Rare-type enrichment (HIGH vs. LOW) | observed/expected > 2.0, p < 0.01 | observed/expected ≈ 1.0 |
| Marker gene log₂FC | > 1.0 (2× higher in HIGH cells) | < 0.5 |
| Wilcoxon p-value | < 0.01 | > 0.05 (not significant) |
| Positive rate (cell-type level) | HIGH > ALL | HIGH ≈ ALL |

## Related

- [Tutorial: Biological Validation](tutorials/02_biological_validation) — full walkthrough.
- [Tutorial: Rare Cell Experiment](tutorials/03_rare_cell_experiment) — held-out rare cell classification.
