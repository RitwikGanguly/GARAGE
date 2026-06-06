# Biological Analysis

Experiments demonstrating the biological relevance of GARAGE-generated data and the GAT attention mechanism.

## Files

| File | Description |
|---|---|
| `rare_cell_utility.py` | Held-out rare-cell classification utility: trains a classifier on real data augmented with synthetic cells and evaluates rare-cell recall/F1/Macro-F1. |
| `marker_gene_clustering.py` | Marker-gene-based clustering evaluation: selects literature-validated marker genes, clusters real data on those genes, reports ARI/NMI. |
| `biological_validation.py` | GAT attention-weight biological validation: trains the GAT on CBMC data, extracts per-cell attention weights, computes enrichment of rare-cell-type marker genes in high-vs-low-attention cells. |
