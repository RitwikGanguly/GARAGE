# Tutorial: End-to-End Guide

This tutorial walks through the complete GARAGE pipeline — from loading real scRNA-seq data to evaluating the synthetic output — using the Muraro dataset as an example.

**Target audience:** Researchers who want to understand and reproduce every step of the GARAGE workflow.

**Time:** ~20 minutes (Muraro) on a GPU; ~40 minutes on CPU.

**Prerequisites:** [Installation](installation) and [Quickstart](quickstart) completed.

---

## Step 1: Understand Your Dataset

Before running GARAGE, inspect your data:

```bash
python -c "
from config import DATASET_CONFIG
cfg = DATASET_CONFIG['muraro']
print('Expression file:', cfg['expression_file'])
print('Label file:', cfg['label_file'])
print('Transpose:', cfg['transpose'])
print('Rare threshold:', cfg['rare_threshold'])
"
```

Output:

```
Expression file: muraro_expression_matrix.csv
Label file: muraro_cell_types.csv
Transpose: False
Rare threshold: 200
```

Let's check the cell-type distribution:

```python
import pandas as pd
labels = pd.read_csv('data/cell_types/muraro_cell_types.csv')
print(labels['cell_type'].value_counts())
```

This tells you which types fall below `rare_threshold` (200) and will receive priority attention.

---

## Step 2: Run GARAGE (GAT + GAN)

```bash
python -m data_generation.garage --dataset muraro
```

### What happens under the hood

**Stage 1 — GAT Cell Selection:**

1. `load_dataset("muraro")` loads the expression matrix (2,126 cells × 19,156 genes) and labels from `config.py`.
2. Flags cell types with < 200 cells as "rare" (priority nodes).
3. Builds a $k$-NN graph ($k=5$) using Ball Tree on the expression matrix.
4. Converts the adjacency to a PyG `edge_index`.
5. Trains a `GATClassifier` for 7,501 iterations:
   - 2 GATConv layers (heads: 8, 1).
   - After layer 1, priority node features are scaled by `1 + priority_weight` (default: 3.0 total).
   - Loss: cross-entropy over cell-type classes.
6. After training, extracts attention weights from layer 2.
7. Aggregates into per-cell scores and selects the top $n \cdot \lambda$ cells as seeds.

**Stage 2 — GAN Generation:**

1. Initialises `Generator` (MLP: 2 hidden layers of 1024) and `Discriminator` (MLP: 512, 256).
2. For each of 20,001 iterations:
   - Constructs a hybrid batch: $\lceil(1 - \lambda) \cdot B\rceil$ noise vectors + $\lfloor \lambda \cdot B\rfloor$ seed cells.
   - **Discriminator update** ($\times 5$): Real batch → `D(x)`; generated batch → `D(G(z))`. BCE loss with label smoothing (0.9/0.1).
   - **Generator update** ($\times 2$): New hybrid batch → `G(z)` → `D(G(z))`. Minimise $-\log D(G(z))$.
3. Every 1,000 iterations, logs losses to `results/losses.csv`.
4. At the end, writes the generated data to `data/gen_data/muraro_data_mixdata_iter3_top_426.csv`.

### Console output (abridged)

```
=== Loading Muraro dataset ===
n_cells=2126, n_genes=19156, n_classes=10, rare_types=2

=== Stage 1: GAT Cell Selection ===
Training GAT... (7501 epochs, priority_weight=2.0)
  epoch    0: loss=2.302, acc=0.140
  epoch 1000: loss=0.387, acc=0.892
  epoch 7000: loss=0.082, acc=0.971
Extracting attention weights...
Selected k=425 seed cells

=== Stage 2: GAN Generation ===
seed=42, leakage=0.2, g_lr=0.0002, d_lr=0.0004
  iter    0: D_loss=1.231, G_loss=0.823
  iter  1000: D_loss=0.752, G_loss=1.412
  iter  5000: D_loss=0.611, G_loss=1.658
  iter 10000: D_loss=0.592, G_loss=1.701
  iter 20000: D_loss=0.581, G_loss=1.712
Saved: data/gen_data/muraro_data_mixdata_iter3_top_426.csv
```

**Convergence signs to watch for:**
- GAT loss decreases and accuracy exceeds 0.85.
- D_loss and G_loss stabilise (not diverging, not oscillating wildly).
- D_loss should be $\approx \log(2) \approx 0.69$ at equilibrium.

---

## Step 3: Compute Wasserstein Distance

```bash
python -m data_generation.wasserstein_distance \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv
```

Output example:

```
Wasserstein distance: 0.00734
Interpretation: Excellent match (WD < 0.01).
```

A Wasserstein distance below 0.01 on Muraro indicates high distributional fidelity.

---

## Step 4: Feature Selection and Clustering Validation

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method cv2 \
    --n_genes 100 \
    --plot_umap
```

### What happens

1. Loads the generated data CSV.
2. Sets up an `AnnData` object for both real and generated data.
3. Applies CV² feature selection:
   - On generated data: computes $\text{CV}^2_g = \sigma_g^2 / \mu_g^2$ for each gene.
   - Selects the top-100 genes by CV².
4. Filters the real data to the same 100 genes.
5. Runs PCA (50 components) on the real (filtered) data.
6. Computes a KNN neighbourhood graph (n_neighbors=15).
7. Runs Leiden clustering over a resolution sweep (Muraro: 0.10 to 3.01, step 0.01).
8. At each resolution:
   - Computes ARI between Leiden clusters and ground-truth labels.
   - Records the best ARI, the corresponding NMI and macro-F1.
9. Optionally generates UMAP plots of real and generated data coloured by Leiden cluster.

### Output (approximate)

```
Dataset: muraro | FS: cv2 | top_genes: 100
--------------------------------------------------------------------------------
Resolution sweep: 0.10 — 3.01
  Best ARI:  0.6734 at resolution 1.42
  NMI:       0.7519
  Macro-F1:  0.6108
--------------------------------------------------------------------------------
UMAP saved: results/muraro_cv2_umap_real.pdf
UMAP saved: results/muraro_cv2_umap_gen.pdf
```

### Interpreting the Scores

| Metric | Muraro expected | What it means |
|--------|----------------|---------------|
| ARI | 0.50 – 0.75 | The Leiden clustering of generated data recovers ~50–75% of the ground-truth structure after correcting for chance. |
| NMI | 0.65 – 0.85 | Normalised mutual info — high for Muraro given its 10 cell types |
| Macro-F1 | 0.50 – 0.70 | Per-type F1 average; lower than ARI/NMI because it equally weights small types |

---

## Step 5: Biological Validation

```bash
python biological_analysis/biological_validation.py
python biological_analysis/marker_gene_clustering.py
```

This checks whether GARAGE-generated data preserves marker genes for rare cell types. See [Tutorial: Biological Validation](tutorials/02_biological_validation) for details.

---

## Step 6: Benchmark Against SOTA Models

```bash
# Run all baselines for Muraro
python -m benchmarking.sota.gan --dataset muraro
python -m benchmarking.sota.wgan --dataset muraro
python -m benchmarking.sota.fgan --dataset muraro
python -m benchmarking.sota.vae --dataset muraro
python -m benchmarking.sota.lsh_gan --dataset muraro

# Run scRNA-seq-specific baselines
python -m benchmarking.scrna_seq_specific.scgan --dataset muraro
python -m benchmarking.scrna_seq_specific.scvae --dataset muraro
python -m benchmarking.scrna_seq_specific.scdiffusion --dataset muraro
python -m benchmarking.scrna_seq_specific.gan_ros --dataset muraro
python -m benchmarking.scrna_seq_specific.vae_ros --dataset muraro
```

Then aggregate results:

```bash
python analysis/distribution_metrics.py  --dataset muraro
python analysis/clustering_evaluation.py --dataset muraro
python analysis/build_summary_tables.py  --dataset muraro
```

See [Tutorial: Benchmark Against SOTA](tutorials/04_benchmark_against_sota).

---

## Step 7: Run Ablation Studies

```bash
python ablation_study/leakage_ablation.py
python ablation_study/multi_seed_synthesis.py
python analysis/plot_wasserstein_vs_leakage.py
```

Produces the leakage-fraction sweep (λ = 0.0, 0.1, 0.2, 0.3 across 4 datasets) and multi-seed reproducibility analysis.

---

## Complete Workflow Script

Here's a single script to run the full Muraro pipeline:

```bash
#!/bin/bash
# Full GARAGE pipeline — Muraro

# 1. Generate
python -m data_generation.garage --dataset muraro

# 2. Wasserstein
python -m data_generation.wasserstein_distance \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv

# 3. Clustering validation (CV²)
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method cv2 --plot_umap

# 4. Clustering validation (Fano)
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method fano

# 5. Clustering validation (PCA loading)
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method pca

# 6. Biological validation
python biological_analysis/biological_validation.py
python biological_analysis/marker_gene_clustering.py

echo "Done. Results in results/"
```

---

## What's Next

- **Tune hyper-parameters:** Edit `config.py` to adjust `leakage_fraction`, learning rates, etc.
- **Add your own dataset:** [Preparing Your Data](prepare_your_data).
- **Rare cell experiments:** [Tutorial: Rare Cell Experiment](tutorials/03_rare_cell_experiment).
- **Interpretation:** [Interpret Outputs How‑to](howto/interpret_outputs).
