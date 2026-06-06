# Tutorial: Rare Cell Experiment

This tutorial demonstrates the held-out rare-cell utility experiment — a direct test of whether GARAGE-generated synthetic data improves a classifier's ability to detect rare cell types.

**Target audience:** Researchers interested in data augmentation or rare-cell biology.

**Time:** ~30 minutes on GPU.

---

## Experimental Design

For each dataset (Yan, Pollen, CBMC, Muraro):

1. **Train/test split:** Hold out 50 % of the **rarest** cell type as unseen test cells. Non-rare cells are also split 50/50 (no overlap between train and test).
2. **Re-train generative models** on the training data only:
   - GARAGE
   - Standard GAN
   - LSH-GAN
3. **Generate synthetic rare cells:** Each model produces $10 \times n_{\text{train\_rare}}$ synthetic cells labelled as the rare type.
4. **Train a Random Forest classifier** on:
   - Real data only (baseline)
   - Real + GAN synthetic
   - Real + LSH-GAN synthetic
   - Real + GARAGE synthetic
5. **Evaluate on held-out test set:**
   - **Rare-cell Recall:** Fraction of true rare cells correctly identified.
   - **Rare-cell F1:** Harmonic mean of rare-cell precision and recall.
   - **Macro-F1:** Standard macro-F1 over all cell types.

---

## Step 1: Run the Experiment

```bash
conda activate venv_garage
python biological_analysis/rare_cell_utility.py
```

**What happens:**

1. Loads all four datasets via `config.py`.
2. For each dataset, identifies the rarest cell type.
3. Performs the 50/50 stratified train/test split (rare + non-rare).
4. Re-trains GARAGE (GAT + GAN), standard GAN, and LSH-GAN **from scratch** on the training data.
5. Generates $10 \times n_{\text{train\_rare}}$ synthetic cells from each model.
6. Trains and evaluates 4 Random Forest classifiers:
   - `RF_real`: trained on real training data only.
   - `RF_real+gan`: trained on real + GAN synthetic.
   - `RF_real+lsh`: trained on real + LSH-GAN synthetic.
   - `RF_real+garage`: trained on real + GARAGE synthetic.
7. Reports Recall, F1, and Macro-F1 on the held-out test set.

---

## Step 2: Interpret the Results

### Console output (representative)

```
Dataset: muraro | Rare type: delta_cell (n_train=63, n_test=63)
================================================================
Model              Rare Recall   Rare F1     Macro-F1
----------------------------------------------------------------
RF_real              0.492         0.511       0.724
RF_real+gan          0.524         0.538       0.731
RF_real+lsh          0.556         0.562       0.739
RF_real+garage       0.683         0.697       0.775
----------------------------------------------------------------
Best: GARAGE (+0.191 recall, +0.186 F1 over real-only)
```

### What to look for

| Pattern | Interpretation |
|---------|---------------|
| **GARAGE > GAN ≈ LSH-GAN** | GAT seeding provides genuine rare-cell benefit beyond random subsampling. |
| **All generative models > real-only** | Synthetic data helps, but GARAGE helps the most. |
| **GARAGE ≈ GAN ≈ LSH-GAN** | The dataset may not have very rare types, or the train/test split is easy. Check your `rare_threshold`. |
| **Synthetic degrades performance** | The generative models may be producing unrealistic rare cells. Check losses and Wasserstein distance. |

---

## Step 3: Statistical Rigour

The script includes several controls:

- **Fixed seed (42):** All models use the same random seed for reproducibility.
- **Multiple metrics:** Recall, F1, and Macro-F1 — a model that overfits to rare cells at the expense of abundant types will drop in Macro-F1.
- **PCA pre-processing:** For small datasets (Yan: 124 cells × 10,564 genes), PCA (50 components) is applied before Random Forest to avoid model degeneracy.

### Output files

| File | Contents |
|------|----------|
| `results/rare_cell_utility.csv` | Full per-dataset, per-model metrics. |
| `results/rare_cell_utility_summary.csv` | Aggregated mean ± std across datasets. |

---

## Customising the Experiment

### Change the rare-cell type

Edit the rare-type selection logic in `rare_cell_utility.py` (it currently selects the type with the fewest cells):

```python
# To target a specific type:
rare_type = "NK cell"
```

### Change the augmentation ratio

The script generates $10 \times$ training rare cells. To change this:

```python
# In the loop: change the multiplier
n_synthetic = 10 * n_train_rare  # e.g., 5× or 20×
```

### Add a new model

Add a new classifier training loop after the existing ones:

```python
# Example: adding a VAE-based augmentation
# Real + VAE synthetic
clf_vae = RandomForestClassifier(n_estimators=100, random_state=42)
X_train_vae = np.vstack([X_train_real, X_train_vae_synth])
y_train_vae = np.hstack([y_train_real, y_train_vae_synth])
clf_vae.fit(X_train_vae, y_train_vae)
recall_vae = recall_score(y_test, clf_vae.predict(X_test), average=None)[rare_idx]
```

---

## Related Pages

- [Tutorial: Biological Validation](tutorials/02_biological_validation) — marker gene enrichment analysis.
- [Tutorial: End-to-End Guide](tutorials/01_end_to_end) — full pipeline walkthrough.
- [How‑to: Ablations](howto/ablations) — leakage fraction and multi-seed studies.
