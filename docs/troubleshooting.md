# Troubleshooting

Common problems and their solutions.

---

## Installation issues

### PyTorch Geometric installation fails

```text
ImportError: No module named 'torch_geometric'
```

**Fix:** Install PyG matching your PyTorch/CUDA version:

```bash
# PyTorch 2.4 with CUDA 12.1
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Alternatively, use Conda:

```bash
conda install pyg -c pyg
```

### NumPy/SciPy version mismatch

```text
UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required
```

**Fix:** This warning appears when SciPy was compiled against an older NumPy. It's generally harmless for GARAGE's usage. To fix, re-install the environment:

```bash
conda create --name venv_garage python=3.12.5
conda activate venv_garage
pip install -r requirements_garage.txt
```

---

## Runtime errors

### CUDA out of memory

```text
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Fixes (in order of preference):**

1. Reduce `leakage_fraction` in `config.py` (0.2 → 0.1).
2. Reduce the GAN batch size (edit `garage.py`).
3. Run on CPU by modifying `garage.py`:

   ```python
   DEVICE = torch.device("cpu")
   ```

4. For CBMC specifically, the 7,895 × 2,000 matrix with KNN graph can be memory-intensive. Consider using a CPU-only run.

### File not found

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data/cell_types/...'
```

**Fix:** Ensure you're running from the repository root:

```bash
cd /path/to/GARAGE
python -m data_generation.garage --dataset yan
```

### Data shape mismatch

```text
ValueError: shapes (124,10564) and (150,100) not aligned
```

**Fix:** The generated file may be from a different dataset than the one specified. Check:
- `--dataset` flag matches the input the generator was trained on.
- The `--gen_csv` file corresponds to the correct dataset.

### Leiden returns 1 cluster

```text
Best resolution: 0.10 | ARI: 0.00  (only 1 cluster found)
```

**Fix:** The resolution sweep range is too low for your data. Edit the range in `data_validation/data_validation.py`:

```python
RESOLUTION_RANGES = {
    "my_dataset": np.arange(0.01, 5.01, 0.05),  # Wider range
}
```

### Leiden returns too many clusters (1 per cell)

```text
Best resolution: 3.00 | ARI: 0.00  (n_clusters = n_cells)
```

**Fix:** The resolution is too high. Narrow the range:

```python
RESOLUTION_RANGES = {
    "my_dataset": np.arange(0.05, 1.01, 0.05),
}
```

---

## Training problems

### GAT loss doesn't decrease

```text
epoch    0: loss=2.302, acc=0.140
epoch 1000: loss=2.301, acc=0.142
epoch 7000: loss=2.298, acc=0.145
```

**Fix:** The GAT is not learning. Check:
1. The expression matrix might be on a scale that confuses the network. Try scaling to $[0, 1]$.
2. The cell-type labels might not be parseable. Check `DATASET_CONFIG['your_dataset']['label_col']`.
3. Reduce the learning rate in the GAT's Adam optimiser.

### GAN generator loss diverges

```text
iter    0: D_loss=1.231, G_loss=0.823
iter 1000: D_loss=0.035, G_loss=8.421
iter 2000: D_loss=0.001, G_loss=12.743
```

**Fix:** The discriminator is too strong relative to the generator. Solutions:
1. Reduce discriminator learning rate (`d_lr`: 0.0004 → 0.0001).
2. Increase `ng_steps` (more generator updates per discriminator step: 2 → 3).
3. Decrease `nd_steps` (fewer discriminator updates: 5 → 2).
4. Remove label smoothing (set `label_smooth_real = 1.0`, `label_smooth_fake = 0.0`).

### Mode collapse (generator produces only one cell type)

Check with UMAP:

```bash
python -m data_validation.data_validation \
    --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
    --method cv2 --plot_umap
```

If the UMAP shows only 1–2 clusters while the real data shows 10+, the generator has mode-collapsed.

**Fix:**
1. Increase `leakage_fraction` (0.2 → 0.3).
2. Increase `priority_weight` (2.0 → 4.0).
3. Train for fewer iterations (20,001 → 10,000) — sometimes longer training worsens mode collapse.

### NaN losses

```text
iter 500: D_loss=nan, G_loss=nan
```

**Fix:** Numerical instability. Solutions:
1. Reduce learning rates.
2. Add gradient clipping to the optimisers.
3. Check that the input data doesn't contain NaN values.
4. Increase the epsilon in Adam (`torch.optim.Adam(..., eps=1e-4)`).

---

## Validation problems

### All metrics are zero

```text
ARI: 0.0000, NMI: 0.0000, F1: 0.0000
```

**Fix:** Nothing was learned. Check:
1. The generated CSV contains variation (not all zeros/ones).
2. Feature selection picked genes that exist in both real and generated data.
3. The `--dataset` flag matches the data the model was trained on.

### UMAPs look identical

If the real and generated UMAPs are near-identical, the leakage fraction may be too high — the generator is simply copying real data. Reduce $\lambda$.

---

## Build and docs

### Sphinx build fails with "WARNING: document isn't included in any toctree"

All `.md` and `.rst` files in the `docs/` directory must be included in the `index.rst` toctree or excluded in `conf.py` → `exclude_patterns`.

### ReadTheDocs build fails on external images

ReadTheDocs cannot access external URLs during build. All images should be hosted locally in `docs/images/` and referenced with relative paths.

---

## Still stuck?

1. Check the [FAQ](faq) for common questions.
2. Open a [GitHub Issue](https://github.com/RitwikGanguly/GARAGE/issues) with:
   - Full error traceback.
   - Your `config.py` settings.
   - Operating system and Python version.
   - Whether you're using CPU or GPU.
