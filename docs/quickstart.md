# Quickstart

This guide gets you from zero to your first GARAGE-generated synthetic dataset in under 5 minutes.

---

## Prerequisites

- Python 3.12.5 installed with Conda.
- The repository cloned and dependencies installed (`requirements_garage.txt`).
- See [Installation](installation) if you haven't done this yet.

---

## 1. Activate the Environment

```bash
conda activate venv_garage
cd GARAGE
```

---

## 2. Verify Your Setup

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from config import DATASET_CONFIG, GARAGE_DEFAULTS; print(list(DATASET_CONFIG))"
```

Expected output (roughly):

```
PyTorch 2.4.0
CUDA: True
['yan', 'pollen', 'cbmc', 'muraro']
```

---

## 3. Run GARAGE on a Small Dataset (Yan)

Yan is the smallest dataset (124 cells, 6 types) — it trains in ~2 minutes on CPU:

```bash
python -m data_generation.garage --dataset yan
```

**What this does:**
1. Loads the Yan expression matrix and cell-type labels.
2. Stage 1: Trains a GAT classifier on a KNN cell-cell graph, identifies seed cells.
3. Stage 2: Trains a GAN with hybrid noise+seed input batches (20,001 iterations).
4. Saves the generated data to `data/gen_data/` (e.g., `yan_data_mixdata_iter3_top_426.csv`).

The console output will show loss values for both stages. Watch for the GAT loss decreasing and the GAN generator/discriminator losses converging.

---

## 4. Validate the Generated Data

```bash
python -m data_validation.data_validation \
    --dataset yan \
    --gen_csv data/gen_data/yan_data_mixdata_iter3_top_426.csv \
    --method cv2 \
    --plot_umap
```

**What this does:**
1. Loads the generated and real data.
2. Applies CV² feature selection (100 genes).
3. Runs Leiden clustering over a resolution sweep.
4. Reports ARI, NMI, and macro-F1 scores.
5. Optionally saves UMAP plots to `results/`.

Expected output (approximate):

```
Dataset: yan | Feature selection: cv2 | top_genes: 100
Best resolution: 1.05
ARI:  0.7243
NMI:  0.8011
F1:   0.6938
```

---

## 5. Check the Wasserstein Distance

```bash
python -m data_generation.wasserstein_distance \
    --dataset yan \
    --gen_csv data/gen_data/yan_data_mixdata_iter3_top_426.csv
```

A good Wasserstein distance for Yan is $< 0.01$.

---

## 6. View the Results

```bash
ls results/
```

Typical output:

```
yan_cv2_ari_nmi_f1.csv
yan_cv2_umap_real.pdf
yan_cv2_umap_gen.pdf
yan_wasserstein_distance.csv
```

---

## Next Steps

Now that you've seen the basic workflow:

1. **Run on larger datasets:** Replace `--dataset yan` with `pollen`, `cbmc`, or `muraro`.
2. **Plug in your own data:** See [Preparing Your Data](prepare_your_data).
3. **Full end-to-end tutorial:** [Tutorial: End-to-End Guide](tutorials/01_end_to_end).
4. **Biological validation:** [Tutorial: Biological Validation](tutorials/02_biological_validation).
5. **Benchmark against SOTA:** [Tutorial: Benchmark Against SOTA](tutorials/04_benchmark_against_sota).

---

## Common First-Time Issues

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'torch_geometric'` | Install PyG: `pip install torch_geometric` (see `requirements_garage.txt`). |
| `CUDA out of memory` | Reduce `leakage_fraction` in `config.py` or run on CPU. |
| `Generated file not found` | Check the filename pattern in the console output; it depends on the iteration and dataset. |
| `Leiden returned 1 cluster` | Increase the resolution sweep range in `data_validation.py`. |
| `ARI = 0.0` | Try a different feature selection method (`--method pca` or `--method fano`). |
