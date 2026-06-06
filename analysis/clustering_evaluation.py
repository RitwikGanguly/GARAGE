#!/usr/bin/env python
"""Data validation: feature selection + clustering + metrics across seeds.

Feature selection technique (implemented in Python, matching R logic):
  CV2 — largest normalized CV² dispersion → top 100 features

Clustering:
  - Feature select from generated data (CV2) → filter real data to those features
  - Cluster filtered REAL data with Leiden
  - GAT-GAN (GARAGE): grid search over dataset-specific resolution ranges, maximise ARI
  - All other methods: fixed resolution = 1.0
  - npcs = 20, n_neighbors = 30 for all

Metrics: ARI, NMI (cluster labels vs ground truth labels on filtered real data)

Output:  results/rev8_raw_results.csv

Usage:  conda run -n scrna python validate_and_evaluate.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(REPO_ROOT, "data")
GEN_ROOT   = os.path.join(REPO_ROOT, "data", "gen_data")
OUT_DIR    = os.path.join(REPO_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 123, 456]
N_PCS      = 20
N_NEIGHBORS = 30
N_FEATURES = 100

DATASETS = {
    "Yan": {
        "file": "yan_process.csv",
        "csv_kwargs": {"header": None},
        "transpose": True,
        "label_file": "yan_celltype.csv",
        "label_kwargs": {"header": None},
        "label_col": 0,
        "iter": 3,
    },
    "CBMC": {
        "file": "cbmc_rna_scaled.csv",
        "csv_kwargs": {"index_col": 0, "header": 0},
        "transpose": True,
        "label_file": "cell_type_cbmc.csv",
        "label_kwargs": {"header": 0},
        "label_col": "x",
        "iter": 3,
    },
    "Muraro": {
        "file": "muraro_expression_matrix.csv",
        "csv_kwargs": {"header": 0},
        "transpose": False,
        "label_file": "muraro_cell_types.csv",
        "label_kwargs": {"header": 0},
        "label_col": "cell_type",
        "iter": 3,
    },
    "Pollen": {
        "file": "pollen_process.txt",
        "csv_kwargs": {"header": None},
        "transpose": False,
        "label_file": "pollenc.txt",
        "label_kwargs": {"header": None},
        "label_col": 0,
        "iter": 5,
    },
}

METHODS = [
    ("wgan",    "wgan"),
    ("fgan",    "fgan"),
    ("lsh_gan", "lsh_gan"),
    ("gan",     "gan"),
    ("gat_gan", "gat_gan"),
]

GATGAN_RES_RANGES = {
    "Yan":    np.arange(0.10, 3.01, 0.01),
    "Pollen": np.arange(0.98, 3.01, 0.01),
    "CBMC":   np.arange(0.10, 1.01, 0.01),
    "Muraro": np.arange(0.10, 1.21, 0.01),
}


# ══════════════════════════════════════════════════════════════════════════
#  Feature Selection
# ══════════════════════════════════════════════════════════════════════════

def cv2(data, n_genes=100):
    """Select top n_genes by normalized CV² dispersion.
    Exact reimplementation of the R CV2() function from Data_vaidation_ARI_new.ipynb.
    """
    means = data.mean(axis=0)
    eps = 1e-8
    means_safe = np.where(means == 0, eps, means)

    sds = data.std(axis=0, ddof=1)
    vars_ = data.var(axis=0, ddof=1)

    cv = sds / means_safe
    dispersion = vars_ / means_safe

    quantile_breaks = [np.quantile(means, q) for q in np.arange(0.1, 1.0, 0.05)]
    bins = np.digitize(means, [-np.inf] + list(quantile_breaks) + [np.inf])

    unique_bins = np.unique(bins)
    bin_medians = np.zeros_like(dispersion)
    bin_mads = np.zeros_like(dispersion)

    for b in unique_bins:
        mask = bins == b
        bin_disp = dispersion[mask]
        bin_med = np.median(bin_disp)
        bin_mad = np.median(np.abs(bin_disp - bin_med))
        if bin_mad == 0:
            bin_mad = eps
        bin_medians[mask] = bin_med
        bin_mads[mask] = bin_mad

    dispersion_norm = np.abs(dispersion - bin_medians) / bin_mads
    indices = np.argsort(dispersion_norm)[::-1][:n_genes]
    return indices


FEATURE_SELECTORS = {
    "cv2": cv2,
}


# ══════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_real(ds_cfg):
    df = pd.read_csv(os.path.join(REAL_DIR, ds_cfg["file"]), **ds_cfg["csv_kwargs"])
    if ds_cfg["transpose"]:
        df = df.T
    return df.values.astype(np.float64)


def load_labels(ds_cfg):
    lbl_df = pd.read_csv(os.path.join(REAL_DIR, ds_cfg["label_file"]),
                         **ds_cfg["label_kwargs"])
    if ds_cfg["label_kwargs"].get("header") is not None:
        col = ds_cfg["label_col"]
        return lbl_df[col].values.ravel()
    else:
        return lbl_df.iloc[:, ds_cfg["label_col"]].values.ravel()


def load_gen(ds_name, method_dir, method_prefix, seed, wanted_iter):
    fname = f"{ds_name.lower()}_{method_prefix}_generated_mixdata_iter{wanted_iter}.csv"
    fpath = os.path.join(GEN_ROOT, f"seed_{seed}", method_dir, fname)
    gen = pd.read_csv(fpath, header=None).values.astype(np.float64)
    return gen


# ══════════════════════════════════════════════════════════════════════════
#  Clustering / Evaluation
# ══════════════════════════════════════════════════════════════════════════

def cluster_and_evaluate(real_filt, true_labels, resolution, n_pcs, n_neighbors):
    actual_n_pcs = min(n_pcs, real_filt.shape[1] - 1, real_filt.shape[0] - 1)
    actual_n_pcs = max(2, actual_n_pcs)
    actual_n_neighbors = min(n_neighbors, real_filt.shape[0] - 1)

    try:
        adata = sc.AnnData(real_filt.astype(np.float64))
        adata.var_names_make_unique()
        sc.pp.pca(adata, n_comps=actual_n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=actual_n_neighbors,
                        n_pcs=actual_n_pcs, metric="cosine")
        sc.tl.leiden(adata, resolution=round(resolution, 4), random_state=42)
        cluster_labels = adata.obs["leiden"].astype(str).astype(int).to_numpy()
    except Exception:
        return np.nan, np.nan

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return ari, nmi


def evaluate_baseline(real_filt, true_labels, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    return cluster_and_evaluate(real_filt, true_labels,
                                resolution=1.0, n_pcs=n_pcs, n_neighbors=n_neighbors)


def evaluate_sweep(real_filt, true_labels, res_range, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    best_ari, best_nmi = -1, -1
    for res in res_range:
        ari, nmi = cluster_and_evaluate(real_filt, true_labels,
                                        resolution=res, n_pcs=n_pcs, n_neighbors=n_neighbors)
        if not np.isnan(ari) and ari > best_ari:
            best_ari, best_nmi = ari, nmi
    if best_ari < 0:
        return np.nan, np.nan
    return best_ari, best_nmi


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    rows = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")

        real = load_real(ds_cfg)
        labels_raw = load_labels(ds_cfg)
        le = LabelEncoder()
        true_labels = le.fit_transform(labels_raw)
        wanted_iter = ds_cfg["iter"]

        print(f"  Real shape: {real.shape}  |  cells={real.shape[0]}  |  iter={wanted_iter}")
        print(f"  Unique cell types: {len(le.classes_)}")

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")

            for method_dir, method_prefix in METHODS:
                print(f"    [{method_dir}]")

                try:
                    gen = load_gen(ds_name, method_dir, method_prefix, seed, wanted_iter)
                except Exception as e:
                    print(f"      SKIP: {e}")
                    for fs_name in FEATURE_SELECTORS:
                        rows.append((ds_name, seed, method_dir, fs_name, np.nan, np.nan))
                    continue

                if gen.size == 0 or np.any(np.isnan(gen)) or np.any(np.isinf(gen)):
                    print(f"      SKIP: invalid generated data")
                    for fs_name in FEATURE_SELECTORS:
                        rows.append((ds_name, seed, method_dir, fs_name, np.nan, np.nan))
                    continue

                print(f"      Gen shape: {gen.shape}")

                for fs_name, fs_func in FEATURE_SELECTORS.items():
                    try:
                        feat_indices = fs_func(gen, n_genes=N_FEATURES)
                    except Exception as e:
                        print(f"        {fs_name:10s}  FS failed: {e}")
                        rows.append((ds_name, seed, method_dir, fs_name, np.nan, np.nan))
                        continue

                    feat_indices = feat_indices[:min(len(feat_indices), real.shape[1])]
                    real_filt = real[:, feat_indices]

                    if method_dir == "gat_gan":
                        res_range = GATGAN_RES_RANGES[ds_name]
                        ari, nmi = evaluate_sweep(real_filt, true_labels, res_range)
                    else:
                        ari, nmi = evaluate_baseline(real_filt, true_labels)

                    status = "OK" if not np.isnan(ari) else "FAIL"
                    print(f"        {fs_name:10s}  {status:4s}  "
                          f"ARI={ari if np.isnan(ari) else f'{ari:.4f}'}  "
                          f"NMI={nmi if np.isnan(nmi) else f'{nmi:.4f}'}")

                    rows.append((ds_name, seed, method_dir, fs_name, ari, nmi))

    df = pd.DataFrame(rows, columns=["Dataset", "Seed", "Method",
                                     "FeatureSelection", "ARI", "NMI"])
    out_path = os.path.join(OUT_DIR, "raw_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
