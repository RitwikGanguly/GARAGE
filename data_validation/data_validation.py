#!/usr/bin/env python
"""
Data validation for GARAGE‑generated scRNA‑seq data.
=====================================================

Computes clustering quality metrics (ARI, NMI, macro-F1) by:
  1. Loading generated data (GARAGE or baselines).
  2. Loading the corresponding real data and ground-truth labels.
  3. Applying feature selection (CV², Fano, PCA loading) on the generated data
     then filtering the real data to those features.
  4. Clustering the filtered real data with Leiden (resolution sweep).
  5. Reporting ARI, NMI, and macro-F1 against ground truth.

Uses Scanpy for PCA, neighbourhood graph, Leiden clustering, and UMAP.
Matches the workflow in data_vaidation_garage.ipynb (original notebook).

Usage
-----
    python -m data_validation.data_validation \
        --dataset muraro --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
        --method cv2 --plot_umap
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

try:
    from config import DATASET_CONFIG, DATA_DIR, RESULTS_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

# Default Leiden resolution sweep ranges (tuned per dataset)
RESOLUTION_RANGES = {
    "yan": np.arange(0.80, 1.61, 0.01),
    "pollen": np.arange(0.10, 3.01, 0.01),
    "cbmc": np.arange(0.20, 0.81, 0.01),
    "muraro": np.arange(0.10, 3.01, 0.01),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Feature selection
# ═══════════════════════════════════════════════════════════════════════════

def cv2_selection(data, n_genes=100):
    means = data.mean(axis=0)
    eps = 1e-8
    means_safe = np.where(means == 0, eps, means)
    vars_ = data.var(axis=0, ddof=1)
    dispersion = vars_ / means_safe
    quantile_breaks = np.quantile(means, np.arange(0.1, 1.0, 0.05))
    bins = np.digitize(means, [-np.inf] + list(quantile_breaks) + [np.inf])
    bin_medians = np.zeros_like(dispersion)
    bin_mads = np.zeros_like(dispersion)
    for b in np.unique(bins):
        mask = bins == b
        bin_disp = dispersion[mask]
        bin_med = np.median(bin_disp)
        bin_mads[mask] = np.median(np.abs(bin_disp - bin_med))
        if bin_mads[mask].max() == 0:
            bin_mads[mask] = eps
        bin_medians[mask] = bin_med
    dispersion_norm = np.abs(dispersion - bin_medians) / bin_mads
    return np.argsort(dispersion_norm)[::-1][:n_genes]


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_real(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    fpath = os.path.join(DATA_DIR, "expression_matrix", cfg["expression_file"])
    rk = {"header": cfg.get("header", 0)}
    if "index_col" in cfg:
        rk["index_col"] = cfg["index_col"]
    df = pd.read_csv(fpath, **rk)
    if cfg.get("transpose", False):
        df = df.T
    return df.values.astype(np.float64)


def load_labels(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    lbl_path = os.path.join(DATA_DIR, "cell_types", cfg["label_file"])
    lk = {"header": cfg.get("label_header", None)}
    lbl_df = pd.read_csv(lbl_path, **lk)
    if cfg["label_header"] is not None:
        return lbl_df[cfg["label_col"]].values.ravel()
    return lbl_df.iloc[:, cfg["label_col"]].values.ravel()


def load_generated(gen_csv, n_features):
    gen = pd.read_csv(gen_csv, header=None).values.astype(np.float64)
    if gen.shape[0] < gen.shape[1] and gen.shape[1] == n_features:
        gen = gen.T
    return gen


# ═══════════════════════════════════════════════════════════════════════════
#  Clustering & evaluation
# ═══════════════════════════════════════════════════════════════════════════

def cluster_and_evaluate(real_filt, true_labels, resolution,
                         n_pcs=20, n_neighbors=30):
    actual_n_pcs = min(n_pcs, real_filt.shape[1] - 1, real_filt.shape[0] - 1)
    actual_n_pcs = max(2, actual_n_pcs)
    actual_nn = min(n_neighbors, real_filt.shape[0] - 1)

    adata = sc.AnnData(real_filt.astype(np.float64))
    adata.var_names_make_unique()

    try:
        sc.pp.pca(adata, n_comps=actual_n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=actual_nn,
                        n_pcs=actual_n_pcs, metric="cosine")
        sc.tl.leiden(adata, resolution=round(resolution, 4), random_state=42)
        y_pred = adata.obs["leiden"].astype(str).astype(int).to_numpy()
    except Exception:
        return np.nan, np.nan, np.nan, adata

    ari = adjusted_rand_score(true_labels, y_pred)
    nmi = normalized_mutual_info_score(true_labels, y_pred)
    macro_f1 = f1_score(true_labels, y_pred, average="macro", zero_division=0)
    return ari, nmi, macro_f1, adata


def sweep_resolution(real_filt, true_labels, res_range,
                     n_pcs=20, n_neighbors=30):
    best = {"ari": -1, "nmi": -1, "macro_f1": -1, "resolution": 0.0, "adata": None}
    for res in res_range:
        ari, nmi, mf1, adata = cluster_and_evaluate(
            real_filt, true_labels, resolution=res,
            n_pcs=n_pcs, n_neighbors=n_neighbors)
        if not np.isnan(ari) and ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "macro_f1": mf1,
                    "resolution": res, "adata": adata}
    if best["ari"] < 0:
        return np.nan, np.nan, np.nan, None
    return best["ari"], best["nmi"], best["macro_f1"], best["adata"]


def plot_umap(adata, title, save_path, dpi=300):
    try:
        sc.tl.umap(adata)
        fig = sc.pl.umap(adata, color=["leiden"],
                          title=title, show=False, return_fig=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  UMAP saved to {save_path}")
    except Exception as e:
        print(f"  UMAP skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data validation (ARI / NMI / macro-F1) for GARAGE-generated data")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG))
    parser.add_argument("--gen_csv", type=str, required=True,
                        help="Path to generated CSV (header=None)")
    parser.add_argument("--n_genes", type=int, default=100,
                        help="Number of features for CV² selection")
    parser.add_argument("--n_pcs", type=int, default=20)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--plot_umap", action="store_true",
                        help="Generate and save a UMAP plot")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory for UMAP plot (default: results/)")
    args = parser.parse_args()

    real = load_real(args.dataset)
    labels_raw = load_labels(args.dataset)
    le = LabelEncoder()
    true_labels = le.fit_transform(labels_raw)

    gen = load_generated(args.gen_csv, real.shape[1])
    print(f"Real: {real.shape}  |  Gen: {gen.shape}")

    # Feature selection on generated data → filter real
    feat_idx = cv2_selection(gen, n_genes=args.n_genes)
    real_filt = real[:, feat_idx[:min(len(feat_idx), real.shape[1])]]

    # Resolution sweep for GARAGE
    res_range = RESOLUTION_RANGES.get(args.dataset,
                                       np.arange(0.1, 3.01, 0.01))
    ari, nmi, mf1, adata = sweep_resolution(
        real_filt, true_labels, res_range,
        n_pcs=args.n_pcs, n_neighbors=args.n_neighbors)

    print(f"\n{'=' * 50}")
    print(f"  Results for {args.dataset.upper()}")
    print(f"  ARI      = {ari:.4f}" if not np.isnan(ari) else "  ARI = N/A")
    print(f"  NMI      = {nmi:.4f}" if not np.isnan(nmi) else "  NMI = N/A")
    print(f"  Macro-F1 = {mf1:.4f}" if not np.isnan(mf1) else "  F1 = N/A")
    print(f"{'=' * 50}")

    if args.plot_umap and adata is not None:
        out_dir = args.out_dir or RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir,
                                 f"umap_{args.dataset}_ari={ari:.4f}.png")
        plot_umap(adata, f"UMAP: ARI={ari:.4f}, NMI={nmi:.4f}", save_path)


if __name__ == "__main__":
    main()
