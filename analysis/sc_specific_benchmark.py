#!/usr/bin/env python
"""ARI/NMI/F1 benchmarking for scRNA-seq-specific baseline methods.
Fixed params: resolution=1.0, cosine distance, n_neighbors=30, n_pcs=20.
Feature selection: Fano factor (lowest variance-to-mean) → top 100 genes.
F1 is macro-averaged (f1_score with average='macro').

Usage:  conda run -n scrna python run_rev4_validation.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
import os, warnings
warnings.filterwarnings("ignore")

# ═══════════════════  CONFIG  ═══════════════════
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(REPO_ROOT, "data")
GEN_DIR_BASE = os.path.join(REPO_ROOT, "data", "gen_data")
OUT_CSV = os.path.join(REPO_ROOT, "results", "table_scrna_seq_benchmarks.csv")
N_GENES = 100

DATASETS_CONFIG = {
    "Yan":    {"real_file": "yan_process.csv",     "transpose": True,  "real_header": None,
               "label_file": "yan_celltype.csv",   "label_header": None, "label_col": 0},
    "Pollen": {"real_file": "pollen_process.txt",  "transpose": False, "real_header": None,
               "label_file": "pollenc.txt",        "label_header": None, "label_col": 0},
    "CBMC":   {"real_file": "cbmc_rna_scaled.csv", "transpose": True,  "real_header": 0, "real_index_col": 0,
               "label_file": "cell_type_cbmc.csv", "label_header": 0, "label_col": "x"},
    "Muraro": {"real_file": "muraro_expression_matrix.csv", "transpose": False, "real_header": 0,
               "label_file": "muraro_cell_types.csv", "label_header": 0, "label_col": "cell_type"},
}

ITER_MAP = {"Yan": 3, "Pollen": 5, "CBMC": 3, "Muraro": 3}

METHODS = [
    ("scGAN",         "scgan",       "scgan"),
    ("scVAE",         "scvae",       "scvae"),
    ("scDiffusion",   "scdiffusion", "scdiffusion"),
    ("GAN+ROS",       "gan_ros",     "gan_ros"),
    ("VAE+ROS",       "vae_ros",     "vae_ros"),
]


def load_real(dataset):
    cfg = DATASETS_CONFIG[dataset]
    rk = {"header": cfg["real_header"]}
    if "real_index_col" in cfg:
        rk["index_col"] = cfg["real_index_col"]
    real = pd.read_csv(os.path.join(REAL_DIR, cfg["real_file"]), **rk)
    if cfg["transpose"]:
        real = real.T
    real = real.values.astype(np.float64)

    lbl = pd.read_csv(os.path.join(REAL_DIR, cfg["label_file"]),
                      header=cfg["label_header"])
    if cfg["label_header"] is not None:
        lbl = lbl[cfg["label_col"]].values.ravel()
    else:
        lbl = lbl.iloc[:, 0].values.ravel()
    return real, lbl


def fano_selection(data, n_genes=N_GENES):
    """Select n_genes with LOWEST Fano factor (variance/mean)."""
    var = np.var(data, axis=0)
    mean = np.mean(data, axis=0) + 1e-12
    fano = var / mean
    return np.argsort(fano)[:n_genes]


def evaluate_ari(gen_data, real_data, labels, n_pcs=20, n_neighbors=30, resolution=1.0):
    """Feature selection on generated data → filter real data → Leiden on real data → ARI."""
    # Select 100 genes from generated data (lowest Fano factor)
    sel_idx = fano_selection(gen_data, N_GENES)

    # Filter real data to those genes
    real_filt = real_data[:, sel_idx]

    # Leiden clustering on filtered real data
    adata = sc.AnnData(real_filt.astype(np.float64))
    adata.var_names_make_unique()

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="cosine")
    sc.tl.leiden(adata, resolution=resolution, random_state=42)

    y_pred = adata.obs["leiden"].astype(str).astype(int).to_numpy()

    ari = adjusted_rand_score(labels_enc, y_pred)
    nmi = normalized_mutual_info_score(labels_enc, y_pred)
    macro_f1 = f1_score(labels_enc, y_pred, average="macro", zero_division=0)
    return ari, nmi, macro_f1


def main():
    rows = []
    for dataset in DATASETS_CONFIG:
        real, labels_raw = load_real(dataset)
        iter_idx = ITER_MAP[dataset]
        print(f"\n{'='*50}\n  {dataset}  |  iter={iter_idx}\n{'='*50}")

        for method_name, dir_name, prefix in METHODS:
            try:
                fpath = os.path.join(GEN_DIR_BASE, dir_name,
                                     f"{dataset.lower()}_{prefix}_mixdata_iter{iter_idx}.csv")
                gen = pd.read_csv(fpath, header=None).values.astype(np.float64)
                n_cells = gen.shape[0]

                ari, nmi, mf1 = evaluate_ari(gen, real, labels_raw)
                rows.append((dataset, method_name, round(ari, 4),
                             round(nmi, 4), round(mf1, 4)))
                print(f"  {method_name:15s}  cells={n_cells:6d}  ARI={ari:.4f}  NMI={nmi:.4f}  F1={mf1:.4f}")
            except Exception as e:
                print(f"  {method_name:15s}  SKIP: {e}")

    # GARAGE values (computed from the original pipeline)
    garage_values = {
        "Yan":    (0.935, 0.930, 0.307),
        "Pollen": (0.882, 0.896, 0.035),
        "CBMC":   (0.541, 0.613, 0.130),
        "Muraro": (0.375, 0.483, 0.205),
    }
    for dataset in DATASETS_CONFIG:
        ari, nmi, mf1 = garage_values[dataset]
        rows.append((dataset, "GARAGE", round(ari, 4), round(nmi, 4), round(mf1, 4)))
        print(f"  {'GARAGE':15s}  ARI={ari:.4f}  NMI={nmi:.4f}  F1={mf1:.4f}")

    df = pd.DataFrame(rows, columns=["Dataset", "Method", "ARI ↑", "NMI ↑", "F1 ↑"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*80}")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
