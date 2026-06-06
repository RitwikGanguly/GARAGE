#!/usr/bin/env python
"""MMD Analysis: Maximum Mean Discrepancy with RBF kernel (median heuristic)
Compares real vs synthetic data on the full preprocessed gene-expression matrix
before feature selection. Lower MMD = better distributional agreement.

Usage:
    python mmd_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────── config ────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(REPO_ROOT, "data")
GEN_DIR = os.path.join(REPO_ROOT, "data", "gen_data")
OUT_CSV = os.path.join(REPO_ROOT, "results", "mmd_results.csv")
SUBSAMPLE = 2000  # optional subsample for speed on large datasets; set None to disable

DATASETS = ["Yan", "Pollen", "CBMC", "Muraro"]


def load_real_data(dataset):
    if dataset == "CBMC":
        df = pd.read_csv(os.path.join(REAL_DIR, "cbmc_rna_scaled.csv"),
                         index_col=0, header=0).T
        real = df.values.astype(np.float64)
    elif dataset == "Muraro":
        df = pd.read_csv(os.path.join(REAL_DIR, "muraro_expression_matrix.csv"),
                         header=0)
        real = df.values.astype(np.float64)
    elif dataset == "Pollen":
        df = pd.read_csv(os.path.join(REAL_DIR, "pollen_process.txt"),
                         header=None)
        real = df.values.astype(np.float64)
    elif dataset == "Yan":
        df = pd.read_csv(os.path.join(REAL_DIR, "yan_process.csv"),
                         header=None)
        real = df.values.astype(np.float64)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return real


def load_garage_data(dataset):
    file_map = {
        "CBMC":   "cbmc_data_mixdata_iter3_top_1579.csv",
        "Muraro": "muraro_data_mixdata_iter3_top_426.csv",
        "Pollen": "pollen_data_mixdata_iter5_top_60_new.csv",
        "Yan":    "yan_data_mixdata_iter3_top_20.csv",
    }
    fpath = os.path.join(GEN_DIR, file_map[dataset])
    fake = pd.read_csv(fpath, index_col=0).values.astype(np.float64)
    # Yan GARAGE was saved transposed (genes x cells) — correct it
    if dataset == "Yan" and fake.shape[0] < fake.shape[1]:
        fake = fake.T
    return fake


def load_gan_data(dataset, iter_idx):
    file_map = {
        "CBMC":   f"cbmc_gan_generated_mixdata_iter{iter_idx}.csv",
        "Muraro": f"muraro_gan_generated_mixdata_iter{iter_idx}.csv",
        "Pollen": f"pollen_gan_generated_mixdata_iter{iter_idx}.csv",
        "Yan":    f"yan_gan_generated_mixdata_iter{iter_idx}.csv",
    }
    fpath = os.path.join(GEN_DIR, "gan", file_map[dataset])
    df = pd.read_csv(fpath, header=None)
    fake = df.values.astype(np.float64)
    return fake


def load_lsh_gan_data(dataset, iter_idx):
    file_map = {
        "CBMC":   f"cbmc_lsh_gan_generated_mixdata_iter{iter_idx}.csv",
        "Muraro": f"muraro_lsh_gan_generated_mixdata_iter{iter_idx}.csv",
        "Pollen": f"pollen_lsh_gan_generated_mixdata_iter{iter_idx}.csv",
        "Yan":    f"yan_lsh_gan_generated_mixdata_iter{iter_idx}.csv",
    }
    fpath = os.path.join(GEN_DIR, "lsh_gan", file_map[dataset])
    df = pd.read_csv(fpath, header=None)
    fake = df.values.astype(np.float64)
    return fake


def rbf_kernel(X, Y, sigma):
    D2 = cdist(X, Y, "sqeuclidean")
    return np.exp(-D2 / (2.0 * sigma ** 2))


def mmd_rbf(real, fake, sigma=None):
    n, m = real.shape[0], fake.shape[0]
    if SUBSAMPLE is not None:
        if n > SUBSAMPLE:
            idx = np.random.RandomState(42).choice(n, SUBSAMPLE, replace=False)
            real = real[idx]
            n = real.shape[0]
        if m > SUBSAMPLE:
            idx = np.random.RandomState(42).choice(m, SUBSAMPLE, replace=False)
            fake = fake[idx]
            m = fake.shape[0]

    if sigma is None:
        combined = np.vstack([real, fake])
        sq_dists = pdist(combined, "sqeuclidean")
        sigma = np.median(np.sqrt(sq_dists + 1e-12))
        if sigma < 1e-8:
            sigma = 1.0

    K_xx = rbf_kernel(real, real, sigma)
    K_yy = rbf_kernel(fake, fake, sigma)
    K_xy = rbf_kernel(real, fake, sigma)

    mmd2 = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1)) \
         + (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) \
         - 2.0 * np.mean(K_xy)

    return max(0.0, mmd2)  # clamp tiny negatives


def main():
    results = []
    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        try:
            real = load_real_data(dataset)
        except Exception as e:
            print(f"  Could not load real data: {e}")
            continue

        # ── GARAGE ──
        try:
            garage = load_garage_data(dataset)
            val = mmd_rbf(real, garage)
            results.append((dataset, "GARAGE", val, "No"))
            print(f"  GARAGE  MMD = {val:.6f}")
        except Exception as e:
            print(f"  GARAGE  SKIP: {e}")

        # ── GAN (try iter0..iter5, take the one with closest size to real) ──
        for iter_i in range(0, 6):
            try:
                gan_data = load_gan_data(dataset, iter_i)
                val = mmd_rbf(real, gan_data)
                results.append((dataset, f"GAN_iter{iter_i}", val, "No"))
                print(f"  GAN_{iter_i}  MMD = {val:.6f}")
                break  # just first available
            except Exception:
                continue
        else:
            print("  GAN     SKIP – no generated data found")

        # ── LSH-GAN ──
        for iter_i in range(0, 6):
            try:
                lsh_data = load_lsh_gan_data(dataset, iter_i)
                val = mmd_rbf(real, lsh_data)
                results.append((dataset, f"LSH-GAN_iter{iter_i}", val, "No"))
                print(f"  LSH-GAN_{iter_i}  MMD = {val:.6f}")
                break
            except Exception:
                continue
        else:
            print("  LSH-GAN SKIP – no generated data found")

    # ── Write results ──
    df = pd.DataFrame(results, columns=["Dataset", "Method", "MMD", "FeatureSelection"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved to {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
