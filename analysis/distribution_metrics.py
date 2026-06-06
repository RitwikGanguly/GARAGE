#!/usr/bin/env python
"""Compute MMD and Sliced Wasserstein Distance for all methods and datasets.
Produces label-agnostic distributional similarity metrics.

Usage:  conda run -n ritwik_base python run_distribution_metrics.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────── config ────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR      = os.path.join(REPO_ROOT, "data")
GEN_DIR       = os.path.join(REPO_ROOT, "data", "gen_data")
OUT_CSV       = os.path.join(REPO_ROOT, "results", "table_distribution_metrics.csv")
SUBSAMPLE     = 2000
N_PROJECTIONS = 200

# Match the exact iter version used by GARAGE for each dataset
GARAGE_ITER = {"Yan": 3, "Pollen": 5, "CBMC": 3, "Muraro": 3}

DATASET_ORDER = ["Yan", "Pollen", "CBMC", "Muraro"]
METHOD_ORDER  = ["GAN", "LSH-GAN", "GARAGE"]


# ──────────────────── data loaders ────────────────────
def load_real(dataset):
    if dataset == "CBMC":
        df = pd.read_csv(os.path.join(REAL_DIR, "cbmc_rna_scaled.csv"),
                         index_col=0, header=0).T
        return df.values.astype(np.float64)
    elif dataset == "Muraro":
        df = pd.read_csv(os.path.join(REAL_DIR, "muraro_expression_matrix.csv"), header=0)
        return df.values.astype(np.float64)
    elif dataset == "Pollen":
        df = pd.read_csv(os.path.join(REAL_DIR, "pollen_process.txt"), header=None)
        return df.values.astype(np.float64)
    elif dataset == "Yan":
        df = pd.read_csv(os.path.join(REAL_DIR, "yan_process.csv"), header=None)
        return df.values.astype(np.float64)
    raise ValueError(dataset)


def load_synthetic(dataset, method):
    lds = dataset.lower()
    if method == "GARAGE":
        fmap = {"cbmc": "cbmc_data_mixdata_iter3_top_1579.csv",
                "muraro": "muraro_data_mixdata_iter3_top_426.csv",
                "pollen": "pollen_data_mixdata_iter5_top_60_new.csv",
                "yan": "yan_data_mixdata_iter3_top_20.csv"}
        path = os.path.join(GEN_DIR, fmap[lds])
        fake = pd.read_csv(path, index_col=0).values.astype(np.float64)
        if dataset == "Yan" and fake.shape[0] < fake.shape[1]:
            fake = fake.T
    elif method == "GAN":
        it = GARAGE_ITER[dataset]
        path = os.path.join(GEN_DIR, "gan",
                            f"{lds}_gan_generated_mixdata_iter{it}.csv")
        fake = pd.read_csv(path, header=None).values.astype(np.float64)
    elif method == "LSH-GAN":
        it = GARAGE_ITER[dataset]
        path = os.path.join(GEN_DIR, "lsh_gan",
                            f"{lds}_lsh_gan_generated_mixdata_iter{it}.csv")
        fake = pd.read_csv(path, header=None).values.astype(np.float64)
    else:
        raise ValueError(method)
    return fake


# ──────────────────── MMD ────────────────────
def mmd_rbf(real, fake):
    rng = np.random.RandomState(42)
    n, m = real.shape[0], fake.shape[0]
    if n > SUBSAMPLE:
        real = real[rng.choice(n, SUBSAMPLE, replace=False)]
        n = real.shape[0]
    if m > SUBSAMPLE:
        fake = fake[rng.choice(m, SUBSAMPLE, replace=False)]
        m = fake.shape[0]

    combined = np.vstack([real, fake])
    sq = pdist(combined, "sqeuclidean")
    sigma = np.median(np.sqrt(sq + 1e-12))
    if sigma < 1e-8:
        sigma = 1.0

    Kxx = np.exp(-cdist(real, real, "sqeuclidean") / (2 * sigma**2))
    Kyy = np.exp(-cdist(fake, fake, "sqeuclidean") / (2 * sigma**2))
    Kxy = np.exp(-cdist(real, fake, "sqeuclidean") / (2 * sigma**2))

    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
         - 2.0 * np.mean(Kxy)
    return max(0.0, mmd2)


# ──────────────────── SWD ────────────────────
def sliced_wasserstein(real, fake, seed=42):
    rng = np.random.RandomState(seed)
    if real.shape[0] > SUBSAMPLE:
        real = real[rng.choice(real.shape[0], SUBSAMPLE, replace=False)]
    if fake.shape[0] > SUBSAMPLE:
        fake = fake[rng.choice(fake.shape[0], SUBSAMPLE, replace=False)]

    d = real.shape[1]
    distances = np.empty(N_PROJECTIONS)
    for i in range(N_PROJECTIONS):
        th = rng.randn(d)
        th /= np.linalg.norm(th) + 1e-12
        distances[i] = wasserstein_distance(real.dot(th), fake.dot(th))
    return np.mean(distances)


# ──────────────────── main ────────────────────
def main():
    rows = []
    for dataset in DATASET_ORDER:
        print(f"\n=== {dataset} ===")
        real = load_real(dataset)
        print(f"  Real data: {real.shape}")

        for method in METHOD_ORDER:
            try:
                fake = load_synthetic(dataset, method)
                # ensure same n_features
                assert real.shape[1] == fake.shape[1], \
                    f"Feature mismatch: real {real.shape[1]} vs {method} {fake.shape[1]}"
                mmd_val = mmd_rbf(real, fake)
                swd_val = sliced_wasserstein(real, fake)
                rows.append((dataset, method, round(mmd_val, 6),
                             round(swd_val, 6), "No"))
                print(f"  {method:8s}  MMD = {mmd_val:.6f}  SWD = {swd_val:.6f}")
            except Exception as e:
                print(f"  {method:8s}  SKIP: {e}")

    df = pd.DataFrame(rows, columns=[
        "Dataset", "Method", "MMD ↓",
        "Sliced Wasserstein Distance ↓", "Feature selection used?"])

    df.to_csv(OUT_CSV, index=False)
    print(f"\n=== Table 4: Label-agnostic distributional similarity ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
