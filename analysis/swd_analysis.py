#!/usr/bin/env python
"""SWD Analysis: Sliced Wasserstein Distance on the full gene-expression space.
Lower distance = better distributional agreement.

Usage:
    python swd_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────── config ────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(REPO_ROOT, "data")
GEN_DIR = os.path.join(REPO_ROOT, "data", "gen_data")
OUT_CSV = os.path.join(REPO_ROOT, "results", "swd_results.csv")
N_PROJECTIONS = 200     # random projections for SWD
SUBSAMPLE     = 2000     # optional subsample; set None to disable

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


def sliced_wasserstein(real, fake, n_proj=N_PROJECTIONS, seed=42):
    rng = np.random.RandomState(seed)

    if SUBSAMPLE is not None:
        if real.shape[0] > SUBSAMPLE:
            real = real[rng.choice(real.shape[0], SUBSAMPLE, replace=False)]
        if fake.shape[0] > SUBSAMPLE:
            fake = fake[rng.choice(fake.shape[0], SUBSAMPLE, replace=False)]

    n_features = real.shape[1]
    swds = np.empty(n_proj)

    for i in range(n_proj):
        theta = rng.randn(n_features)
        theta /= np.linalg.norm(theta) + 1e-12
        proj_real = real.dot(theta)
        proj_fake = fake.dot(theta)
        swds[i] = wasserstein_distance(proj_real, proj_fake)

    return np.mean(swds)


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
            val = sliced_wasserstein(real, garage)
            results.append((dataset, "GARAGE", val, "No"))
            print(f"  GARAGE    SWD = {val:.6f}")
        except Exception as e:
            print(f"  GARAGE    SKIP: {e}")

        # ── GAN ──
        for iter_i in range(0, 6):
            try:
                gan_data = load_gan_data(dataset, iter_i)
                val = sliced_wasserstein(real, gan_data)
                results.append((dataset, f"GAN_iter{iter_i}", val, "No"))
                print(f"  GAN_{iter_i}     SWD = {val:.6f}")
                break
            except Exception:
                continue
        else:
            print("  GAN       SKIP")

        # ── LSH-GAN ──
        for iter_i in range(0, 6):
            try:
                lsh_data = load_lsh_gan_data(dataset, iter_i)
                val = sliced_wasserstein(real, lsh_data)
                results.append((dataset, f"LSH-GAN_iter{iter_i}", val, "No"))
                print(f"  LSH-GAN_{iter_i} SWD = {val:.6f}")
                break
            except Exception:
                continue
        else:
            print("  LSH-GAN   SKIP")

    df = pd.DataFrame(results, columns=["Dataset", "Method",
                                         "SlicedWassersteinDistance",
                                         "FeatureSelectionUsed"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved to {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
