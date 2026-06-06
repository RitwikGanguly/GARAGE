#!/usr/bin/env python
"""
Wasserstein distance between real and generated scRNA-seq distributions.
=========================================================================

For each (dataset, leakage_level), this script:
  1. Loads the real expression matrix.
  2. Loads the GARAGE‑generated data for the requested iteration.
  3. Computes the 1‑Wasserstein (Earth Mover's) distance between the two
     histograms using the Python Optimal Transport library (POT).

Datasets
--------
  Yan    — header=None, transpose, gen data header=0+index_col=0
  Pollen — header=None, no transpose, gen data header=0+index_col=0
  CBMC   — header=0+index_col=0, transpose, gen data header=0+index_col=0
  Muraro — header=0, no transpose, gen data header=0+index_col=0

Usage
-----
    python -m data_generation.wasserstein_distance \
        --dataset muraro --leakage 0.2 --gen_iter 3 \
        --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist

try:
    from config import DATASET_CONFIG, DATA_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR


def load_real(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    fpath = os.path.join(DATA_DIR, "expression_matrix", cfg["expression_file"])
    rk = {"header": cfg.get("header", 0)}
    if "index_col" in cfg:
        rk["index_col"] = cfg["index_col"]
    df = pd.read_csv(fpath, **rk)
    if cfg.get("transpose", False):
        df = df.T
    df.reset_index(drop=True, inplace=True)
    return df.values.astype(np.float64)


def load_generated(gen_csv, transpose=False):
    """Load generated CSV.  Default: header=None (no index_col)."""
    gen = pd.read_csv(gen_csv, header=None).values.astype(np.float64)
    if transpose and gen.shape[1] < gen.shape[0]:
        gen = gen.T
    return gen


def wasserstein_distance(data, generated):
    """
    Compute 1-Wasserstein (Earth Mover's) distance using POT.emd2.

    Each row is treated as a sample; cost matrix is Euclidean between rows.
    """
    if data.shape[1] != generated.shape[1]:
        raise ValueError(f"Feature mismatch: real has {data.shape[1]} cols, "
                         f"gen has {generated.shape[1]} cols")
    w1 = np.ones(len(data)) / len(data)
    w2 = np.ones(len(generated)) / len(generated)
    M = cdist(data, generated, metric="euclidean")
    return ot.emd2(w1, w2, M, numItermax=100000)


def main():
    parser = argparse.ArgumentParser(
        description="Wasserstein distance between real and generated scRNA-seq data")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG),
                        help="Dataset name (default: muraro)")
    parser.add_argument("--gen_csv", type=str, required=True,
                        help="Path to the generated CSV file")
    parser.add_argument("--gen_iter", type=int, default=None,
                        help="(Optional) iteration index for logging")
    parser.add_argument("--leakage", type=float, default=0.2,
                        help="Leakage fraction used for generation (for logging)")
    args = parser.parse_args()

    real = load_real(args.dataset)
    gen = load_generated(args.gen_csv)

    print(f"Real data:  {real.shape}")
    print(f"Gen data:   {gen.shape}")

    wd = wasserstein_distance(real, gen)
    print(f"\nWasserstein distance (iter={args.gen_iter}, leakage={args.leakage}):  "
          f"{wd:.6f}")


if __name__ == "__main__":
    main()
