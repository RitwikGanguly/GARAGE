#!/usr/bin/env python
"""
Feature selection for scRNA-seq data (Python port of feature_selection.R).
=======================================================================

Three methods are provided, mirroring the original R implementation:

  1. **Fano factor** (`fano_selection`)
     Selects genes with the *lowest* variance-to-mean ratio (Fano factor).
     R equivalent: ``Fano_ind()`` at feature_selection.R:12-19

  2. **PCA loading** (`pca_loading_selection`)
     Selects top-k genes by absolute loading on PC1–PC3.
     R equivalent: ``PCA_loading()`` at feature_selection.R:10-21

  3. **CV² (normalised coefficient-of-variation squared)** (`cv2_selection`)
     Computes per-gene dispersion (variance/mean), bins by mean expression,
     normalises by bin median/MAD, and returns the top-k by normalised dispersion.
     R equivalent: ``CV2()`` at feature_selection.R:21-55

Strategy
--------
Feature selection from generated data applied to real data.
Feature selection from real data applied to real data.
Feature selection from combined (gen + real) data applied to real data.

Usage
-----
    python -m data_validation.feature_selection \
        --method cv2 --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv \
        --real_csv data/muraro_expression_matrix.csv --transpose False --header 0
"""

import argparse
import os
import numpy as np
import pandas as pd

try:
    from config import DATASET_CONFIG, DATA_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR


# ═══════════════════════════════════════════════════════════════════════════
#  Selection methods  (each returns an np.ndarray of column indices)
# ═══════════════════════════════════════════════════════════════════════════

def fano_selection(data, n_genes=100):
    """
    Select *n_genes* with the LOWEST Fano factor (variance / mean).

    R equivalent: Fano_ind() in feature_selection.R.
    """
    var = np.var(data, axis=0)
    mean = np.mean(data, axis=0) + 1e-12
    fano = var / mean
    return np.argsort(fano)[:n_genes]


def pca_loading_selection(data, n_genes=100, n_components=3):
    """
    Rank genes by the maximum absolute loading across the first *n_components*
    principal components.

    R equivalent: PCA_loading() in feature_selection.R.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, min(data.shape)), random_state=42)
    pca.fit(data)
    loadings = np.abs(pca.components_)  # shape (n_components, n_genes)
    max_loadings = loadings.max(axis=0)
    return np.argsort(max_loadings)[::-1][:n_genes]


def cv2_selection(data, n_genes=100):
    """
    Select top *n_genes* by normalised CV² dispersion.

    Procedure:
      1. For each gene, compute dispersion = variance / mean.
      2. Partition genes into bins by mean-expression quantiles.
      3. Normalise each gene's dispersion by its bin's median and MAD.
      4. Return the *n_genes* with the largest normalised dispersion.

    R equivalent: CV2() in feature_selection.R.
    """
    means = data.mean(axis=0)
    eps = 1e-8
    means_safe = np.where(means == 0, eps, means)

    vars_ = data.var(axis=0, ddof=1)
    dispersion = vars_ / means_safe

    # Bin genes by mean expression (10th – 95th percentiles, step 5 %)
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


FEATURE_SELECTORS = {
    "fano": fano_selection,
    "pca": pca_loading_selection,
    "cv2": cv2_selection,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline  (mirrors feature_selection.R lines 59-118)
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_selection(gen_csv, real_csv, out_dir, method="cv2",
                          header=None, transpose=False, index_col=None,
                          label_csv=None, n_genes=100):
    """
    Feature-selection pipeline.

    Reads generated and real CSV files, applies *method* to each, then writes
    three filtered versions of the real data to *out_dir*:

      datafilt1.csv  — real data filtered to features selected from gen data
      datafilt2.csv  — real data filtered to features selected from real data
      datafilt_combined.csv — real data filtered to features from gen+real

    Parameters
    ----------
    gen_csv, real_csv : str  Paths to the CSV files.
    out_dir : str  Output directory.
    method : str  One of {"fano", "pca", "cv2"}.
    header, transpose, index_col : passed to pd.read_csv for the real data.
    label_csv : str or None  If the real CSV includes a label column, separate it.
    n_genes : int  Number of features to select (default 100).
    """
    selector = FEATURE_SELECTORS[method]
    os.makedirs(out_dir, exist_ok=True)

    # Load generated data
    gen = pd.read_csv(gen_csv, header=None, index_col=0).values.astype(np.float64)

    # Load real data
    rk = {"header": header}
    if index_col is not None:
        rk["index_col"] = index_col
    real = pd.read_csv(real_csv, **rk)
    if transpose:
        real = real.T
    real = real.values.astype(np.float64)

    print(f"Gen shape:  {gen.shape}")
    print(f"Real shape: {real.shape}")
    assert real.shape[1] == gen.shape[1], "Column mismatch between real and gen data"

    # Feature selection from gen data
    sel_gen = selector(gen, n_genes=n_genes)

    # Feature selection from real data
    sel_real = selector(real, n_genes=n_genes)

    # Feature selection from combined data
    combined = np.vstack([gen, real])
    sel_comb = selector(combined, n_genes=n_genes)

    # Filter real data
    datafilt1 = real[:, sel_gen]
    datafilt2 = real[:, sel_real]
    datafilt_comb = real[:, sel_comb]

    # Save
    pd.DataFrame(datafilt1).to_csv(
        os.path.join(out_dir, "datafilt1.csv"), index=False)
    pd.DataFrame(datafilt2).to_csv(
        os.path.join(out_dir, "datafilt2.csv"), index=False)
    pd.DataFrame(datafilt_comb).to_csv(
        os.path.join(out_dir, "datafilt_combined.csv"), index=False)

    print(f"\nSaved filtered real data ({n_genes} features, method={method}):")
    print(f"  {os.path.join(out_dir, 'datafilt1.csv')}")
    print(f"  {os.path.join(out_dir, 'datafilt2.csv')}")
    print(f"  {os.path.join(out_dir, 'datafilt_combined.csv')}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection for scRNA-seq data validation")
    parser.add_argument("--method", type=str, default="cv2",
                        choices=list(FEATURE_SELECTORS),
                        help="Feature selection method (default: cv2)")
    parser.add_argument("--gen_csv", type=str, required=True,
                        help="Path to generated data CSV")
    parser.add_argument("--real_csv", type=str, required=True,
                        help="Path to real expression matrix CSV")
    parser.add_argument("--out_dir", type=str, default="results/feature_selection",
                        help="Output directory for filtered CSVs")
    parser.add_argument("--header", type=int, default=None,
                        help="Header row for real CSV (0 = first row, None = no header)")
    parser.add_argument("--transpose", action="store_true",
                        help="Transpose the real expression matrix after loading")
    parser.add_argument("--index_col", type=int, default=None,
                        help="Index column for real CSV (0 = first col)")
    parser.add_argument("--n_genes", type=int, default=100,
                        help="Number of features to select")
    args = parser.parse_args()

    run_feature_selection(
        gen_csv=args.gen_csv,
        real_csv=args.real_csv,
        out_dir=args.out_dir,
        method=args.method,
        header=args.header,
        transpose=args.transpose,
        index_col=args.index_col,
        n_genes=args.n_genes,
    )


if __name__ == "__main__":
    main()
