#!/usr/bin/env python
"""Create publication-ready summary tables from raw clustering results.

Extended table: per-seed, per-method, per-dataset, per-feature-selection
Main table:     per-method, per-dataset, per-feature-selection → mean±std (3 seeds)

Outputs:
  results/extended_table.csv
  results/main_table.csv

Usage:  python analysis/build_summary_tables.py
"""

import numpy as np
import pandas as pd
import os, warnings
from config import RESULTS_DIR
warnings.filterwarnings("ignore")

OUT_DIR = RESULTS_DIR
RAW_CSV = os.path.join(OUT_DIR, "raw_results.csv")


def fmt_mean_std(series):
    """Format as 'mean±std' to 4 decimal places, or '--' if all NaN."""
    vals = series.dropna()
    if len(vals) == 0:
        return "--"
    m = vals.mean()
    s = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}"


def main():
    df = pd.read_csv(RAW_CSV)

    print(f"Loaded {len(df)} raw results")

    df_raw = df.copy()
    df_raw["ARI"] = df_raw["ARI"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "--")
    df_raw["NMI"] = df_raw["NMI"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "--")

    df_raw = df_raw.sort_values(["Dataset", "Method", "FeatureSelection", "Seed"])
    df_raw.to_csv(os.path.join(OUT_DIR, "extended_table.csv"), index=False)
    print(f"Extended table saved: {len(df_raw)} rows")

    grouped = df.groupby(["Dataset", "Method", "FeatureSelection"])
    main_rows = []
    for (ds, method, fs), grp in grouped:
        ari_str = fmt_mean_std(grp["ARI"])
        nmi_str = fmt_mean_std(grp["NMI"])
        main_rows.append((ds, method, fs, ari_str, nmi_str))

    df_main = pd.DataFrame(main_rows, columns=[
        "Dataset", "Method", "FeatureSelection", "ARI", "NMI"])
    df_main = df_main.sort_values(["Dataset", "Method", "FeatureSelection"])
    df_main.to_csv(os.path.join(OUT_DIR, "main_table.csv"), index=False)

    print(f"\nMain table saved: {len(df_main)} rows")
    print(f"\n{'='*100}")
    print("  SUMMARY TABLE")
    print("  Per-method, per-dataset, per-feature-selection → mean±std (3 seeds)")
    print(f"{'='*100}")
    print(df_main.to_string(index=False))


if __name__ == "__main__":
    main()
