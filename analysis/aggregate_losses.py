#!/usr/bin/env python
"""
Aggregate the long leakage_losses.csv (4 datasets x 4 leakages x N
iterations) into a compact per-(dataset, leakage) summary table.

Output: results/leakage_losses_summary.csv
"""

import os
import numpy as np
import pandas as pd

from config import RESULTS_DIR

IN_PATH = os.path.join(RESULTS_DIR, "leakage_losses.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "leakage_losses_summary.csv")

# ── read ──
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} records from {IN_PATH}")
print(f"  Datasets: {sorted(df['dataset'].unique())}")
print(f"  Leakage levels: {sorted(df['leakage'].unique())}")
print(f"  Iterations per (dataset, leakage): {df.groupby(['dataset','leakage']).size().iloc[0]}")

# ── drop the iter-0 spike (initial random output) so the "training regime"
#    statistics are not dominated by the bootstrap step ──
df_train = df[df["iteration"] > 0].copy()

# ── group & aggregate ──
agg = (
    df_train.groupby(["dataset", "leakage"], as_index=False)
    .agg(
        g_loss_mean=("g_loss", "mean"),
        g_loss_std=("g_loss", "std"),
        g_loss_min=("g_loss", "min"),
        g_loss_max=("g_loss", "max"),
        d_loss_mean=("d_loss", "mean"),
        d_loss_std=("d_loss", "std"),
        d_loss_min=("d_loss", "min"),
        d_loss_max=("d_loss", "max"),
        n_iterations_logged=("iteration", "count"),
    )
)

# ── final (last-logged) losses — converged-state snapshot ──
last_iter = df.loc[df.groupby(["dataset", "leakage"])["iteration"].idxmax()][
    ["dataset", "leakage", "iteration", "g_loss", "d_loss"]
].rename(columns={"iteration": "final_iteration",
                  "g_loss": "g_loss_final",
                  "d_loss": "d_loss_final"})

agg = agg.merge(last_iter, on=["dataset", "leakage"])

# ── nice column order ──
cols = [
    "dataset", "leakage",
    "g_loss_mean", "g_loss_std",
    "d_loss_mean", "d_loss_std",
]
agg = agg[cols].sort_values(["dataset", "leakage"]).reset_index(drop=True)

# ── round for readability ──
for c in cols[2:]:
    agg[c] = agg[c].astype(float).round(4)

# ── save ──
agg.to_csv(OUT_PATH, index=False)
print(f"\nSaved summary ({len(agg)} rows) to {OUT_PATH}\n")

# ── print pretty table ──
print("=" * 100)
print("GAN Training Stability Ablation — Per (Dataset, Leakage) Summary")
print("(mean / std computed over iterations 1k – 20k; final = loss at iter 20k)")
print("=" * 100)
print(
    f"{'dataset':<8} | {'lambda':<6} | {'G mean ± std':<22} | {'D mean ± std':<22} | {'G final':<8} | {'D final':<8}"
)
print("-" * 100)
print("=" * 100)
