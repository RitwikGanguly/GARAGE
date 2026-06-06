#!/usr/bin/env python
"""Generate figures of Wasserstein distance vs. leakage percentage
for the four benchmark datasets.

Usage:  python analysis/plot_wasserstein_vs_leakage.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR

DEMO_DIR = RESULTS_DIR
os.makedirs(DEMO_DIR, exist_ok=True)

leakage = [10, 20, 30]

datasets = {
    "Yan":    [2.0277, 1.9218, 1.6862],
    "Pollen": [1.2982, 1.2185, 1.4592],
    "CBMC":   [0.0020, 0.0032, 0.0033],
    "Muraro": [0.0415, 0.0434, 0.0441],
}

styles = {
    "Yan":    {"color": "#2e86ab", "marker": "o"},
    "Pollen": {"color": "#a23b72", "marker": "s"},
    "CBMC":   {"color": "#f18f01", "marker": "^"},
    "Muraro": {"color": "#3a7d44", "marker": "D"},
}

fig, ax = plt.subplots(figsize=(7, 5))

for name, values in datasets.items():
    s = styles[name]
    ax.plot(leakage, values,
            color=s["color"], marker=s["marker"],
            markersize=8, linewidth=2,
            label=name)

ax.set_yscale("log")
ax.set_xticks(leakage)
ax.set_xticklabels([f"{p}%" for p in leakage])
ax.set_xlabel("Leakage percentage", fontsize=12)
ax.set_ylabel("Wasserstein distance", fontsize=12)
# ax.set_title("Wasserstein distance vs. leakage percentage",
#              fontsize=13, fontweight="bold")
ax.legend(title="Dataset", loc="best", fontsize=10, title_fontsize=11)
ax.grid(False)

for ext in ("pdf", "png"):
    out = os.path.join(DEMO_DIR, f"wassertstein_vs_leakage.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")

plt.close()
