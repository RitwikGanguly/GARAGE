#!/usr/bin/env python
"""
GAN+ROS — Vanilla GAN trained on Random-Oversampled class-balanced data.
==========================================================================
Before training, data is ROS-resampled so every cell type has the same
number of samples.  Same GAN architecture as benchmark gan.py (32-tanh-32).
Adam, 200 epochs.

Usage:  python -m benchmarking.scrna_seq_specific.gan_ros --dataset muraro
Output: data/gen_data/gan_ros/<dataset>_gan_ros_mixdata_iter{iter}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.utils import resample

try:
    from config import DATASET_CONFIG, DATA_DIR, SCRNASEQ_DATASETS
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import DATASET_CONFIG, DATA_DIR, SCRNASEQ_DATASETS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 200
SEED = 42


class Generator(nn.Module):
    def __init__(self, zdim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 32), nn.Tanh(),
            nn.Linear(32, out_dim))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def load_real_with_labels(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    fpath = os.path.join(DATA_DIR, "expression_matrix", cfg["expression_file"])
    rk = {"header": cfg.get("header", 0)}
    if "index_col" in cfg:
        rk["index_col"] = cfg["index_col"]
    df = pd.read_csv(fpath, **rk)
    if cfg.get("transpose", False):
        df = df.T
    X = df.values.astype(np.float32)

    lbl_path = os.path.join(DATA_DIR, "cell_types", cfg["label_file"])
    lk = {"header": cfg.get("label_header", None)}
    lbl_df = pd.read_csv(lbl_path, **lk)
    if cfg["label_header"] is not None:
        y = lbl_df[cfg["label_col"]].values.ravel()
    else:
        y = lbl_df.iloc[:, cfg["label_col"]].values.ravel()
    return X, y


def oversample(X, y):
    classes = np.unique(y)
    max_cnt = max((y == c).sum() for c in classes)
    X_list, y_list = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        Xc, yc = X[idx], y[idx]
        if len(idx) < max_cnt:
            Xc, yc = resample(Xc, yc, replace=True, n_samples=max_cnt,
                              random_state=SEED)
        X_list.append(Xc)
        y_list.append(yc)
    return np.vstack(X_list), np.hstack(y_list)


def train(real, n_features, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = real.shape[0]

    G = Generator(LATENT_DIM, n_features).to(DEVICE)
    D = Discriminator(n_features).to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=1e-4)
    optD = optim.Adam(D.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    for _ in range(EPOCHS):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            b = torch.tensor(real[bi], device=DEVICE)
            bs = b.shape[0]
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            with autocast("cuda"):
                fake = G(z).detach()
                d_loss = bce(D(b), torch.ones(bs, 1, device=DEVICE)) \
                       + bce(D(fake), torch.zeros(bs, 1, device=DEVICE))
            optD.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optD)
            scaler.update()

            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            with autocast("cuda"):
                g_loss = bce(D(G(z)), torch.ones(bs, 1, device=DEVICE))
            optG.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()

    return G


def generate(G, n_gen):
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        return G(z).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="GAN+ROS benchmark")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=SCRNASEQ_DATASETS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real, labels_raw = load_real_with_labels(args.dataset)
    real_bal, _ = oversample(real, labels_raw)
    n_sample, n_features = real.shape
    print(f"GAN+ROS — {args.dataset.upper()}  |  real={real.shape}  balanced={real_bal.shape}")

    G = train(real_bal, n_features, seed=SEED)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "gan_ros")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(G, n_gen)
        fname = f"{args.dataset.lower()}_gan_ros_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
