#!/usr/bin/env python
"""
LSH-GAN (Locality-Sensitive Hashing GAN) for scRNA-seq data generation.
=========================================================================
Two-stage baseline: (1) KNN-based subsampling to select prototypical cells,
(2) a vanilla GAN trained on the reduced set.  The generator receives a
hybrid input — noise concatenated with the LSH-selected real cells.
Generator: 16-leakyReLU-16-linear.  Discriminator: 16-leakyReLU-16-linear → col → 1.
RMSprop, label smoothing, 2,001 epochs.

Usage:  python -m benchmarking.sota.lsh_gan --dataset muraro
Output: data/gen_data/lsh_gan/<dataset>_lsh_gan_generated_mixdata_iter{iter}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

try:
    from config import DATASET_CONFIG, DATA_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2001
ND_STEPS = 10
NG_STEPS = 10
BATCH_SIZE = 64
KNN_K = 5


def knn_subsample(X, k=5):
    """Select a subset of cells whose KNN neighbourhoods are disjoint."""
    n, d = X.shape
    nn_model = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X)
    indices = nn_model.kneighbors(X, return_distance=False)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if keep[i]:
            keep[indices[i][1:k]] = False
    return X[keep]


def sample_Z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n]).astype(np.float32)


class Generator(nn.Module):
    def __init__(self, io_dim, hsize=(16, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(io_dim, hsize[0]), nn.LeakyReLU(0.2),
            nn.Linear(hsize[0], hsize[1]), nn.LeakyReLU(0.2),
            nn.Linear(hsize[1], io_dim))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, io_dim, hsize=(16, 16)):
        super().__init__()
        self.fc1 = nn.Linear(io_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc3 = nn.Linear(hsize[1], io_dim)
        self.fc_out = nn.Linear(io_dim, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return self.fc_out(h)


def load_real(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'.  "
                         f"Choose from {list(DATASET_CONFIG)}.")
    cfg = DATASET_CONFIG[dataset_name]
    fpath = os.path.join(DATA_DIR, "expression_matrix", cfg["expression_file"])
    rk = {"header": cfg.get("header", 0)}
    if "index_col" in cfg:
        rk["index_col"] = cfg["index_col"]
    df = pd.read_csv(fpath, **rk)
    if cfg.get("transpose", False):
        df = df.T
    return df.values.astype(np.float32)


def train(x_plot, n_features, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xnew = knn_subsample(x_plot, k=KNN_K)
    print(f"  LSH subsample: {x_plot.shape[0]} → {Xnew.shape[0]}")

    G = Generator(n_features, n_features).to(DEVICE)
    D = Discriminator(n_features).to(DEVICE)
    optG = optim.RMSprop(G.parameters(), lr=0.001)
    optD = optim.RMSprop(D.parameters(), lr=0.001)
    bce = nn.BCEWithLogitsLoss()

    row = x_plot.shape[0]
    num_batches = (row + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        da1 = sample_Z(row - Xnew.shape[0], n_features)
        Z_batch = np.row_stack((da1, Xnew))
        idx_X = np.random.permutation(row)
        idx_Z = np.random.permutation(row)

        for _ in range(ND_STEPS):
            kk = np.random.randint(0, num_batches)
            s = kk * BATCH_SIZE
            e = min((kk + 1) * BATCH_SIZE, row)
            X_mb = torch.tensor(x_plot[idx_X[s:e]], dtype=torch.float32, device=DEVICE)
            Z_mb = torch.tensor(Z_batch[idx_Z[s:e]], dtype=torch.float32, device=DEVICE)

            fake = G(Z_mb)
            d_loss = bce(D(X_mb), torch.ones(X_mb.shape[0], 1, device=DEVICE)) \
                   + bce(D(fake), torch.zeros(fake.shape[0], 1, device=DEVICE))
            optD.zero_grad()
            d_loss.backward()
            optD.step()

        for _ in range(NG_STEPS):
            kk = np.random.randint(0, num_batches)
            s = kk * BATCH_SIZE
            e = min((kk + 1) * BATCH_SIZE, row)
            Z_mb = torch.tensor(Z_batch[idx_Z[s:e]], dtype=torch.float32, device=DEVICE)
            g_loss = bce(D(G(Z_mb)), torch.ones(Z_mb.shape[0], 1, device=DEVICE))
            optG.zero_grad()
            g_loss.backward()
            optG.step()

        if epoch % 500 == 0:
            print(f"  LSHGAN ep {epoch:5d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

    return G


def generate(G, n_gen, n_features):
    z = torch.tensor(sample_Z(n_gen, n_features), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return G(z).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="LSH-GAN benchmark")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"LSH-GAN — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    G = train(real, n_features, seed=args.seed)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "lsh_gan")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(G, n_gen, n_features)
        fname = f"{args.dataset.lower()}_lsh_gan_generated_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
