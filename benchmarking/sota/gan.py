#!/usr/bin/env python
"""
Vanilla GAN for scRNA-seq data generation (PyTorch).
=====================================================
Baseline generative model: a standard adversarial network with Bernoulli
cross-entropy loss (BCEWithLogitsLoss).  Generator: 32-tanh-32-linear.
Discriminator: 32-leakyReLU-32-linear.  Adam optimiser, 2,001 epochs.

Architecture matches the TF1.11 reference implementation in
benchmarking/sota/gan_tf1.py  (the original benchmark file).

Usage
-----
    python -m benchmarking.sota.gan --dataset muraro

Outputs
-------
    data/<dataset>_gan_generated_mixdata_iter{iter}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from config import DATASET_CONFIG, DATA_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 2001
LR = 0.0001


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


def load_real(dataset_name):
    """Load expression matrix matching the dataset config in config.py."""
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


def train(real, n_features, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = real.shape[0]

    G = Generator(LATENT_DIM, n_features).to(DEVICE)
    D = Discriminator(n_features).to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=LR)
    optD = optim.Adam(D.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            batch = torch.tensor(real[bi], device=DEVICE)
            bs = batch.shape[0]
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)

            # train discriminator
            fake = G(z).detach()
            d_loss = bce(D(batch), torch.ones(bs, 1, device=DEVICE)) \
                   + bce(D(fake), torch.zeros(bs, 1, device=DEVICE))
            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # train generator
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            g_loss = bce(D(G(z)), torch.ones(bs, 1, device=DEVICE))
            optG.zero_grad()
            g_loss.backward()
            optG.step()

        if epoch % 500 == 0:
            print(f"  GAN epoch {epoch:5d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

    return G


def generate(G, n_gen):
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        return G(z).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Vanilla GAN benchmark")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG),
                        help="Dataset name (default: muraro)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: data/gen_data/gan)")
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"GAN — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    G = train(real, n_features, seed=args.seed)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "gan")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(G, n_gen)
        fname = f"{args.dataset.lower()}_gan_generated_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
