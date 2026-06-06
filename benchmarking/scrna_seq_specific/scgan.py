#!/usr/bin/env python
"""
scGAN — WGAN-GP for scRNA-seq data generation (PyTorch).
==========================================================
Unconditional Wasserstein GAN with gradient penalty.  Deeper architecture:
Generator: 256-bn-ReLU → 512-bn-ReLU → 1024-bn-ReLU → linear.
Critic: 512-leakyReLU → 256-leakyReLU → 1.
Adam(β₁=0.5, β₂=0.9), 200 epochs, GP weight λ=10.

Usage:  python -m benchmarking.scrna_seq_specific.scgan --dataset muraro
Output: data/gen_data/scgan/<dataset>_scgan_mixdata_iter{iter}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

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
D_STEPS = 5
GP_LAMBDA = 10.0
SEED = 42


class Generator(nn.Module):
    def __init__(self, zdim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


def gradient_penalty(D, real, fake):
    bs = real.shape[0]
    alpha = torch.rand(bs, 1, device=DEVICE)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    d_interp = D(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def load_real(dataset_name):
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
    optG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scaler = GradScaler("cuda")

    for _ in range(EPOCHS):
        for _ in range(D_STEPS):
            bi = np.random.choice(n, min(BATCH_SIZE, n), replace=False)
            r = torch.tensor(real[bi], device=DEVICE)
            bs = r.shape[0]
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            with autocast("cuda"):
                fake = G(z).detach()
                d_loss = D(fake).mean() - D(r).mean() + GP_LAMBDA * gradient_penalty(D, r, fake)
            optD.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optD)
            scaler.update()

        z = torch.randn(min(BATCH_SIZE, n), LATENT_DIM, device=DEVICE)
        with autocast("cuda"):
            g_loss = -D(G(z)).mean()
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
    parser = argparse.ArgumentParser(description="scGAN benchmark (WGAN-GP)")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=SCRNASEQ_DATASETS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"scGAN — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    G = train(real, n_features, seed=SEED)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "scgan")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(G, n_gen)
        fname = f"{args.dataset.lower()}_scgan_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
