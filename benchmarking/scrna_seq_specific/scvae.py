#!/usr/bin/env python
"""
scVAE — beta-VAE with deeper architecture for scRNA-seq data generation.
==========================================================================
Encoder: 256-ReLU → 128-ReLU → 64-ReLU → μ, logσ².
Decoder: 64-ReLU → 128-ReLU → 256-ReLU → linear.
Adam, β=0.1, 200 epochs.

Usage:  python -m benchmarking.scrna_seq_specific.scvae --dataset muraro
Output: data/gen_data/scvae/<dataset>_scvae_mixdata_iter{iter}.csv
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
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 200
BETA = 0.1
SEED = 42


class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim))

    def forward(self, z):
        return self.net(z)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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

    enc = Encoder(n_features, LATENT_DIM).to(DEVICE)
    dec = Decoder(LATENT_DIM, n_features).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)
    scaler = GradScaler("cuda")

    for ep in range(EPOCHS):
        bi = np.random.choice(n, min(BATCH_SIZE, n), replace=False)
        x = torch.tensor(real[bi], device=DEVICE)
        with autocast("cuda"):
            mu, logvar = enc(x)
            z = reparameterize(mu, logvar)
            x_recon = dec(z)
            recon = ((x - x_recon) ** 2).sum(1).mean()
            kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        loss = recon + BETA * kl
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    return dec


def generate(dec, n_gen):
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        return dec(z).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="scVAE benchmark (beta-VAE)")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=SCRNASEQ_DATASETS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"scVAE — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    dec = train(real, n_features, seed=SEED)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "scvae")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(dec, n_gen)
        fname = f"{args.dataset.lower()}_scvae_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
