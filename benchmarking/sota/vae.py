#!/usr/bin/env python
"""
VAE (Variational Autoencoder) for scRNA-seq data generation (PyTorch).
=======================================================================
Baseline: encoder 32-tanh → mu, logvar; decoder 32-tanh → linear.
Adam optimiser, MSE recon + KL divergence, 2,001 epochs.

Usage:  python -m benchmarking.sota.vae --dataset muraro
Output: data/gen_data/vae/<dataset>_vae_generated_mixdata_iter{iter}.csv
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


class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, 32), nn.Tanh())
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = self.hidden(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.Tanh(),
            nn.Linear(32, out_dim))

    def forward(self, z):
        return self.net(z)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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


def train(real, n_features, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = real.shape[0]

    enc = Encoder(n_features, LATENT_DIM).to(DEVICE)
    dec = Decoder(LATENT_DIM, n_features).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            x = torch.tensor(real[bi], device=DEVICE)

            mu, logvar = enc(x)
            z = reparameterize(mu, logvar)
            x_recon = dec(z)

            recon_loss = ((x - x_recon) ** 2).mean()
            kl_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean()
            loss = recon_loss + kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 500 == 0:
            print(f"  VAE epoch {epoch:5d}  loss={loss.item():.4f}  "
                  f"recon={recon_loss.item():.4f}  KL={kl_loss.item():.4f}")

    return dec


def generate(dec, n_gen):
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        return dec(z).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="VAE benchmark")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"VAE — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    dec = train(real, n_features, seed=args.seed)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "vae")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(dec, n_gen)
        fname = f"{args.dataset.lower()}_vae_generated_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
