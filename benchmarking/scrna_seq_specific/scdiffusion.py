#!/usr/bin/env python
"""
scDiffusion — DDPM-style diffusion model for scRNA-seq data generation.
=========================================================================
100 diffusion steps, linear β schedule, MLP denoiser with time embedding.
Denoiser: input+64 → 512-ReLU → 256-ReLU → linear.
Adam, 200 epochs.

Usage:  python -m benchmarking.scrna_seq_specific.scdiffusion --dataset muraro
Output: data/gen_data/scdiffusion/<dataset>_scdiffusion_mixdata_iter{iter}.csv
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
T = 100
BATCH_SIZE = 64
EPOCHS = 200
SEED = 42

betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, 0)


class Denoiser(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.t_embed = nn.Embedding(T, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 64, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, in_dim))

    def forward(self, x, t):
        te = self.t_embed(t)
        return self.net(torch.cat([x, te], dim=1))


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alpha_cumprod[t].view(-1, 1)
    return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise, noise


def p_sample(model, xt, t):
    a_bar_t = alpha_cumprod[t].view(-1, 1)
    eps = model(xt, t)
    a_t = alphas[t].view(-1, 1)
    coeff1 = 1 / a_t.sqrt()
    coeff2 = (1 - a_t) / (1 - a_bar_t).sqrt()
    mu = coeff1 * (xt - coeff2 * eps)
    if t.min() > 0:
        return mu + betas[t].sqrt().view(-1, 1) * torch.randn_like(xt)
    return mu


@torch.no_grad()
def generate(model, n_gen, n_features):
    x = torch.randn(n_gen, n_features, device=DEVICE)
    for t_step in range(T - 1, -1, -1):
        tt = torch.full((n_gen,), t_step, dtype=torch.long, device=DEVICE)
        x = p_sample(model, x, tt)
    return x.cpu().numpy()


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

    model = Denoiser(n_features).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")

    for ep in range(EPOCHS):
        bi = np.random.choice(n, min(BATCH_SIZE, n), replace=False)
        x0 = torch.tensor(real[bi], device=DEVICE)
        bs = x0.shape[0]
        t = torch.randint(0, T, (bs,), device=DEVICE)
        xt, noise = q_sample(x0, t)
        with autocast("cuda"):
            loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    return model


def main():
    parser = argparse.ArgumentParser(description="scDiffusion benchmark (DDPM)")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=SCRNASEQ_DATASETS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    real = load_real(args.dataset)
    n_sample, n_features = real.shape
    print(f"scDiffusion — {args.dataset.upper()}  |  cells={n_sample}  genes={n_features}")

    model = train(real, n_features, seed=SEED)

    out_dir = args.out or os.path.join(DATA_DIR, "gen_data", "scdiffusion")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    for i, n_gen in enumerate(batch_sizes):
        syn = generate(model, n_gen, n_features)
        fname = f"{args.dataset.lower()}_scdiffusion_mixdata_iter{i}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(syn).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={syn.shape}")


if __name__ == "__main__":
    main()
