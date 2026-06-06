#!/usr/bin/env python
"""
GAN training stability ablation study.
---------------------------------------
Varies leakage fraction (lambda) — the proportion of GAT-selected real data
mixed into the generator's noise input Z_batch — across 4 scRNA-seq datasets.

Datasets:   Muraro, CBMC, Yan, Pollen
Leakage:    lambda in {0.0, 0.1, 0.2, 0.3}
Seed:       single seed = 42
Logging:    every 1000 iterations over 40001 total iterations

Output: results/rev6_losses.csv

Usage:
    conda activate ritwik_base
    python ablation_study/leakage_ablation.py
"""

import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REAL_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────
MASTER_SEED = 42
LEAKAGE_LEVELS = [0.0, 0.1, 0.2, 0.3]
DATASETS = ["muraro", "cbmc", "yan", "pollen"]
TOTAL_ITERS = 20001
LOG_EVERY = 1000
ND_STEPS = 5
NG_STEPS = 2
GAT_EPOCHS = 7501
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.1


# ═══════════════════════════════════════════════════════════════
# 1. HELPER: Noise sampler
# ═══════════════════════════════════════════════════════════════

def sample_Z(rows, cols):
    """Uniform noise  U(-1, 1)  of shape (rows, cols)."""
    return np.random.uniform(-1.0, 1.0, size=[rows, cols]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 2. DATA LOADING  (one function per dataset keeps each loader
#    self-contained and easy to audit)
# ═══════════════════════════════════════════════════════════════

def load_muraro():
    df_expr = pd.read_csv(os.path.join(REAL_DIR, "muraro_expression_matrix.csv"))
    df_label = pd.read_csv(os.path.join(REAL_DIR, "muraro_cell_types.csv"))
    df_expr.reset_index(drop=True, inplace=True)
    df_label.reset_index(drop=True, inplace=True)
    df_label = df_label.rename(columns={df_label.columns[0]: "cell_type"})
    df_combined = pd.concat([df_expr, df_label], axis=1)
    y_str = df_combined["cell_type"].values
    X = df_combined.iloc[:, :-1].values.astype(np.float32)
    return X, y_str


def load_cbmc():
    X = (
        pd.read_csv(
            os.path.join(REAL_DIR, "cbmc_rna_scaled.csv"), index_col=0, header=0
        )
        .T.values.astype(np.float32)
    )
    y_str = pd.read_csv(os.path.join(REAL_DIR, "cell_type_cbmc.csv"), header=0)[
        "x"
    ].values.ravel()
    return X, y_str


def load_yan():
    X = (
        pd.read_csv(os.path.join(REAL_DIR, "yan_process.csv"), header=None)
        .T.values.astype(np.float32)
    )
    y_str = (
        pd.read_csv(os.path.join(REAL_DIR, "yan_celltype.csv"), header=None)
        .iloc[:, 0]
        .values.ravel()
    )
    return X, y_str


def load_pollen():
    X = pd.read_csv(
        os.path.join(REAL_DIR, "pollen_process.txt"), header=None
    ).values.astype(np.float32)
    y_str = (
        pd.read_csv(os.path.join(REAL_DIR, "pollenc.txt"), header=None)
        .iloc[:, 0]
        .values.ravel()
    )
    return X, y_str


LOADERS = {
    "muraro": load_muraro,
    "cbmc": load_cbmc,
    "yan": load_yan,
    "pollen": load_pollen,
}


# ═══════════════════════════════════════════════════════════════
# 3. GAT SUBSAMPLING  (identify top-k informative cells)
# ═══════════════════════════════════════════════════════════════

def _build_knn_graph(X_np, k_neighbors=5):
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree")
    nn_model.fit(X_np)
    _, indices = nn_model.kneighbors(X_np)
    adj = torch.zeros((len(X_np), len(X_np)), dtype=torch.float32)
    for i in range(len(X_np)):
        adj[i, indices[i]] = 1.0
    row, col = adj.nonzero().t()
    return torch.stack([row, col], dim=0)


def _compute_attention_weights(model, data, edge_index, X_np):
    """After training, extract node-level attention from the 2nd GAT layer."""
    model.eval()
    with torch.no_grad():
        _, _, att2 = model(data)
        att_coeff = att2[0]              # shape [heads=1, num_edges]
        edge_src = edge_index[0]          # source node of each edge
        n_nodes = len(X_np)
        att_weights = torch.zeros(n_nodes, device=DEVICE)
        for i in range(n_nodes):
            mask = edge_src == i
            if mask.any():
                inc = att_coeff[:, mask].float()   # [1, num_incident]
                att_weights[i] = inc.mean()
    return att_weights


def gat_subsample(X_np, y_str, gat_seed=42):
    """
    Run GAT to rank cells by attention importance.
    Returns Xnew (top-k rows as float32 np array) and k.
    """
    torch.manual_seed(gat_seed)
    np.random.seed(gat_seed)

    n_sample = X_np.shape[0]
    k = math.ceil(n_sample * 0.2)

    # --- Label encoding & rare-type identification ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    class_counts = pd.Series(y_str).value_counts()
    rare_types = class_counts[class_counts <= 200].index.tolist()
    rare_mask = pd.Series(y_str).isin(rare_types)
    index_list = np.where(rare_mask.values)[0].tolist()

    # --- KNN graph ---
    edge_index = _build_knn_graph(X_np, k_neighbors=5)

    priority_nodes = torch.tensor(index_list, dtype=torch.long)

    data = Data(
        x=torch.tensor(X_np, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_enc, dtype=torch.long),
        priority_nodes=priority_nodes,
    ).to(DEVICE)

    # --- GAT classifier ---
    class GATClassifier(nn.Module):
        def __init__(self, num_features, num_classes, priority_weight):
            super().__init__()
            self.conv1 = GATConv(num_features, 32, heads=8)
            self.conv2 = GATConv(32 * 8, num_classes, heads=1)
            self.priority_weight = priority_weight

        def forward(self, d):
            x, ei, pn = d.x, d.edge_index, d.priority_nodes
            x, a1 = self.conv1(x, ei, return_attention_weights=True)
            att = torch.ones(x.size(0), device=x.device)
            att[pn] += self.priority_weight
            x = x * att.view(-1, 1)
            x = torch.relu(x)
            x, a2 = self.conv2(x, ei, return_attention_weights=True)
            return x, a1, a2

    num_features = X_np.shape[1]
    num_classes = len(np.unique(y_enc))
    model = GATClassifier(num_features, num_classes, priority_weight=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for _ in range(GAT_EPOCHS):
        model.train()
        optimizer.zero_grad()
        output, _, _ = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

    # --- Attention ranking ---
    att_weights = _compute_attention_weights(model, data, edge_index, X_np)
    sorted_idx = torch.argsort(att_weights, descending=True)
    top_k_nodes = sorted_idx[:k].cpu().numpy()
    Xnew = X_np[top_k_nodes].astype(np.float32)
    return Xnew, k


# ═══════════════════════════════════════════════════════════════
# 4. GAN MODELS
# ═══════════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hsize=(1024, 1024)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc_out = nn.Linear(hsize[1], output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, z):
        h = self.leaky_relu(self.fc1(z))
        h = self.leaky_relu(self.fc2(h))
        return self.fc_out(h)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hsize=(512, 256)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc3 = nn.Linear(hsize[1], input_dim)
        self.fc_out = nn.Linear(input_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_in):
        h = self.leaky_relu(self.fc1(x_in))
        h = self.leaky_relu(self.fc2(h))
        h = self.fc3(h)
        logits = self.fc_out(h)
        return logits, h


# ═══════════════════════════════════════════════════════════════
# 5. GAN TRAINING with variable leakage
# ═══════════════════════════════════════════════════════════════

def train_gan_with_leakage(X_np, Xnew, leakage, seed, device):
    """
    Train GAN for TOTAL_ITERS iterations, logging losses every LOG_EVERY steps.

    leakage : float
        Fraction of Z_batch rows that are real GAT-selected data.
        0.0 = all noise,  0.2 = 20 % real (approx. current behaviour).

    Returns list of dicts: [{iteration, d_loss, g_loss}, ...]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_sample, n_features = X_np.shape
    n_available_real = len(Xnew)

    # ---- leakage-aware Z_batch sizing ----
    n_real_desired = int(leakage * n_sample)
    n_real = min(n_real_desired, n_available_real)   # clamp to what we actually have
    n_noise = n_sample - n_real
    if n_real_desired > n_available_real:
        print(f"    (clamped: desired {n_real_desired} real rows, only {n_available_real} available)" )
    print(f"    Z_batch: {n_noise} noise + {n_real} real  = {n_sample} rows")

    # ---- models ----
    gen = Generator(n_features, n_features, hsize=(1024, 1024)).to(device)
    disc = Discriminator(n_features, hsize=(512, 256)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    gen_opt = optim.RMSprop(gen.parameters(), lr=0.0002)
    disc_opt = optim.RMSprop(disc.parameters(), lr=0.0004)

    # fixed tensors  (reused across iterations to avoid re-allocation overhead)
    X_batch = torch.tensor(X_np, dtype=torch.float32).to(device)
    real_fixed = torch.tensor(Xnew[:n_real], dtype=torch.float32).to(device) if n_real > 0 else None

    log_records = []

    for it in range(TOTAL_ITERS):

        # --- build Z_batch  (noise resampled every iter, real part fixed) ---
        if n_noise > 0:
            noise_np = sample_Z(n_noise, n_features)
            noise_t = torch.tensor(noise_np, dtype=torch.float32).to(device)
            Z_batch = torch.cat([noise_t, real_fixed], dim=0) if n_real > 0 else noise_t
        else:
            Z_batch = real_fixed  # corner case: 100 % real (unlikely given k ≈ 0.2n)

        # --- train Discriminator ---
        dloss_val = 0.0
        for _ in range(ND_STEPS):
            disc_opt.zero_grad()
            real_out, _ = disc(X_batch)
            with torch.no_grad():
                fake_samples = gen(Z_batch)
            fake_out, _ = disc(fake_samples)

            real_labels = torch.full_like(real_out, LABEL_SMOOTH_REAL).to(device)
            fake_labels = torch.full_like(fake_out, LABEL_SMOOTH_FAKE).to(device)

            d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            disc_opt.step()
            dloss_val = d_loss.item()

        # --- train Generator ---
        gloss_val = 0.0
        for _ in range(NG_STEPS):
            gen_opt.zero_grad()
            gen_samples = gen(Z_batch)
            gen_out, _ = disc(gen_samples)
            g_loss = criterion(gen_out, torch.ones_like(gen_out).to(device))
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            gen_opt.step()
            gloss_val = g_loss.item()

        # --- log ---
        if it % LOG_EVERY == 0:
            log_records.append(
                {"iteration": it, "d_loss": round(dloss_val, 6), "g_loss": round(gloss_val, 6)}
            )
            print(
                f"    [lambda={leakage}] iter {it:5d}  |  "
                f"D: {dloss_val:.4f}  |  G: {gloss_val:.4f}"
            )

    return log_records


# ═══════════════════════════════════════════════════════════════
# 6. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

def main():
    all_rows = []

    for ds_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'=' * 60}")

        # ---------- load ----------
        loader = LOADERS[ds_name]
        X, y_str = loader()
        n_sample, n_features = X.shape
        n_classes = len(np.unique(y_str))
        print(f"  Cells: {n_sample}  |  Genes: {n_features}  |  Types: {n_classes}")

        # ---------- GAT subsample (once per dataset, shared across leakage) ----------
        print("  Running GAT subsampling ...")
        t0 = time.time()
        Xnew, k = gat_subsample(X, y_str, gat_seed=MASTER_SEED)
        print(f"  GAT done in {time.time() - t0:.1f} s  |  k = {k}  |  Xnew = {Xnew.shape}")

        # ---------- sweep leakage ----------
        for leakage in LEAKAGE_LEVELS:
            print(f"\n  --- Leakage lambda = {leakage} ---")
            t0 = time.time()
            records = train_gan_with_leakage(
                X_np=X,
                Xnew=Xnew,
                leakage=leakage,
                seed=MASTER_SEED,
                device=DEVICE,
            )
            elapsed = time.time() - t0
            print(f"  Finished in {elapsed:.0f} s  ({elapsed / 60:.1f} min)")

            for rec in records:
                all_rows.append(
                    {
                        "dataset": ds_name,
                        "leakage": leakage,
                        "seed": MASTER_SEED,
                        "iteration": rec["iteration"],
                        "g_loss": rec["g_loss"],
                        "d_loss": rec["d_loss"],
                    }
                )

    # ---------- save ----------
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "rev6_losses.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"  Saved {len(df)} records to {out_path}")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    main()
