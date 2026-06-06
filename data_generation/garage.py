#!/usr/bin/env python
"""
GARAGE — Graph-Attentive Rare-cell-Aware single-cell data GEneration.
=========================================================================

A two-stage framework for generating high-fidelity synthetic scRNA-seq data:

  Stage 1 (GAT Subsampling):
    A Graph Attention Network (GAT) classifier is trained on a KNN cell-cell
    graph.  Rare cell types receive a priority weight boost.  After training,
    per-cell attention scores from the second GAT layer are extracted and the
    top-k cells (k = leakage_fraction * n_cells) are selected as "seeds".

  Stage 2 (GAN Generation with Attention-Guided Seeding):
    A Generator/Discriminator GAN is trained.  Instead of pure noise, the
    generator receives a *hybrid* input batch: a mix of random noise vectors
    and the GAT-selected seed cells.  This seeding anchors the generator to
    biologically realistic states, stabilises training, and ensures rare cell
    types are represented in the output.

Datasets supported: Yan (124 cells, 6 types), Pollen (301 cells, 11 types),
                     CBMC (7,895 cells, 13 types), Muraro (2,126 cells, 10 types).

Usage
-----
    python -m data_generation.garage --dataset muraro

Citation
--------
    Ganguly, R., et al.  "GARAGE: A Graph-Attentive GAN for Rare-Cell-Aware
    Single-Cell RNA-seq Data Generation."  bioRxiv, 2025.
    DOI: 10.1101/2025.09.28.679012
"""

import argparse
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

try:
    from config import DATASET_CONFIG, DATA_DIR, RESULTS_DIR, GARAGE_DEFAULTS
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_CONFIG, DATA_DIR, RESULTS_DIR, GARAGE_DEFAULTS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = GARAGE_DEFAULTS["random_seed"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name):
    """
    Load expression matrix and cell-type labels for *dataset_name*.

    Returns
    -------
    X : np.ndarray  (n_cells, n_genes) float32
    y_str : np.ndarray  (n_cells,)  string labels
    n_sample : int
    n_features : int
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'.  "
                         f"Choose from {list(DATASET_CONFIG)}.")
    cfg = DATASET_CONFIG[dataset_name]

    # --- expression matrix ---
    expr_path = os.path.join(DATA_DIR, "expression_matrix", cfg["expression_file"])
    rk = {"header": cfg.get("header", 0)}
    if "index_col" in cfg:
        rk["index_col"] = cfg["index_col"]
    df_expr = pd.read_csv(expr_path, **rk)
    if cfg.get("transpose", False):
        df_expr = df_expr.T
    df_expr.reset_index(drop=True, inplace=True)

    # --- cell-type labels ---
    lbl_path = os.path.join(DATA_DIR, "cell_types", cfg["label_file"])
    lk = {"header": cfg.get("label_header", None)}
    df_label = pd.read_csv(lbl_path, **lk)
    df_label.reset_index(drop=True, inplace=True)

    if cfg["label_header"] is not None:
        label_col = df_label.columns[0] if cfg["label_col"] not in df_label.columns else cfg["label_col"]
        y_str = df_label[label_col].values.ravel()
    else:
        y_str = df_label.iloc[:, cfg["label_col"]].values.ravel()

    X = df_expr.values.astype(np.float32)
    n_sample, n_features = X.shape
    print(f"  Loaded {dataset_name}: {n_sample} cells x {n_features} genes  "
          f"({len(np.unique(y_str))} cell types)")
    return X, y_str, n_sample, n_features


# ═══════════════════════════════════════════════════════════════════════════
# 2. GAT SUBSAMPLING (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════

def _build_knn_graph(X, k_neighbors=5):
    """Construct an adjacency matrix from a KNN graph of *X*."""
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree")
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)
    adj = torch.zeros((len(X), len(X)), dtype=torch.float32)
    for i in range(len(X)):
        adj[i, indices[i]] = 1.0
    row, col = adj.nonzero().t()
    return torch.stack([row, col], dim=0)


class GATClassifier(nn.Module):
    """
    2-layer GAT classifier for node-level cell-type prediction.
    The *priority_weight* boosts attention toward rare-type cells
    after the first GAT layer, encouraging the model to attend to them.
    """

    def __init__(self, num_features, num_classes, priority_weight):
        super().__init__()
        self.conv1 = GATConv(num_features, 32, heads=8)
        self.conv2 = GATConv(32 * 8, num_classes, heads=1)
        self.priority_weight = priority_weight

    def forward(self, data):
        x, edge_index, priority_nodes = data.x, data.edge_index, data.priority_nodes

        x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)

        attention = torch.ones(x.size(0), device=x.device)
        attention[priority_nodes] += self.priority_weight
        x = x * attention.view(-1, 1)
        x = torch.relu(x)

        x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, attention1, attention2


def gat_subsample(X, y_str, dataset_name, seed=SEED):
    """
    Train the GAT classifier and return the indices of the top-k cells
    ranked by attention weight from the second GAT layer.

    Parameters
    ----------
    X : np.ndarray  (n_cells, n_genes)
    y_str : np.ndarray  (n_cells,) raw string labels
    dataset_name : str
    seed : int

    Returns
    -------
    top_k_indices : np.ndarray  (k,)
    k : int
    Xnew : np.ndarray  (k, n_genes)  the actual feature rows
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_sample = X.shape[0]
    k = math.ceil(n_sample * GARAGE_DEFAULTS["leakage_fraction"])

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    # Identify rare cell types (count <= threshold)
    rare_thresh = DATASET_CONFIG[dataset_name]["rare_threshold"]
    class_counts = pd.Series(y_str).value_counts()
    rare_types = class_counts[class_counts <= rare_thresh].index.tolist()
    rare_mask = pd.Series(y_str).isin(rare_types).values
    index_list = np.where(rare_mask)[0].tolist()
    print(f"  Rare types: {rare_types}  ({len(index_list)} cells,  "
          f"threshold ≤ {rare_thresh})")
    print(f"  GAT top-k:  k = {k}  ({k / n_sample:.0%} of {n_sample} cells)")

    # Build KNN graph
    edge_index = _build_knn_graph(X, k_neighbors=5)
    priority_nodes = torch.tensor(index_list, dtype=torch.long)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_enc, dtype=torch.long),
        priority_nodes=priority_nodes,
    ).to(DEVICE)

    # Initialise GAT
    num_features = X.shape[1]
    num_classes = len(np.unique(y_enc))
    model = GATClassifier(
        num_features, num_classes,
        priority_weight=GARAGE_DEFAULTS["priority_weight"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train
    t0 = time.time()
    for epoch in range(GARAGE_DEFAULTS["gat_epochs"]):
        model.train()
        optimizer.zero_grad()
        output, a1, a2 = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 1500 == 0:
            print(f"    GAT epoch {epoch:5d}  loss={loss.item():.4f}")
    print(f"  GAT trained in {time.time() - t0:.1f} s")

    # Extract per-cell attention from 2nd GAT layer
    model.eval()
    with torch.no_grad():
        _, _, att2 = model(data)
        att_coeff = att2[0]              # [heads=1, num_edges]
        edge_src = edge_index[0]          # source node of each edge

        att_weights = torch.zeros(n_sample, device=DEVICE)
        for i in range(n_sample):
            mask = edge_src == i
            if mask.any():
                inc = att_coeff[:, mask].float()
                att_weights[i] = inc.mean()

    sorted_idx = torch.argsort(att_weights, descending=True)
    top_k_nodes = sorted_idx[:k].cpu().numpy()
    Xnew = X[top_k_nodes].astype(np.float32)

    print(f"  GAT top-k done:  {len(top_k_nodes)} seeds  "
          f"(attention range [{att_weights.min():.4f}, {att_weights.max():.4f}])")
    return top_k_nodes, k, Xnew


# ═══════════════════════════════════════════════════════════════════════════
# 3. GAN MODELS (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

def sample_Z(m, n):
    """Uniform noise  U(-1, 1)  of shape (m, n)."""
    return np.random.uniform(-1.0, 1.0, size=[m, n]).astype(np.float32)


class Generator(nn.Module):
    """Generator: 1024 → 1024 → n_genes, LeakyReLU(0.2), no output activation."""

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
    """Discriminator: 512 → 256 → n_genes → 1, LeakyReLU(0.2).
    Returns a (logits, intermediate_representation) tuple."""

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


# ═══════════════════════════════════════════════════════════════════════════
# 4. GAN TRAINING (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

def train_gan(x_plot, Xnew, n_features, seed=SEED):
    """
    Train the GAN with the GAT-seeded hybrid input.

    The generator's input batch Z_batch is a vertical stack of random
    noise plus GAT-selected seed cells, giving the generator a biological
    "anchor" for rare cell types.

    Parameters
    ----------
    x_plot : np.ndarray  (n_cells, n_features)
    Xnew : np.ndarray  (k, n_features)  GAT-selected seeds
    n_features : int
    seed : int

    Returns
    -------
    generator : Generator
    log_records : list of dict  [{iteration, d_loss, g_loss}, ...]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_sample = x_plot.shape[0]
    col = n_features

    gen = Generator(col, col, hsize=GARAGE_DEFAULTS["generator_hidden"]).to(DEVICE)
    disc = Discriminator(col, hsize=GARAGE_DEFAULTS["discriminator_hidden"]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    gen_opt = optim.RMSprop(gen.parameters(), lr=GARAGE_DEFAULTS["g_lr"])
    disc_opt = optim.RMSprop(disc.parameters(), lr=GARAGE_DEFAULTS["d_lr"])

    X_batch = torch.tensor(x_plot, dtype=torch.float32).to(DEVICE)
    log_records = []

    total_iters = GARAGE_DEFAULTS["gan_total_iters"]
    nd = GARAGE_DEFAULTS["nd_steps"]
    ng = GARAGE_DEFAULTS["ng_steps"]

    for i in range(total_iters):
        row1 = n_sample - Xnew.shape[0]
        if row1 < 0:
            row1 = 0

        da1 = sample_Z(row1, col)
        cur_xn = Xnew if Xnew.ndim == 2 else Xnew.reshape(1, -1)
        Z_batch_np = np.vstack((da1, cur_xn))
        Z_batch = torch.tensor(Z_batch_np, dtype=torch.float32).to(DEVICE)

        # --- train Discriminator ---
        dloss_epoch = 0.0
        for _ in range(nd):
            disc_opt.zero_grad()
            real_out, _ = disc(X_batch)
            fake_samples = gen(Z_batch).detach()
            fake_out, _ = disc(fake_samples)

            real_labels = torch.full_like(real_out,
                                          GARAGE_DEFAULTS["label_smooth_real"]).to(DEVICE)
            fake_labels = torch.full_like(fake_out,
                                          GARAGE_DEFAULTS["label_smooth_fake"]).to(DEVICE)
            disc_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            disc_opt.step()
            dloss_epoch = disc_loss.item()

        # --- train Generator ---
        gloss_epoch = 0.0
        for _ in range(ng):
            gen_opt.zero_grad()
            generated_samples = gen(Z_batch)
            gen_out, _ = disc(generated_samples)
            gen_loss = criterion(gen_out, torch.ones_like(gen_out).to(DEVICE))
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            gen_opt.step()
            gloss_epoch = gen_loss.item()

        if i % 1000 == 0:
            log_records.append({
                "iteration": i,
                "d_loss": round(dloss_epoch, 6),
                "g_loss": round(gloss_epoch, 6),
            })
            print(f"    Iter {i:5d}  |  D: {dloss_epoch:.4f}  |  G: {gloss_epoch:.4f}")

    return gen, log_records


# ═══════════════════════════════════════════════════════════════════════════
# 5. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(generator, n_sample, n_features, Xnew, out_dir, dataset_name, k):
    """
    Generate synthetic data at multiple volume multipliers (0.25x – 1.5x n_sample).
    Save each batch as a CSV file.

    Parameters
    ----------
    generator : Generator
    n_sample, n_features : int
    Xnew : np.ndarray  GAT seeds
    out_dir : str
    dataset_name : str
    k : int  (top-k value, used in filename for traceability)
    """
    os.makedirs(out_dir, exist_ok=True)
    batch_sizes_gen = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)

    for i, current_batch_size in enumerate(batch_sizes_gen):
        row1_gen = current_batch_size - Xnew.shape[0]
        if row1_gen < 0:
            row1_gen = 0

        da1_gen = sample_Z(row1_gen, n_features)
        Z_batch_gen_np = np.vstack((da1_gen, Xnew)) if row1_gen > 0 else Xnew
        Z_batch_gen = torch.tensor(Z_batch_gen_np, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            g_plot_np = generator(Z_batch_gen).cpu().numpy()

        fname = f"{dataset_name}_data_mixdata_iter{i}_top_{k}.csv"
        fpath = os.path.join(out_dir, fname)
        pd.DataFrame(g_plot_np).to_csv(fpath, index=False, header=False)
        print(f"  Saved {fpath}  shape={g_plot_np.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_garage(dataset_name, out_dir=None, seed=SEED):
    """
    Full GARAGE pipeline for a single dataset:
      (1) GAT subsampling  →  (2) GAN training  →  (3) data generation.

    Parameters
    ----------
    dataset_name : str  one of {"yan", "pollen", "cbmc", "muraro"}
    out_dir : str or None  output directory; defaults to data/gen_data/
    seed : int
    """
    print(f"\n{'=' * 60}")
    print(f"  GARAGE  —  {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # 1. Load data
    X, y_str, n_sample, n_features = load_dataset(dataset_name)

    # 2. GAT subsampling
    print(f"\n[Stage 1] GAT subsampling ...")
    top_k_indices, k, Xnew = gat_subsample(X, y_str, dataset_name, seed=seed)

    # 3. GAN training
    print(f"\n[Stage 2] GAN training ({GARAGE_DEFAULTS['gan_total_iters']} iterations) ...")
    t0 = time.time()
    generator, log_records = train_gan(X, Xnew, n_features, seed=seed)
    print(f"  GAN trained in {time.time() - t0:.0f} s")

    # 4. Generate
    if out_dir is None:
        out_dir = os.path.join(DATA_DIR, "gen_data")
    print(f"\n[Generation] Writing synthetic data to {out_dir}/ ...")
    generate_data(generator, n_sample, n_features, Xnew, out_dir, dataset_name, k)

    print(f"\n  ✓ GARAGE pipeline complete.")
    return log_records


def main():
    parser = argparse.ArgumentParser(
        description="GARAGE: Graph-Attentive Rare-cell-Aware scRNA-seq data Generation")
    parser.add_argument("--dataset", type=str, default="muraro",
                        choices=list(DATASET_CONFIG),
                        help="Dataset name (default: muraro)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for generated CSVs")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_garage(dataset_name=args.dataset,
               out_dir=args.out,
               seed=args.seed)


if __name__ == "__main__":
    main()
