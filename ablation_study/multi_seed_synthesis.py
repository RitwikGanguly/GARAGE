#!/usr/bin/env python
"""Multi-seed synthetic data generation for 5 methods × 5 seeds × 4 datasets.

Methods: WGAN, f-GAN, LSH-GAN, Vanilla GAN, GAT-GAN.
Seeds: 42, 123, 456, 789, 1024 (same seed used by all 5 methods per run).

Iteration mapping:
  Yan/CBMC/Muraro → iter3 (1.0 × n_real)
  Pollen           → iter5 (1.5 × n_real)

Output:
  gen_data/seed_{s}/{method}/{dataset_lower}_{prefix}_generated_mixdata_iter{iter}.csv

Usage:  conda run -n ritwik_base python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os, math, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(REPO_ROOT, "data")
GEN_ROOT   = os.path.join(REPO_ROOT, "data", "gen_data")

SEEDS = [42, 123, 456, 789, 1024]

DATASETS = {
    "Yan": {
        "file": "yan_process.csv",
        "csv_kwargs": {"header": None},
        "transpose": True,
        "label_file": "yan_celltype.csv",
        "label_kwargs": {"header": None},
        "label_col": 0,
        "iter": 3,
    },
    "CBMC": {
        "file": "cbmc_rna_scaled.csv",
        "csv_kwargs": {"index_col": 0, "header": 0},
        "transpose": True,
        "label_file": "cell_type_cbmc.csv",
        "label_kwargs": {"header": 0},
        "label_col": "x",
        "iter": 3,
    },
    "Muraro": {
        "file": "muraro_expression_matrix.csv",
        "csv_kwargs": {"header": 0},
        "transpose": False,
        "label_file": "muraro_cell_types.csv",
        "label_kwargs": {"header": 0},
        "label_col": "cell_type",
        "iter": 3,
    },
    "Pollen": {
        "file": "pollen_process.txt",
        "csv_kwargs": {"header": None},
        "transpose": False,
        "label_file": "pollenc.txt",
        "label_kwargs": {"header": None},
        "label_col": 0,
        "iter": 5,
    },
}

def load_real(ds_cfg):
    df = pd.read_csv(os.path.join(REAL_DIR, ds_cfg["file"]), **ds_cfg["csv_kwargs"])
    if ds_cfg["transpose"]:
        df = df.T
    real = df.values.astype(np.float32)
    return real

def load_labels(ds_cfg):
    lbl_df = pd.read_csv(os.path.join(REAL_DIR, ds_cfg["label_file"]),
                         **ds_cfg["label_kwargs"])
    if ds_cfg["label_kwargs"].get("header") is not None:
        col = ds_cfg["label_col"]
        return lbl_df[col].values.ravel()
    else:
        return lbl_df.iloc[:, ds_cfg["label_col"]].values.ravel()


# ══════════════════════════════════════════════════════════════════════════
#  Method 1 — WGAN
# ══════════════════════════════════════════════════════════════════════════

LATENT_WGAN   = 100
BATCH_WGAN    = 64
EPOCHS_WGAN   = 2001
LR_WGAN       = 5e-4
CLIP_WGAN     = 0.01
N_CRITIC_WGAN = 5

class WGenerator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_WGAN, 32), nn.Tanh(),
                                 nn.Linear(32, n_features))

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 32), nn.Tanh(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)

def train_and_generate_wgan(real, n_features, device, seed, ds_name, wanted_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)

    G = WGenerator(n_features).to(device)
    C = Critic(n_features).to(device)
    opt_G = optim.RMSprop(G.parameters(), lr=LR_WGAN)
    opt_C = optim.RMSprop(C.parameters(), lr=LR_WGAN)

    n = real.shape[0]
    for epoch in range(EPOCHS_WGAN):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_WGAN):
            batch = real[idx[i:i + BATCH_WGAN]]
            bsz = len(batch)
            batch_t = torch.tensor(batch, device=device)
            noise = torch.randn(bsz, LATENT_WGAN, device=device)
            c_loss = -C(batch_t).mean() + C(G(noise).detach()).mean()
            opt_C.zero_grad()
            c_loss.backward()
            opt_C.step()
            for p in C.parameters():
                p.data.clamp_(-CLIP_WGAN, CLIP_WGAN)

        noise_g = torch.randn(BATCH_WGAN, LATENT_WGAN, device=device)
        g_loss = -C(G(noise_g)).mean()
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        if epoch % 500 == 0:
            print(f"    WGAN ep {epoch:5d}  C={c_loss.item():.4f}  G={g_loss.item():.4f}")

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * real.shape[0]).astype(int)
    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}", "wgan")
    os.makedirs(out_dir, exist_ok=True)

    for i, n_gen in enumerate(batch_sizes):
        if i != wanted_iter:
            continue
        z = torch.randn(n_gen, LATENT_WGAN, device=device)
        with torch.no_grad():
            synthetic = G(z).cpu().numpy()
        fname = f"{ds_name.lower()}_wgan_generated_mixdata_iter{i}.csv"
        pd.DataFrame(synthetic).to_csv(os.path.join(out_dir, fname),
                                       index=False, header=False)
        print(f"    Saved {fname}  shape={synthetic.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  Method 2 — f-GAN
# ══════════════════════════════════════════════════════════════════════════

LATENT_FGAN  = 100
BATCH_FGAN   = 64
EPOCHS_FGAN  = 2001
LR_FGAN      = 5e-4
LAMBDA_FGAN  = 10.0

class FGenerator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_FGAN, 128), nn.LeakyReLU(),
                                 nn.Linear(128, n_features))

    def forward(self, z):
        return self.net(z)

class FDiscriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.LeakyReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)

def fisher_ratio(T_real, T_fake):
    eps = 1e-8
    mr, mf = T_real.mean(), T_fake.mean()
    vr = ((T_real - mr) ** 2).mean()
    vf = ((T_fake - mf) ** 2).mean()
    return (mr - mf) / torch.sqrt(vr + vf + eps)

def train_and_generate_fgan(real, n_features, device, seed, ds_name, wanted_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)

    G = FGenerator(n_features).to(device)
    D = FDiscriminator(n_features).to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR_FGAN)
    opt_D = optim.Adam(D.parameters(), lr=LR_FGAN)

    n = real.shape[0]
    for epoch in range(EPOCHS_FGAN):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_FGAN):
            batch = real[idx[i:i + BATCH_FGAN]]
            bsz = len(batch)
            batch_t = torch.tensor(batch, device=device)
            z = torch.randn(bsz, LATENT_FGAN, device=device)

            fake = G(z)
            T_real, T_fake = D(batch_t), D(fake)
            fr = fisher_ratio(T_real, T_fake)
            constraint = (T_real ** 2 + T_fake ** 2).mean() - 1.0
            penalty = torch.clamp(constraint, min=0.0) ** 2
            D_loss = -fr + LAMBDA_FGAN * penalty
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            fake_g = G(z)
            fr_g = fisher_ratio(D(batch_t), D(fake_g))
            G_loss = -fr_g
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

        if epoch % 500 == 0:
            print(f"    fGAN  ep {epoch:5d}  D={D_loss.item():.4f}  G={G_loss.item():.4f}")

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * real.shape[0]).astype(int)
    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}", "fgan")
    os.makedirs(out_dir, exist_ok=True)

    for i, n_gen in enumerate(batch_sizes):
        if i != wanted_iter:
            continue
        z = torch.randn(n_gen, LATENT_FGAN, device=device)
        with torch.no_grad():
            synthetic = G(z).cpu().numpy()
        fname = f"{ds_name.lower()}_fgan_generated_mixdata_iter{i}.csv"
        pd.DataFrame(synthetic).to_csv(os.path.join(out_dir, fname),
                                       index=False, header=False)
        print(f"    Saved {fname}  shape={synthetic.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  Method 3 — LSH-GAN
# ══════════════════════════════════════════════════════════════════════════

EPOCHS_LSH   = 2001
ND_STEPS_LSH = 10
NG_STEPS_LSH = 10
BATCH_LSH    = 64
KNN_K_LSH    = 5

def knn_subsample(X, k=5):
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    arr1 = np.ones(n, dtype=np.float32)
    for i in range(n):
        if arr1[i] != 0:
            nb = indices[i][1:k]
            arr1[nb] = 0
    selected = np.nonzero(arr1)[0]
    return X[selected]

class LSHGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hsize=(16, 16)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hsize[0]), nn.LeakyReLU(0.2),
                                 nn.Linear(hsize[0], hsize[1]), nn.LeakyReLU(0.2),
                                 nn.Linear(hsize[1], output_dim))

    def forward(self, z):
        return self.net(z)

class LSHDiscriminator(nn.Module):
    def __init__(self, input_dim, hsize=(16, 16)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc3 = nn.Linear(hsize[1], input_dim)
        self.fc_out = nn.Linear(input_dim, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h1 = self.lrelu(self.fc1(x))
        h2 = self.lrelu(self.fc2(h1))
        h3 = self.fc3(h2)
        return self.fc_out(h3)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def train_and_generate_lshgan(real, n_features, device, seed, ds_name, wanted_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xnew = knn_subsample(real, k=KNN_K_LSH)

    G = LSHGenerator(n_features, n_features).to(device)
    D = LSHDiscriminator(n_features).to(device)
    opt_G = optim.RMSprop(G.parameters(), lr=0.001)
    opt_D = optim.RMSprop(D.parameters(), lr=0.001)
    bce = nn.BCEWithLogitsLoss()

    row = real.shape[0]
    num_batches = (row + BATCH_LSH - 1) // BATCH_LSH

    for epoch in range(EPOCHS_LSH):
        da1 = sample_Z(row - Xnew.shape[0], n_features)
        Z_batch = np.row_stack((da1, Xnew))
        idx_X = np.random.permutation(row)
        idx_Z = np.random.permutation(row)

        for _ in range(ND_STEPS_LSH):
            k = np.random.randint(0, num_batches)
            s, e = k * BATCH_LSH, min((k + 1) * BATCH_LSH, row)
            X_mb = torch.tensor(real[idx_X[s:e]], dtype=torch.float32, device=device)
            Z_mb = torch.tensor(Z_batch[idx_Z[s:e]], dtype=torch.float32, device=device)

            fake = G(Z_mb).detach()
            d_loss = bce(D(X_mb), torch.ones(X_mb.shape[0], 1, device=device)) \
                   + bce(D(fake), torch.zeros(fake.shape[0], 1, device=device))
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        for _ in range(NG_STEPS_LSH):
            k = np.random.randint(0, num_batches)
            s, e = k * BATCH_LSH, min((k + 1) * BATCH_LSH, row)
            Z_mb = torch.tensor(Z_batch[idx_Z[s:e]], dtype=torch.float32, device=device)
            g_loss = bce(D(G(Z_mb)), torch.ones(Z_mb.shape[0], 1, device=device))
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        if epoch % 500 == 0:
            print(f"    LSHGAN ep {epoch:5d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * real.shape[0]).astype(int)
    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}", "lsh_gan")
    os.makedirs(out_dir, exist_ok=True)

    for i, n_gen in enumerate(batch_sizes):
        if i != wanted_iter:
            continue
        z = torch.tensor(sample_Z(n_gen, n_features), dtype=torch.float32, device=device)
        with torch.no_grad():
            synthetic = G(z).cpu().numpy()
        fname = f"{ds_name.lower()}_lsh_gan_generated_mixdata_iter{i}.csv"
        pd.DataFrame(synthetic).to_csv(os.path.join(out_dir, fname),
                                       index=False, header=False)
        print(f"    Saved {fname}  shape={synthetic.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  Method 4 — Vanilla GAN
# ══════════════════════════════════════════════════════════════════════════

LATENT_GAN  = 100
BATCH_GAN   = 64
EPOCHS_GAN  = 2001
LR_GAN      = 0.0001

class VanillaGenerator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_GAN, 32), nn.Tanh(),
                                 nn.Linear(32, n_features))

    def forward(self, z):
        return self.net(z)

class VanillaDiscriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 32), nn.LeakyReLU(0.2),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)

def train_and_generate_gan(real, n_features, device, seed, ds_name, wanted_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)

    G = VanillaGenerator(n_features).to(device)
    D = VanillaDiscriminator(n_features).to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR_GAN)
    opt_D = optim.Adam(D.parameters(), lr=LR_GAN)
    bce = nn.BCEWithLogitsLoss()
    n = real.shape[0]

    for epoch in range(EPOCHS_GAN):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_GAN):
            batch = real[idx[i:i + BATCH_GAN]]
            bsz = len(batch)
            batch_t = torch.tensor(batch, device=device)
            z = torch.randn(bsz, LATENT_GAN, device=device)

            fake = G(z).detach()
            d_loss_real = bce(D(batch_t), torch.ones(bsz, 1, device=device))
            d_loss_fake = bce(D(fake), torch.zeros(bsz, 1, device=device))
            d_loss = d_loss_real + d_loss_fake
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            z = torch.randn(bsz, LATENT_GAN, device=device)
            g_loss = bce(D(G(z)), torch.ones(bsz, 1, device=device))
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        if epoch % 500 == 0:
            print(f"    GAN   ep {epoch:5d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * real.shape[0]).astype(int)
    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}", "gan")
    os.makedirs(out_dir, exist_ok=True)

    for i, n_gen in enumerate(batch_sizes):
        if i != wanted_iter:
            continue
        z = torch.randn(n_gen, LATENT_GAN, device=device)
        with torch.no_grad():
            synthetic = G(z).cpu().numpy()
        fname = f"{ds_name.lower()}_gan_generated_mixdata_iter{i}.csv"
        pd.DataFrame(synthetic).to_csv(os.path.join(out_dir, fname),
                                       index=False, header=False)
        print(f"    Saved {fname}  shape={synthetic.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  Method 5 — GAT-GAN
# ══════════════════════════════════════════════════════════════════════════

EPOCHS_GAT_CLASS = 7501
EPOCHS_GAT_GAN   = 20001
ND_GAT = 5
NG_GAT = 2
LR_GAT_G = 0.0002
LR_GAT_D = 0.0004

class GATG_Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hsize=(1024, 1024)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc_out = nn.Linear(hsize[1], output_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, z):
        h1 = self.lrelu(self.fc1(z))
        h2 = self.lrelu(self.fc2(h1))
        return self.fc_out(h2)

class GATG_Discriminator(nn.Module):
    def __init__(self, input_dim, hsize=(512, 256)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc3 = nn.Linear(hsize[1], input_dim)
        self.fc_out = nn.Linear(input_dim, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x_in):
        h1 = self.lrelu(self.fc1(x_in))
        h2 = self.lrelu(self.fc2(h1))
        h3 = self.fc3(h2)
        out_logits = self.fc_out(h3)
        return out_logits, h3


def gat_subsample(X, y, index_list, k, device):
    try:
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GAT-GAN. Install with: pip install torch_geometric")

    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)
    adjacency = torch.zeros((len(X), len(X)), dtype=torch.float32)
    for i in range(len(X)):
        adjacency[i, indices[i]] = 1.0
    row, col = adjacency.nonzero().t()
    edge_index = torch.stack([row, col], dim=0)

    class GATClassifier(nn.Module):
        def __init__(self, num_features, num_classes, priority_weight):
            super().__init__()
            self.conv1 = GATConv(num_features, 32, heads=8)
            self.conv2 = GATConv(32 * 8, num_classes, heads=1)
            self.priority_weight = priority_weight

        def forward(self, data):
            x, ei, pn = data.x, data.edge_index, data.priority_nodes
            x, _ = self.conv1(x, ei, return_attention_weights=True)
            attn = torch.ones(x.size(0), device=x.device)
            attn[pn] += self.priority_weight
            x = x * attn.view(-1, 1)
            x = torch.relu(x)
            x, attn2 = self.conv2(x, ei, return_attention_weights=True)
            return x, attn2

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        priority_nodes=torch.tensor(index_list, dtype=torch.long),
    ).to(device)

    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    model = GATClassifier(num_features, num_classes, priority_weight=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(EPOCHS_GAT_CLASS):
        model.train()
        optimizer.zero_grad()
        output, attn2 = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"    GAT classifier ep {epoch:5d}  loss={loss.item():.4f}")

    attention_coefficients = attn2[0]
    edge_source = data.edge_index[0]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    num_nodes = X_t.shape[0]
    attention_weights = torch.zeros(num_nodes, device=device)
    for i in range(num_nodes):
        mask = edge_source == i
        inc = attention_coefficients[:, mask].float()
        node_attn = torch.mean(inc, dim=1)
        attention_weights[i] = torch.mean(node_attn)

    sorted_indices = torch.argsort(attention_weights, descending=True)
    return sorted_indices[:k]


def train_and_generate_gatgan(real, labels, n_features, device, seed, ds_name, wanted_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_labels = pd.Series(labels).value_counts()
    rare_types = class_labels[class_labels <= 200].index.tolist()
    df_with_labels = pd.DataFrame(np.column_stack([real, labels]))
    if len(rare_types) == 0:
        rare_types = class_labels.index[:1].tolist()
    mask = pd.Series(labels).isin(rare_types)
    index_list = np.where(mask.values)[0].tolist()

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_sample = real.shape[0]
    k = math.ceil(n_sample * 0.2)

    print(f"    GAT subsampling: n={n_sample}, k={k}, rare_types={rare_types}")
    top_k_indices = gat_subsample(real, y_enc, index_list, k, device)
    Xnew = real[top_k_indices.cpu().numpy(), :]
    print(f"    Subsampled shape: {Xnew.shape}")

    col = Xnew.shape[1]
    G = GATG_Generator(col, col).to(device)
    D = GATG_Discriminator(col).to(device)
    opt_G = optim.RMSprop(G.parameters(), lr=LR_GAT_G)
    opt_D = optim.RMSprop(D.parameters(), lr=LR_GAT_D)
    criterion_gan = nn.BCEWithLogitsLoss()

    row = real.shape[0]

    for epoch in range(EPOCHS_GAT_GAN):
        X_batch = torch.tensor(real, dtype=torch.float32).to(device)
        row1 = row - Xnew.shape[0]
        if row1 < 0:
            row1 = 0
        da1 = sample_Z(row1, col)
        Z_batch_np = np.vstack((da1, Xnew))
        Z_batch = torch.tensor(Z_batch_np, dtype=torch.float32).to(device)

        for _ in range(ND_GAT):
            opt_D.zero_grad()
            real_output, _ = D(X_batch)
            fake_samples = G(Z_batch).detach()
            fake_output, _ = D(fake_samples)
            real_labels_t = torch.full_like(real_output, 0.9)
            fake_labels_t = torch.full_like(fake_output, 0.1)
            disc_loss = criterion_gan(real_output, real_labels_t) \
                      + criterion_gan(fake_output, fake_labels_t)
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()

        for _ in range(NG_GAT):
            opt_G.zero_grad()
            generated_samples = G(Z_batch)
            gen_output, _ = D(generated_samples)
            gen_loss = criterion_gan(gen_output, torch.ones_like(gen_output))
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

        if epoch % 5000 == 0:
            print(f"    GATGAN ep {epoch:6d}  D={disc_loss.item():.4f}  G={gen_loss.item():.4f}")

    batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_sample).astype(int)
    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}", "gat_gan")
    os.makedirs(out_dir, exist_ok=True)

    for i, n_gen in enumerate(batch_sizes):
        if i != wanted_iter:
            continue
        row1_gen = n_gen - Xnew.shape[0]
        if row1_gen < 0:
            row1_gen = 0
        da1_gen = sample_Z(row1_gen, n_features)
        Z_batch_gen_np = np.vstack((da1_gen, Xnew))
        Z_batch_gen = torch.tensor(Z_batch_gen_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            g_plot_np = G(Z_batch_gen).cpu().numpy()
        fname = f"{ds_name.lower()}_gat_gan_generated_mixdata_iter{i}.csv"
        pd.DataFrame(g_plot_np).to_csv(os.path.join(out_dir, fname),
                                       index=False, header=False)
        print(f"    Saved {fname}  shape={g_plot_np.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"#  SEED = {seed}")
        print(f"{'#'*70}")

        for ds_name, ds_cfg in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"  {ds_name}  (iter={ds_cfg['iter']})")
            print(f"{'='*60}")

            real = load_real(ds_cfg)
            n_features = real.shape[1]
            print(f"  Real shape: {real.shape}")

            wanted_iter = ds_cfg["iter"]

            methods_to_run = [
                ("[1/5] WGAN",      train_and_generate_wgan,   [real, n_features, device, seed, ds_name, wanted_iter]),
                ("[2/5] f-GAN",     train_and_generate_fgan,   [real, n_features, device, seed, ds_name, wanted_iter]),
                ("[3/5] LSH-GAN",   train_and_generate_lshgan, [real, n_features, device, seed, ds_name, wanted_iter]),
                ("[4/5] Vanilla GAN", train_and_generate_gan,  [real, n_features, device, seed, ds_name, wanted_iter]),
            ]

            labels = load_labels(ds_cfg)
            methods_to_run.append(
                ("[5/5] GAT-GAN",   train_and_generate_gatgan, [real, labels, n_features, device, seed, ds_name, wanted_iter])
            )

            for label, func, args in methods_to_run:
                print(f"  {label}")
                try:
                    func(*args)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue

    print("\nDONE: All 5 methods × 5 seeds × 4 datasets generated.")


if __name__ == "__main__":
    main()
