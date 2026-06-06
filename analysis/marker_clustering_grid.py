#!/usr/bin/env python
"""Fixed marker-gene clustering evaluation (grid sweep).
===========================================================
Methods: GAN, VAE, LSH-GAN, GARAGE.
Metrics: ARI, NMI.

All methods sweep Leiden resolution 0.1–3.0 (step 0.01) across all datasets.
Pseudo-labels: NearestCentroid; fallback 3-NN majority vote.

Outputs:
  - results/marker_genes.csv
  - results/clustering_performance.csv

Usage:  conda run -n scrna python run_rev5_marker_clustering_grid.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os, warnings
warnings.filterwarnings("ignore")

REAL_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
GEN_ROOT  = os.path.join(os.path.dirname(__file__), "..", "data", "gen_data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

ITER_MAP     = {"Yan": 3, "Pollen": 5, "CBMC": 3, "Muraro": 3}
N_PCS        = 20
N_NEIGHBORS  = 30

RES_SWEEP = np.arange(0.10, 3.01, 0.01)

DATASET_CONFIG = {
    "Yan": {
        "real_file": "yan_process.csv",
        "transpose": True, "real_header": None,
        "label_file": "yan_celltype.csv", "label_header": None, "label_col": 0,
        "cell_types": ["zygote", "2cell", "4cell", "8cell", "16cell", "blast"],
    },
    "Pollen": {
        "real_file": "pollen_process.txt",
        "transpose": False, "real_header": None,
        "label_file": "pollenc.txt", "label_header": None, "label_col": 0,
        "cell_types": ["HL60", "K562", "Kera", "BJ", "GW16", "iPS",
                       "2338", "2339", "GW21+3", "NPC", "GW21"],
    },
    "CBMC": {
        "real_file": "cbmc_rna_scaled.csv",
        "transpose": True, "real_header": 0, "real_index_col": 0,
        "label_file": "cell_type_cbmc.csv", "label_header": 0, "label_col": "x",
        "cell_types": ["Eryth", "NK", "CD14+ Mono", "CD16+ Mono", "CD34+",
                       "CD8 T", "Memory CD4 T", "Naive CD4 T", "B", "Mk",
                       "pDCs", "DC", "T/Mono doublets"],
    },
    "Muraro": {
        "real_file": "muraro_expression_matrix.csv",
        "transpose": False, "real_header": 0,
        "label_file": "muraro_cell_types.csv", "label_header": 0, "label_col": "cell_type",
        "cell_types": ["alpha", "beta", "delta", "ductal", "acinar",
                       "endothelial", "gamma", "epsilon", "mesenchymal", "unclear"],
    },
}

METHODS = [
    ("GAN",     "gan",      "gan",         "generated"),
    ("VAE",     "vae",      "vae",         "generated"),
    ("LSH-GAN", "lsh_gan",  "lsh_gan",     "generated"),
    ("GARAGE",  None,       None,          None),
]

CBMC_MARKERS = {
    "Eryth":      ["HBB", "HBA1", "HBA2", "GYPA", "AHSP", "ALAS2", "SLC4A1", "KLF1"],
    "NK":         ["NKG7", "GNLY", "KLRF1", "PRF1", "GZMB", "KLRD1"],
    "CD14+ Mono": ["CD14", "S100A8", "S100A9", "CST3", "FCN1"],
    "CD16+ Mono": ["FCGR3A", "CDKN1C"],
    "CD34+":      ["CD34", "KIT", "PROM1"],
    "CD8 T":      ["CD8A", "CD8B", "GZMK"],
    "Memory CD4 T": [],
    "Naive CD4 T": [],
    "B":          ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1"],
    "Mk":         ["PPBP", "PF4", "ITGA2B", "GP9", "GP1BA"],
    "pDCs":       ["LILRA4", "TCF4", "CLEC4C", "IRF7", "IRF8"],
    "DC":         ["FCER1A", "CLEC10A", "FLT3"],
    "T/Mono doublets": [],
}

MURARO_MARKERS = {
    "alpha":       ["GCG", "TTR", "ARX", "IRX2"],
    "beta":        ["INS", "IAPP", "NKX6-1", "MAFA", "PDX1", "DLK1"],
    "delta":       ["SST", "HHEX", "RBP4", "LEPR"],
    "ductal":      ["KRT19", "CFTR", "SOX9", "SPP1", "MMP7", "TFF1", "TFF3"],
    "acinar":      ["CPA1", "PRSS1", "CTRB1", "CTRB2", "CELA3A", "AMY2A",
                    "REG1A", "REG1B"],
    "endothelial": ["PECAM1", "CDH5", "VWF", "CLDN5", "ENG", "KDR", "PLVAP"],
    "gamma":       ["PPY", "PAX6"],
    "epsilon":     ["GHRL"],
    "mesenchymal": ["COL1A1", "COL1A2", "PDGFRA", "THY1", "LUM", "DCN"],
}


# ═══════════  DATA LOADING  ═══════════
def load_real_data(dataset):
    cfg = DATASET_CONFIG[dataset]
    rk = {"header": cfg["real_header"]}
    if "real_index_col" in cfg:
        rk["index_col"] = cfg["real_index_col"]
    real = pd.read_csv(os.path.join(REAL_DIR, cfg["real_file"]), **rk)
    if cfg["transpose"]:
        real = real.T
    real = real.values.astype(np.float64)
    lbl = pd.read_csv(os.path.join(REAL_DIR, cfg["label_file"]),
                      header=cfg["label_header"])
    if cfg["label_header"] is not None:
        lbl = lbl[cfg["label_col"]].values.ravel()
    else:
        lbl = lbl.iloc[:, 0].values.ravel()
    return real, lbl


def load_gen_data(dataset, dir_name, prefix, suffix, iter_idx):
    fname = f"{dataset.lower()}_{prefix}_{suffix}_mixdata_iter{iter_idx}.csv"
    fpath = os.path.join(GEN_ROOT, dir_name, fname)
    gen = pd.read_csv(fpath, header=None).values.astype(np.float64)
    if dataset == "Yan" and gen.shape[1] < gen.shape[0]:
        gen = gen.T
    return gen


def load_garage_data(dataset):
    lds = dataset.lower()
    fmap = {"cbmc": "cbmc_data_mixdata_iter3_top_1579.csv",
            "muraro": "muraro_data_mixdata_iter3_top_426.csv",
            "pollen": "pollen_data_mixdata_iter5_top_60_new.csv",
            "yan": "yan_data_mixdata_iter3_top_20.csv"}
    path = os.path.join(GEN_ROOT, fmap[lds])
    return pd.read_csv(path, index_col=0).values.astype(np.float64)


# ═══════════  MARKER GENE SELECTION  ═══════════
def select_markers_cbmc(real, labels):
    cfg = DATASET_CONFIG["CBMC"]
    rk = {"header": cfg["real_header"]}
    if "real_index_col" in cfg:
        rk["index_col"] = cfg["real_index_col"]
    df = pd.read_csv(os.path.join(REAL_DIR, cfg["real_file"]), **rk)
    gene_names = [str(g).strip() for g in df.index]
    gene_upper = [g.upper() for g in gene_names]
    all_marker_idx, marker_details = [], {}
    for ctype, markers in CBMC_MARKERS.items():
        found = []
        for m in markers:
            if m in gene_upper:
                idx = gene_upper.index(m)
                if idx not in all_marker_idx:
                    all_marker_idx.append(idx)
                    found.append(m)
        marker_details[ctype] = found
    if len(all_marker_idx) < 30:
        top_supp = _select_by_between_type_variance(real, labels, 80)
        for idx in top_supp:
            if idx not in all_marker_idx:
                all_marker_idx.append(idx)
        all_marker_idx = all_marker_idx[:80]
    mgs = {ct: ", ".join(gs) if gs else "variance-selected"
           for ct, gs in marker_details.items()}
    return np.array(all_marker_idx), mgs


def select_markers_muraro(real, labels):
    cfg = DATASET_CONFIG["Muraro"]
    df = pd.read_csv(os.path.join(REAL_DIR, cfg["real_file"]),
                     header=cfg["real_header"])
    raw_cols = df.columns.tolist()
    gene_map = {}
    for i, raw in enumerate(raw_cols):
        gene_symbol = str(raw).split("__")[0].strip().upper()
        gene_map[gene_symbol] = i
    all_marker_idx, marker_details = [], {}
    for ctype, markers in MURARO_MARKERS.items():
        found = []
        for m in markers:
            m_up = m.upper()
            if m_up in gene_map:
                idx = gene_map[m_up]
                if idx not in all_marker_idx:
                    all_marker_idx.append(idx)
                    found.append(m)
        marker_details[ctype] = found
    if len(all_marker_idx) < 40:
        top_supp = _select_by_between_type_variance(real, labels, 80)
        for idx in top_supp:
            if idx not in all_marker_idx:
                all_marker_idx.append(idx)
        all_marker_idx = all_marker_idx[:80]
    mgs = {ct: ", ".join(gs) if gs else "variance-selected"
           for ct, gs in marker_details.items()}
    return np.array(all_marker_idx), mgs


def _select_by_between_type_variance(real, labels, top_k=50):
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    unique_types = np.unique(enc)
    K, G = len(unique_types), real.shape[1]
    type_means = np.zeros((K, G))
    for k_idx, k in enumerate(unique_types):
        type_means[k_idx] = real[enc == k].mean(axis=0)
    return np.argsort(-np.var(type_means, axis=0))[:top_k]


def select_markers_yan(real, labels):
    top_idx = _select_by_between_type_variance(real, labels, 100)
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    type_names = le.classes_
    marker_details = {}
    for i, tn in enumerate(type_names):
        mask = enc == i
        other_mask = enc != i
        fc = real[mask].mean(axis=0) - real[other_mask].mean(axis=0)
        top_for_type = np.argsort(-np.abs(fc))[:12]
        overlap = [int(j) for j in top_for_type if j in top_idx]
        marker_details[str(tn)] = _format_indices(overlap[:10])
    return top_idx, marker_details


def select_markers_pollen(real, labels):
    top_idx = _select_by_between_type_variance(real, labels, 120)
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    type_names = le.classes_
    marker_details = {}
    for i, tn in enumerate(type_names):
        mask = enc == i
        other_mask = enc != i
        fc = real[mask].mean(axis=0) - real[other_mask].mean(axis=0)
        top_for_type = np.argsort(-np.abs(fc))[:10]
        overlap = [int(j) for j in top_for_type if j in top_idx]
        marker_details[str(tn)] = _format_indices(overlap[:8])
    return top_idx, marker_details


def _format_indices(indices):
    if not indices:
        return "variance-selected"
    if len(indices) <= 4:
        return "[" + ",".join(str(x) for x in indices) + "]"
    return "[" + ",".join(str(x) for x in indices[:3]) + ",...," + str(indices[-1]) + "]"


# ═══════════  PSEUDO-LABELS  ═══════════
def get_pseudo_labels(gen_filt, real_filt, real_labels_enc):
    """Primary: NearestCentroid on real -> predict on gen.
    Fallback (NC < 2 classes): 3-NN majority vote on real data (cosine)."""
    nc = NearestCentroid()
    nc.fit(real_filt, real_labels_enc)
    labels_nc = nc.predict(gen_filt)

    if len(np.unique(labels_nc)) >= 2:
        return labels_nc

    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(real_filt)
    _, idx = nn.kneighbors(gen_filt)
    fallback = np.empty(len(idx), dtype=real_labels_enc.dtype)
    for i in range(len(idx)):
        fallback[i] = np.bincount(real_labels_enc[idx[i]]).argmax()
    return fallback


# ═══════════  CLUSTERING / EVALUATION  ═══════════
def compute_ari_nmi(gen_filt, real_filt, real_labels_enc,
                    resolution, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    pseudo_labels = get_pseudo_labels(gen_filt, real_filt, real_labels_enc)

    adata = sc.AnnData(gen_filt.astype(np.float64))
    adata.var_names_make_unique()
    actual_n_pcs = min(n_pcs, gen_filt.shape[1] - 1, gen_filt.shape[0] - 1)
    actual_n_pcs = max(2, actual_n_pcs)

    try:
        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, gen_filt.shape[0]-1),
                        n_pcs=actual_n_pcs, metric="cosine")
        sc.tl.leiden(adata, resolution=round(resolution, 4), random_state=42)
        y_pred = adata.obs["leiden"].astype(str).astype(int).to_numpy()
    except Exception:
        return np.nan, np.nan

    ari = adjusted_rand_score(pseudo_labels, y_pred)
    nmi = normalized_mutual_info_score(pseudo_labels, y_pred)
    return ari, nmi


def evaluate_sweep(gen_filt, real_filt, real_labels_enc,
                   n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    best_ari, best_nmi = -1, -1
    for res in RES_SWEEP:
        ari, nmi = compute_ari_nmi(gen_filt, real_filt, real_labels_enc,
                                   resolution=res, n_pcs=n_pcs, n_neighbors=n_neighbors)
        if not np.isnan(ari) and ari > best_ari:
            best_ari, best_nmi = ari, nmi
    if best_ari < 0:
        return np.nan, np.nan
    return best_ari, best_nmi


def evaluate_real_reference(real_filt, real_labels_enc,
                             n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    adata = sc.AnnData(real_filt.astype(np.float64))
    adata.var_names_make_unique()
    actual_n_pcs = min(n_pcs, real_filt.shape[1] - 1, real_filt.shape[0] - 1)
    actual_n_pcs = max(2, actual_n_pcs)

    try:
        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, real_filt.shape[0]-1),
                        n_pcs=actual_n_pcs, metric="cosine")
    except Exception:
        return np.nan, np.nan

    best_ari, best_nmi = -1, -1
    for res in RES_SWEEP:
        try:
            sc.tl.leiden(adata, resolution=round(res, 4), random_state=42)
            y_pred = adata.obs["leiden"].astype(str).astype(int).to_numpy()
            ari = adjusted_rand_score(real_labels_enc, y_pred)
            nmi = normalized_mutual_info_score(real_labels_enc, y_pred)
            if ari > best_ari:
                best_ari, best_nmi = ari, nmi
        except Exception:
            continue
    if best_ari < 0:
        return np.nan, np.nan
    return best_ari, best_nmi


def evaluate_clustering(gen_data, real_data, real_labels, marker_idx,
                        method_name, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    gen_filt  = gen_data[:, marker_idx]
    real_filt = real_data[:, marker_idx]

    if gen_filt.shape[0] < 5:
        return np.nan, np.nan

    le = LabelEncoder()
    real_labels_enc = le.fit_transform(real_labels)

    if method_name == "Real (marker ref.)":
        return evaluate_real_reference(real_filt, real_labels_enc, n_pcs, n_neighbors)
    else:
        return evaluate_sweep(gen_filt, real_filt, real_labels_enc, n_pcs, n_neighbors)


# ═══════════  MAIN  ═══════════
def main():
    marker_rows, perf_rows = [], []

    for dataset in DATASET_CONFIG:
        print(f"\n{'='*70}")
        print(f"  {dataset}")
        print(f"{'='*70}")
        real, labels_raw = load_real_data(dataset)
        iter_idx = ITER_MAP[dataset]
        print(f"  Real shape: {real.shape}  |  iter={iter_idx}")

        if dataset == "CBMC":
            marker_idx, marker_genes_str = select_markers_cbmc(real, labels_raw)
            sel_method = "Literature-based blood cell markers"
        elif dataset == "Muraro":
            marker_idx, marker_genes_str = select_markers_muraro(real, labels_raw)
            sel_method = "Literature-based pancreatic cell markers"
        elif dataset == "Yan":
            marker_idx, marker_genes_str = select_markers_yan(real, labels_raw)
            sel_method = "Between-cell-type variance (Yan et al. 2013)"
        elif dataset == "Pollen":
            marker_idx, marker_genes_str = select_markers_pollen(real, labels_raw)
            sel_method = "Between-cell-type variance (Pollen et al. 2014)"
        else:
            raise ValueError(dataset)

        print(f"  {len(marker_idx)} markers  |  {sel_method}")

        for ct in DATASET_CONFIG[dataset]["cell_types"]:
            gs = marker_genes_str.get(ct, "variance-selected")
            marker_rows.append((dataset, ct, gs, sel_method, len(marker_idx)))

        r_ari, r_nmi = evaluate_clustering(
            real, real, labels_raw, marker_idx, "Real (marker ref.)")
        perf_rows.append((dataset, "Real (marker ref.)", _fmt(r_ari), _fmt(r_nmi)))
        print(f"  {'Real (ref)':15s}  ARI={r_ari:.4f}  NMI={r_nmi:.4f}")

        for method_name, dir_name, prefix, suffix in METHODS:
            try:
                if method_name == "GARAGE":
                    gen = load_garage_data(dataset)
                else:
                    gen = load_gen_data(dataset, dir_name, prefix, suffix, iter_idx)
                ari, nmi = evaluate_clustering(
                    gen, real, labels_raw, marker_idx, method_name)
                perf_rows.append((dataset, method_name, _fmt(ari), _fmt(nmi)))
                print(f"  {method_name:15s}  cells={gen.shape[0]:6d}  "
                      f"ARI={ari:.4f}  NMI={nmi:.4f}")
            except Exception as e:
                perf_rows.append((dataset, method_name, "--", "--"))
                print(f"  {method_name:15s}  SKIP: {e}")

    df_markers = pd.DataFrame(marker_rows, columns=[
        "Dataset", "Cell type", "Marker genes", "Selection method", "N"])
    df_markers.to_csv(os.path.join(OUT_DIR, "marker_genes.csv"), index=False)

    df_perf = pd.DataFrame(perf_rows, columns=["Dataset", "Method", "ARI", "NMI"])
    df_perf.to_csv(os.path.join(OUT_DIR, "clustering_performance.csv"), index=False)

    print(f"\n{'='*80}")
    print(" TABLE Sx : Fixed marker-gene clustering performance")
    print(" All methods: Leiden resolution 0.1-3.0 sweep (step 0.01)")
    print(f"{'='*80}")
    print(df_perf.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return round(float(v), 4)


if __name__ == "__main__":
    main()
