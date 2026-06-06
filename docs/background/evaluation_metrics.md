# Evaluation Metrics

GARAGE uses a multi-metric evaluation strategy to assess synthetic data quality. No single metric is sufficient — distributional metrics (WD, MMD, SWD) measure *fidelity*, while clustering metrics (ARI, NMI, F1) measure *biological utility*.

---

## Distributional Metrics

### Wasserstein Distance (Earth Mover's Distance)

The Wasserstein distance between two probability distributions $P$ (real) and $Q$ (generated) is:

$$W(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]$$

where $\Gamma(P, Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

**In GARAGE:** We compute the exact Earth Mover's Distance via Optimal Transport (`ot.emd2`) on the Euclidean distance matrix of the (normalised) expression matrices.

- **Range:** $\geq 0$; lower is better.
- **Typical good values:** $< 0.01$ for small datasets (Yan), $< 0.005$ for large datasets (CBMC).

**See:** `data_generation/wasserstein_distance.py` and the [How-to Guide](howto/wasserstein_distance).

---

### Maximum Mean Discrepancy (MMD)

MMD uses a kernel function $k(x, y)$ to measure the distance between distributions in a Reproducing Kernel Hilbert Space (RKHS):

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x'} [k(x, x')] + \mathbb{E}_{y,y'} [k(y, y')] - 2\mathbb{E}_{x,y} [k(x, y)]$$

**See:** `analysis/distribution_metrics.py`

---

### Sliced Wasserstein Distance (SWD)

SWD projects high-dimensional data onto random 1D lines and computes the Wasserstein distance in each projection:

$$\text{SWD}(P, Q) = \int_{\mathbb{S}^{d-1}} W(P_\theta, Q_\theta) \, d\theta$$

where $P_\theta$ is the projection of $P$ along direction $\theta$.

SWD is computationally cheaper than WD and scales well to large datasets.

**See:** `analysis/swd_analysis.py`

---

## Clustering-Based Metrics

### Adjusted Rand Index (ARI)

The ARI measures the similarity between two clusterings, corrected for chance:

$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$

where $n_{ij}$ is the number of objects in both cluster $i$ (of clustering A) and cluster $j$ (of clustering B).

- **Range:** $-1$ to $1$; $1$ = perfect agreement.
- **Procedure in GARAGE:** `adjusted_rand_score(y_true, y_pred_leiden)` where `y_pred_leiden` is the Leiden clustering of the generated data and `y_true` is the ground-truth cell type labels.

---

### Normalised Mutual Information (NMI)

$$\text{NMI}(U, V) = \frac{I(U; V)}{\sqrt{H(U) H(V)}}$$

where $I(U; V)$ is the mutual information between clusterings $U$ and $V$, and $H(\cdot)$ is the entropy.

- **Range:** $0$ to $1$; $1$ = perfect agreement.
- NMI is less sensitive to cluster size imbalance than ARI.

---

### Macro-F1 Score

For $K$ cell types, the macro-F1 is the average of per-class F1 scores:

$$\text{macro-F1} = \frac{1}{K} \sum_{k=1}^{K} 2 \cdot \frac{\text{precision}_k \cdot \text{recall}_k}{\text{precision}_k + \text{recall}_k}$$

- **Range:** $0$ to $1$; higher is better.
- Macro-F1 gives equal weight to each cell type, making it especially useful for evaluating rare-cell preservation.

---

## Visualisation

### UMAP (Uniform Manifold Approximation and Projection)

UMAP projects high-dimensional data into 2D while preserving both local and global neighbourhood structure. It is used in GARAGE for **qualitative comparison** of real and generated data clusters.

**Procedure:**
1. Apply PCA (50 components) to reduce noise.
2. Compute the neighbourhood graph (n_neighbors=15).
3. Run UMAP to project to 2D.
4. Colour by Leiden cluster or ground-truth cell type.

**See:** `data_validation/data_validation.py` and the [How-to Guide](howto/clustering_validation).

<p align="center">
  <img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/real_umap_gatgat_yan_k_30_cv2_actual_labels.pdf" alt="Yan real UMAP" width="300"/>
  <img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/generated_umap_gatgat_yan_k_30_cv2_actual_labels.pdf" alt="Yan generated UMAP" width="300"/>
</p>

*Example: Real (left) and GARAGE-generated (right) UMAP embeddings for the Yan dataset, coloured by Leiden cluster.*

---

### PCA (Principal Component Analysis)

PCA transforms data into orthogonal components ranked by variance explained:

$$X_{\text{reduced}} = X \, W_k$$

where $W_k$ is the matrix of the top-$k$ eigenvectors of the covariance matrix $X^T X$.

In GARAGE, PCA serves two roles:
1. **Preprocessing before UMAP:** Reduces noise and computational cost.
2. **Feature selection:** PCA loadings on the first 3 components rank genes by importance.

---

## Feature Selection Methods

GARAGE offers three feature selection approaches to identify the most informative genes before clustering:

### 1. CV² (Coefficient of Variation Squared)

$$\text{CV}^2_g = \frac{\sigma_g^2}{\mu_g^2}$$

Select the top-$k$ genes with the highest CV². High CV² indicates high relative variability — a marker of biological informativeness in scRNA-seq.

### 2. Fano Index

$$\text{Fano}_g = \frac{\sigma_g^2}{\mu_g}$$

Select the top-$k$ genes with the highest Fano index. Similar to CV² but with different scaling — the Fano index is less penalising of high-mean genes.

### 3. PCA Loading

Fit PCA on the generated data $X_{\text{gen}}$. For each gene $g$, aggregate its absolute loading across the first 3 principal components:

$$\ell_g = |w_{g,1}| + |w_{g,2}| + |w_{g,3}|$$

Select the top-$k$ genes by $\ell_g$. Genes with high PCA loadings are the primary drivers of variance.

**Default:** CV² with $k=100$ genes. See [Feature Selection How‑to](howto/feature_selection).

---

## Leiden Clustering

Leiden clustering is a community detection algorithm that partitions cells into clusters by optimising modularity:

$$Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \gamma \frac{d_i d_j}{2m} \right) \delta(c_i, c_j)$$

where $A_{ij}$ is the (PCA-based KNN) adjacency matrix, $d_i$ is node degree, and $\gamma$ is the **resolution parameter**.

- **Resolution $\uparrow$** → More clusters (higher granularity).
- **Resolution $\downarrow$** → Fewer clusters (coarser grouping).

In GARAGE, Leiden is run over a resolution sweep (dataset-specific range in `data_validation/data_validation.py`) to find the optimal partition for ARI/NMI/F1 reporting.

---

## Recommended Evaluation Workflow

1. **Run GARAGE** → generate synthetic data.
2. **Compute WD** → check distributional match. If $> 0.1$, re-train with more iterations or higher leakage.
3. **Feature selection** (CV², 100 genes) → select informative genes.
4. **Leiden clustering** on generated data (resolution sweep).
5. **ARI / NMI / macro-F1** against ground-truth labels.
6. **UMAP** → visual sanity check.
7. **Biological validation** (marker-gene enrichment, rare-cell recall) → confirm biological utility.

This protocol is implemented in `data_validation/data_validation.py` and the [End-to-End Tutorial](tutorials/01_end_to_end).
