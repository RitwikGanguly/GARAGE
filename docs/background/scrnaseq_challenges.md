# scRNA-seq Data Challenges

## The Structure of scRNA-seq Data

Let $X \in \mathbb{R}^{n \times g}$ be a gene expression matrix with $n$ cells and $g$ genes. Each entry $X_{ij}$ represents the expression level (often log-normalised counts) of gene $j$ in cell $i$.

### Key Properties

1. **High-dimensionality:** $g$ ranges from $2{,}000$ to $>30{,}000$ genes. Typically $g \gg n$.
2. **Sparsity:** Most entries are zero (the "zero-inflation" problem). Typical dropout rates are 50–90%.
3. **Heteroscedastic noise:** Variance depends on mean expression level — highly expressed genes show more technical noise. This is captured by the coefficient of variation ($\text{CV}^2 = \sigma^2 / \mu^2$).
4. **Batch effects:** Technical variation between sequencing runs, labs, or protocols can mimic or mask biological signal.

---

## Rare Cell Collapse in Generative Models

Consider a dataset with $K$ cell types where type $k$ has $n_k$ cells. If one type is rare (e.g., $n_{\text{rare}} = 10$ while $n_{\text{abundant}} = 5{,}000$), standard generative models produce output that is overwhelmingly from the abundant types.

**Why this happens:**
- The GAN generator's loss gradient is dominated by the abundant types.
- The discriminator rarely sees real rare cells, so it cannot penalise the generator for failing to produce them.
- Mode collapse reinforces this — once the generator settles on producing abundant cell types, it has no incentive to explore rare ones.

### The Fano Index as a Diagnostic

The **Fano index** (variance-to-mean ratio) quantifies gene expression variability:

$$\text{Fano}(g) = \frac{\sigma_g^2}{\mu_g}$$

Genes with high Fano indices have high biological variability and often define rare cell type identities. A generator that does not preserve high-Fano genes is dropping rare cell types.

---

## Dimensionality Reduction for scRNA-seq

scRNA-seq data requires aggressive dimensionality reduction before clustering or GAN training:

| Method | Use in GARAGE | When to use |
|--------|--------------|-------------|
| **PCA** | Pre-step before UMAP | Reduce from $g$ to 50–100 components. |
| **CV² filtering** | Select top-$k$ genes (default: 100) | Retain the most variable genes. |
| **Fano index** | Alternative to CV² | Useful for count-based data. |
| **PCA loading** | Rank genes by loading on PC1–PC3 | Focus on genes that define major axes of variation. |
| **UMAP** | 2D visualisation | Qualitative sanity check. |
| **Leiden clustering** | Community detection for ARI/NMI/F1 | Quantitative comparison of real vs. generated. |

Co-dependence between these methods: feature selection ($\to$ PCA $\to$ KNN graph $\to$ Leiden clustering $\to$ ARI/NMI/F1/UMAP.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Leiden resolution too low** | Only 1–2 clusters found. | Increase resolution (range: 0.1–3.0 depending on dataset). |
| **Leiden resolution too high** | Every cell is its own cluster. | Decrease resolution. |
| **Feature selection too aggressive** | ARI ~ 0 on all methods. | Increase `n_genes` or use PCA loading instead of CV². |
| **GAT attention collapse** | All cells get equal attention. | Increase `priority_weight` or reduce `gat_epochs`. |
| **CUDA OOM on CBMC** | 7,895 × 2,000 matrix with KNN graph. | Reduce `k` in KNN or use CPU fallback. |

---

## Further Reading

- Hicks, S.C., et al. "Missing data and technical variability in single-cell RNA-sequencing experiments." *Biostatistics*, 2018.
- Luecken, M.D. and Theis, F.J. "Current best practices in single-cell RNA-seq analysis: a tutorial." *Molecular Systems Biology*, 2019.
