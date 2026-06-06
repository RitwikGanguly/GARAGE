# Glossary

Terms and concepts used throughout the GARAGE documentation and codebase.

---

## A

**ARI (Adjusted Rand Index)**  
A metric that measures the similarity between two clusterings, adjusting for chance. Range: $[-1, 1]$. Used in GARAGE to compare Leiden clusters of generated data to ground-truth cell type labels.

**Attention mechanism**  
A neural network component that assigns different weights to inputs based on their relevance. In GAT, attention scores determine how much each neighbor contributes to a node's update. In GARAGE, attention weights from the trained GAT are used to select seed cells.

---

## B

**Batch effect**  
Systematic technical variation between samples processed in different batches, labs, or protocols. Can confound biological signal in scRNA-seq data.

---

## C

**CV² (Coefficient of Variation Squared)**  
$\sigma^2 / \mu^2$ — a measure of relative variability for each gene. Used as a feature selection criterion: genes with high CV² are selected for clustering.

**Cell type**  
A biologically defined category of cells sharing a common identity and function (e.g., "T cell CD4+", "B cell", "neuron").

---

## D

**DDPM (Denoising Diffusion Probabilistic Model)**  
A generative model that learns to reverse a gradual noising process. Used in `benchmarking/scrna_seq_specific/scdiffusion.py`.

**Discriminator**  
The neural network in a GAN that tries to distinguish real data from generated data.

---

## E

**Embedding**  
A low-dimensional vector representation of a cell (or gene) learned by a neural network.

**Expression matrix**  
A matrix $X \in \mathbb{R}^{n \times g}$ where $X_{ij}$ is the expression level of gene $j$ in cell $i$.

---

## F

**Fano index**  
$\sigma^2 / \mu$ — variance-to-mean ratio for each gene. Used as an alternative to CV² for feature selection.

**Feature selection**  
The process of selecting a subset of informative genes from the full gene set before clustering or classification. GARAGE supports CV², Fano, and PCA loading methods.

---

## G

**GAN (Generative Adversarial Network)**  
A framework of two competing neural networks: a Generator that produces synthetic data and a Discriminator that tries to distinguish real from synthetic.

**GAT (Graph Attention Network)**  
A graph neural network that uses self-attention to weight the importance of neighboring nodes. Used in GARAGE to identify seed cells.

**Generator**  
The neural network in a GAN that produces synthetic data from input noise (or a hybrid batch in GARAGE).

---

## H

**Hybrid input batch**  
In GARAGE, the generator receives a mixed batch: $(1-\lambda) \cdot z_{\text{noise}} \oplus \lambda \cdot x_{\text{seed}}$, where a fraction $\lambda$ of the input is real seed cells from the GAT and the rest is random noise.

---

## L

**Leakage fraction ($\lambda$)**  
The proportion of the GAN generator's input batch that consists of real seed cells (selected by the GAT) rather than random noise. Default: 0.2.

**Leiden clustering**  
A community detection algorithm that partitions cells into clusters by optimising modularity. Used in GARAGE for quantitative evaluation.

---

## M

**Macro-F1**  
The average of per-class F1 scores, weighted equally. Used in GARAGE to evaluate rare-cell preservation — a low macro-F1 indicates that rare cell types are poorly captured.

**MMD (Maximum Mean Discrepancy)**  
A kernel-based metric for comparing two distributions. Computed in `analysis/distribution_metrics.py`.

**Mode collapse**  
A failure mode of GANs where the generator produces only a few types of output repeatedly, rather than the full diversity of the training data.

---

## N

**NMI (Normalised Mutual Information)**  
A normalised measure of mutual information between two clusterings. Range: $[0, 1]$. Less sensitive to cluster size imbalance than ARI.

---

## P

**PCA (Principal Component Analysis)**  
A linear dimensionality reduction method that projects data onto axes of maximum variance. Used in GARAGE as a preprocessing step before UMAP and as a feature selection criterion.

**PCA loading**  
The contribution (weight) of each original feature to a principal component. Genes with high aggregate loading on PC1–PC3 are selected as informative features.

**Priority node / Priority cell**  
A cell belonging to a rare cell type (with cell count below `rare_threshold`). These cells receive extra attention weight during GAT training.

**Priority weight**  
A scalar multiplier applied to rare-cell features in the GAT to boost their influence. Default: 2.0 (effective factor: $1 + 2.0 = 3.0$).

---

## R

**Rare cell type**  
A cell type with fewer than `rare_threshold` cells in the dataset. GARAGE is designed to explicitly preserve these populations in synthetic data.

**Resolution (Leiden)**  
A parameter controlling the granularity of Leiden clustering. Higher resolution → more clusters. Swept over a range to find the optimal partition for ARI/NMI/F1.

---

## S

**scRNA-seq (single-cell RNA sequencing)**  
A technology that measures gene expression in individual cells, revealing cellular heterogeneity that is masked in bulk RNA-seq.

**Seed cell**  
A real cell selected by the GAT as highly representative. Seed cells are mixed into the generator's input batch to guide and stabilise GAN training.

**SOTA (State of the Art)**  
Refers to the collection of baseline generative models implemented in `benchmarking/sota/` for comparison with GARAGE.

**SWD (Sliced Wasserstein Distance)**  
A computationally cheaper variant of the Wasserstein distance, computed by projecting data onto random 1D lines and averaging the resulting 1D Wasserstein distances.

---

## U

**UMAP (Uniform Manifold Approximation and Projection)**  
A non-linear dimensionality reduction technique for visualising high-dimensional data in 2D or 3D. Used in GARAGE for qualitative comparison of real and generated data clusters.

---

## V

**VAE (Variational Autoencoder)**  
A generative model that learns a latent representation of data via an encoder-decoder architecture with a KL divergence regularisation term.

---

## W

**Wasserstein distance (WD)**  
Also known as Earth Mover's Distance. Measures the minimum cost to transform one probability distribution into another. Used in GARAGE as a primary distributional fidelity metric. Computed via Optimal Transport.
