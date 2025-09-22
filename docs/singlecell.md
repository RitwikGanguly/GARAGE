# Single Cell Clustering

## 1) What is single-cell clustering ?

### Overview

Single-cell clustering is a computational technique applied in genomics to analyze single-cell RNA sequencing (scRNA-seq) data. This method groups individual cells into clusters based on gene expression profiles, aiding in the identification of cell types and states.

### Key Steps

1. **Data Preprocessing:**
   - Remove noise, correct artifacts, and normalize expression values in raw scRNA-seq data.

2. **Dimensionality Reduction:**
   - Utilize techniques like PCA, t-SNE, or UMAP to reduce dimensionality and visualize relationships between cells.

3. **Clustering Algorithms:**
   - Employ K-means, hierarchical clustering, density-based methods (e.g., DBSCAN), or graph-based methods (e.g., Graph Attention Networks) for grouping cells.

4. **Visualization:**
   - Visualize results using t-SNE plots, UMAP plots, or other techniques to explore relationships between different cell clusters.

### Applications

- **Cell Type Identification:**
  - Single-cell clustering helps identify distinct cell types within a heterogeneous tissue. This is crucial for understanding the cellular composition of organs and tissues.

- **Disease Subtyping:**
  - Clustering can reveal disease-specific cell subpopulations, aiding in the identification of subtypes and potential biomarkers for diseases such as cancer

- **Developmental Biology:**
  - Studying gene expression at the single-cell level allows researchers to track cell differentiation and developmental processes, providing insights into tissue development.

- **Immunology:**
  - Identifying different immune cell populations helps in understanding immune responses, discovering rare cell types, and characterizing immune cell heterogeneity.

- **Drug Discovery:**
  - Identify target cells, understand cellular responses, and uncover potential drug resistance mechanisms.

- **Rare Cell Identification:**
  - Detecting and characterizing rare cell populations, which may be crucial in understanding diseases or uncovering unique cellular functions.
    
## 2) Comparison between single-cell RNA sequencing (scRNA-seq) technology and bulk RNA sequencing (RNA-seq) technology:

### Single-Cell RNA Sequencing (scRNA-seq)
1. **Resolution**: Analyzes gene expression at the level of individual cells.
2. **Heterogeneity**: Detects cellular heterogeneity within a tissue or sample, identifying different cell types and states.
3. **Applications**: Useful for studying complex tissues, developmental processes, and rare cell populations.
4. **Data Complexity**: Generates high-dimensional data requiring advanced computational methods for analysis.
5. **Sample Size**: Requires isolation of individual cells, which can be challenging and requires specialized equipment.
6. **Cost**: Typically more expensive due to the complexity of isolating and processing individual cells.
7. **Sensitivity**: Can detect rare transcripts and subtle differences in gene expression between cells.
8. **Library Preparation**: Each cell generates its own RNA library, leading to a large number of libraries for sequencing.

### Bulk RNA Sequencing (RNA-seq)
1. **Resolution**: Analyzes average gene expression across a bulk population of cells.
2. **Heterogeneity**: Cannot distinguish between different cell types or states within the sample; provides an averaged expression profile.
3. **Applications**: Suitable for samples with homogeneous cell populations or for obtaining an overall gene expression profile of a tissue.
4. **Data Complexity**: Generates lower-dimensional data compared to scRNA-seq, simpler to analyze.
5. **Sample Size**: Requires a larger amount of starting material but does not require individual cell isolation.
6. **Cost**: Generally less expensive due to simpler sample preparation and fewer libraries to sequence.
7. **Sensitivity**: May miss rare transcripts and subtle differences in gene expression due to averaging effects.
8. **Library Preparation**: One RNA library is prepared from the pooled RNA of all cells in the sample, leading to fewer libraries for sequencing.

### Summary
- **Single-cell RNA-seq** provides high-resolution insights into the cellular heterogeneity within a sample but is more complex and expensive.
- **Bulk RNA-seq** gives an averaged gene expression profile of a population of cells and is more straightforward and cost-effective.

![Bulk RNA](https://cdn.10xgenomics.com/image/upload/f_auto,q_auto,w_680,h_510,c_limit/v1574196658/blog/singlecell-v.-bulk-image.png)

<p align="center"><small>Image source: <a href="https://www.10xgenomics.com/resources/blog/single-cell-vs-bulk-rna-seq">10x Genomics</a></small></p>
