# Summary

## Project Title
**Synthetic Cell Sample Generation using Graph Attention-based Generative Adversarial Network**

## Project Description
In this work, we aim to enhance the efficiency and utility of single-cell RNA sequencing (scRNA-seq) data by employing a two-step process:

1. **Subsampling with Graph Attention Networks (GAT):**  
   We first perform subsampling to select a representative set of nodes with fewer sample counts from the initial dataset. GATs employ a self-attention mechanism to compute hidden representations for each node, selecting the most relevant information from its neighbors in the graph structure. This allows GATs to adaptively assign different weights to neighboring features based on their importance, enhancing the model’s ability to capture complex relationships within the graph. This step reduces data complexity and dimensionality while preserving essential information.

2. **Synthetic Data Generation with Generative Adversarial Networks (GANs):**  
   After subsampling, we use GANs to generate new synthetic cell samples based on the subsampled dataset. These generated data points expand the dataset, enabling more comprehensive analysis and improving the robustness of single-cell clustering.

This methodology is not limited to a single dataset (e.g., Yan or CBMC datasets) and can be applied to various single-cell clustering datasets. The approach enhances scalability and versatility. The final output includes both the subsampled dataset and the generated synthetic data, paving the way for more effective single-cell clustering analysis in genomics research.
