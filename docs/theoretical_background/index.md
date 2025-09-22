# Project Title: Synthetic Cell sample generation using Graph Attention based Generative Adversarial Network
---

## Project Description: 
In this work, we aim to enhance the efficiency and utility of single-cell RNA sequencing data by employing a two-step process. 
First, we perform subsampling using Graph Attention Networks (GAT) to select a representative set of nodes with lesser number of sample value counts from the initial sample nodes in the dataset. It employs a self-attention mechanism to compute the hidden representations of each node and choose the most relevant information from its neighbours in the graph structure. This enables GATs to adaptively assign different weights or priorities to the features of neighbouring nodes based on their importance, thereby enhancing the model’s ability to capture complex relationships within the graph. This step helps reduce data complexity and dimensionality while preserving essential information.

Next, we employ Generative Adversarial Networks (GANs) to generate new sythetic cell samples based on the subsampled dataset. These generated data points expand the dataset, enabling more comprehensive analysis and improving the robustness of single-cell clustering.

This methodology is not limited to a single dataset(i.e. Yan, CMBC Dataset) but can be applied to various single-cell clustering datasets, enhancing the scalability and versatility of our approach. The final output includes both the subsampled dataset and the generated data, paving the way for more effective single-cell clustering analysis in genomics research.

---
