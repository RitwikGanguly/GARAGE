# Adjusted Rand Index (ARI)
## ARI Overview

The Adjusted Rand Index (ARI) is a widely used metric in clustering analysis that quantifies the similarity between two clusterings by adjusting for the chance grouping of elements. Unlike the Rand Index, which can give high scores to random clusterings, the ARI corrects for this by considering the expected similarity of all pairwise assignments given the cluster sizes.

### Advantages

- Accounts for chance in random clustering scenarios.
- Ranges from -1 to 1, indicating complete disagreement to perfect agreement.

### Use Cases

- Evaluate clusters in machine learning.
- Compare different clustering algorithms
- ARI score determines the simiarities between different clusters.

## ARI score in GAT-GAN:
```python
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y,gen.obs['leiden'])
```
The ARI is computed based on the contingency table of the two clusterings, where the counts of data points falling into the same or different clusters are compared. This metric is essential in scenarios where the clustering structure needs to be validated against a known ground truth or when comparing the results of different clustering methods.
The ARI scores are provided for both real data and generated data of severel datasets of our work, with the scores being obtained from feature selection applied to each data type.The ARI is a crucial metric in clustering analysis as it quantifies the similarity between two data clusterings, adjusting for the chance grouping of elements.
