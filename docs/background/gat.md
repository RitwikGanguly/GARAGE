# Graph Attention Networks in GARAGE

## GAT Overview

Graph Attention Networks (GAT) are a class of graph neural networks that use self-attention to weigh the importance of neighboring nodes differently during message passing. Unlike Graph Convolutional Networks (GCNs), which treat all neighbors equally, GATs learn which neighbors are most relevant for each node.

### Attention Mechanism

For a node $i$ with neighbor $j$, the attention coefficient $\alpha_{ij}$ is computed as:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_k]))}$$

where:
- $\mathbf{h}_i$ is the feature vector of node $i$,
- $\mathbf{W}$ is a learnable weight matrix,
- $\mathbf{a}$ is a learnable attention vector,
- $\|$ denotes concatenation,
- $\mathcal{N}(i)$ is the neighbourhood of node $i$.

The output for node $i$ is then:

$$\mathbf{h}'_i = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right)$$

---

## GCN vs. GAT

| Property | GCN | GAT |
|----------|-----|-----|
| **Aggregation** | Fixed weights based on graph degree. | Dynamic, learned attention weights. |
| **Interpretability** | Simple and deterministic. | Less interpretable (learned weights vary per edge). |
| **Neighbor importance** | All neighbors equal (after degree normalisation). | Some neighbors are more important than others. |
| **Computational cost** | Lower. | Higher (attention computation per edge). |
| **Use case** | Homogeneous graphs. | Graphs with varying node relevance (e.g., rare vs. abundant cell types). |

GAT is the preferred choice for GARAGE because rare cell types should receive *higher* attention than abundant types — exactly what the attention mechanism enables.

---

## GAT in GARAGE: The Cell Selection Stage

GARAGE's first stage uses a GAT **classifier** to identify the most representative cells in the dataset.

### Step 1: Build the KNN Graph

```python
from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn_model.fit(expression_matrix)
_, indices = nn_model.kneighbors(expression_matrix)

# Convert to PyG edge index
edge_index = torch.stack([
    torch.arange(n_cells).repeat_interleave(5),
    torch.tensor(indices).flatten()
], dim=0)
```

Each cell becomes a graph node. Edges connect each cell to its 5 nearest neighbors in gene-expression space.

### Step 2: GAT Classifier with Priority Weighting

```python
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    def __init__(self, num_features, num_classes, priority_weight=2.0):
        super().__init__()
        self.conv1 = GATConv(num_features, 16, heads=8)
        self.conv2 = GATConv(16 * 8, num_classes, heads=1)
        self.priority_weight = priority_weight

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GAT layer: project to 128 dimensions (16 × 8 heads)
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)

        # Multiply rare-cell node features by priority_weight
        attention_boost = torch.ones(x.size(0), device=x.device)
        attention_boost[data.priority_nodes] += self.priority_weight
        x = x * attention_boost.unsqueeze(-1)

        x = torch.relu(x)

        # Second GAT layer: produce class logits
        x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)

        return x, attn_weights
```

**Key innovation — priority weighting:** Cells belonging to rare cell types (those with fewer than `rare_threshold` samples in `config.py`) receive an extra attention boost via `self.priority_weight` (default: 2.0). This ensures rare cells are not "averaged out" by abundant cells.

### Step 3: Extract Seed Cells

After training the GAT classifier:
1. Extract attention weights from the second GAT layer.
2. Aggregate per-head attention into a single score per cell.
3. Select the top $k = \lambda \cdot n$ cells, where $\lambda$ is the leakage fraction.

These selected cells become the "seeds" that are fed to the GAN generator in Stage 2.

---

## Full Implementation

See `data_generation/garage.py` for the complete GAT cell-selection implementation (`gat_main()` function). Key hyper-parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gat_epochs` | 7501 | Number of GAT training iterations |
| `priority_weight` | 2.0 | Attention boost for rare cell types |
| `leakage_fraction` | 0.2 | Fraction of cells selected as seeds |

---

## References

- Velickovic, P., et al. "Graph Attention Networks." *ICLR* 2018.
- Kipf, T. and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR* 2017.
