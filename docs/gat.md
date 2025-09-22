# Graph Attention Networks

## GAT Overview

Graph Attention Networks (GAT) is a type of neural network architecture designed to work with data represented as graphs. GATs are particularly useful for tasks involving graph-structured data, such as node classification, link prediction, and graph classification. GAT is basically a type of graph neural network (GNN) derived from Graph Convolutional Networks (GCN) by introducing the concept of attention mechanisms into the graph convolutional operation. GCNs are also a type of neural network architecture that works on graph-structured data. They performs convolutional operations over the graph by aggregating information from neighboring nodes. But, GCNs treat neighbouring nodes equally aggregation, i.e. they don’t use any specific extra attentions to certain nodes, regardless of their relevance to the target node. This limitation may lead to sub-optimal performance, especially in graphs with varying degrees of node importance. 

## 1) Key Features

- **Attention Mechanism:**
   - GAT uses attention mechanisms to allow nodes to weigh the importance of neighboring nodes differently during the learning process.
This attention mechanism enables GAT to capture complex relationships within the graph, giving more weight to relevant nodes.

- **Node-Level Representations:**
   - GAT learns node-level representations, capturing intricate relationships in graph-structured data.
The model produces embeddings that reflect the characteristics of nodes and their connections.

- **Versatility:**
   - Applicable to various domains, including social network analysis, citation networks, and biological network analysis.
In the context of biology, GAT can be applied to study protein-protein interaction networks or gene co-expression networks.

## 2) Applications

- **Biological Network Analysis:**
  - Analyze protein-protein interaction networks and gene regulatory networks.

- **Drug-Target Interaction Prediction:**
  - Predict interactions between drugs and target proteins.

- **Social Network Analysis:**
  - Model social networks to identify influential individuals and detect communities.

- **Recommendation Systems:**
  - Apply in recommendation systems to model user-item interactions.

- **Fraud Detection:**
  - Employed in fraud detection scenarios for accurate detection based on complex relationships.
 
## Difference between GAT and GCN
Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) are both popular architectures used for processing graph-structured data. While they share similarities in their ability to operate on graphs, they differ fundamentally in how they aggregate and update node features. Here is a pointwise comparison:

### Graph Convolutional Network (GCN)
1. **Aggregation Mechanism**:
   - GCNs use a neighborhood aggregation scheme where each node aggregates features from its neighbors using a weighted sum. The weights are usually based on the degree of the nodes and are fixed once the graph structure is defined.
   
2. **Weight Sharing**:
   - In GCNs, the weights used for aggregation are shared across all edges in the graph. This means that the same transformation is applied to all nodes regardless of their specific neighborhood structure.
   
3. **Formulation**:
   The typical formulation of a GCN layer can be expressed as:

    ![image1](assets/images/GCN_Eq.png)
   
4. **Interpretability**:
   - GCNs are relatively easier to interpret compared to GATs because the aggregation mechanism is straightforward and deterministic.

5. **Scalability**:
   - GCNs can face scalability issues when applied to very large graphs due to the necessity of computing neighborhood aggregations over potentially large sets of neighbors.

## Graph Attention Network (GAT)
1. **Aggregation Mechanism**:
   - GATs use attention mechanisms to aggregate features from neighbors. Each neighbor's contribution is weighted by an attention score, which is learned dynamically based on the features of the nodes involved.
   
2. **Weight Sharing**:
   - In GATs, the weights for aggregation are not shared uniformly; instead, the attention mechanism assigns different weights to different edges based on the features of the connected nodes. This allows for a more nuanced aggregation process.
   
3. **Formulation**:
   - The typical formulation of a GAT layer involves computing attention coefficients \(\alpha_{ij}\) for each edge, and then aggregating the neighbor features weighted by these coefficients:
     ![image2](assets/images/GAT_eq.png)
   
4. **Interpretability**:
   - GATs are less interpretable due to the dynamic and learned nature of the attention weights, making it harder to understand the exact influence of each neighbor.
   
5. **Scalability**:
   - GATs can be more computationally intensive than GCNs because computing attention coefficients for each edge adds overhead. However, the attention mechanism can also lead to more efficient and focused aggregations, potentially improving performance on certain tasks.

**In summary:
- **GCNs** rely on fixed aggregation weights based on the graph structure and degree normalization, making them simpler and potentially more scalable for certain types of graphs.
- **GATs** use an attention mechanism to dynamically assign weights to neighbors, allowing for more expressive and potentially more powerful modeling of graph data at the cost of increased computational complexity and less interpretability.**


Each architecture has its strengths and is suited to different types of graph-related problems. GCNs are often preferred for their simplicity and efficiency, while GATs are chosen for their flexibility and ability to capture complex relationships in the data.

## GAT implementation in GAT-GAN:
### 1) Build the KNN Graph and prioritize the nodes:
``` python
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn_model.fit(X)
_, indices = nn_model.kneighbors(X)
adjacency_matrix = torch.zeros((len(X), len(X)), dtype=torch.float32)

for i in range(len(X)):
    adjacency_matrix[i, indices[i]] = 1.0

# Convert adjacency matrix to a COO format tensor
row, col = adjacency_matrix.nonzero().t()
edge_index = torch.stack([row, col], dim=0)

priority_nodes = torch.tensor(index_list, dtype=torch.long)
```
A k-nearest neighbors graph is created to find the k nearest neighbors for each data point in the dataset. An adjacency matrix is created to represent the graph. Each entry (i, j) in this matrix is set to 1 if node j is a neighbor of node i. The adjacency matrix is converted into a Coordinate List (COO) format edge index, which is commonly used in graph processing libraries. A tensor of priority nodes is created from a list of indices, which can be used for specific analyses or operations within the graph.

### 2) Build the GAT model:
``` python
class GATClassifier(nn.Module):
    def __init__(self, num_features, num_classes, priority_weight):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=8)
        self.conv2 = GATConv(16 * 8, num_classes, heads=1)
        self.priority_weight = priority_weight

    def forward(self, data):
        x, edge_index, priority_nodes = data.x, data.edge_index, data.priority_nodes

        # First GAT layer
        x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)

        # Modify attention scores to give more priority to specific nodes
        attention = torch.ones(x.size(0), device=x.device)  # Initialize with ones
        attention[priority_nodes] += self.priority_weight  # Increase attention to priority nodes
        x = x * attention.view(-1, 1)  # Element-wise multiplication with attention scores

        x = torch.relu(x)

        # Second GAT layer
        x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)

        # You can now access attention coefficients for each node from attention1 and attention2

        return x, attention1, attention2
```
In our GAT model, the nodes which are belonging to the less samples cell types are subsampled from the model. Basically all the nodes are assigned some attention score but the priority nodes are given more attention score, and thus the model returned the attention score of the subsampled nodes.

---
