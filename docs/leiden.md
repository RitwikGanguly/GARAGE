# Leiden Clustering
Leiden clustering is an advanced community detection algorithm used to uncover highly interconnected groups, or communities, within complex networks. It improves upon the Louvain algorithm by addressing its limitations, particularly in ensuring that the communities detected are well-connected. The Leiden algorithm operates by optimizing a modularity score, which measures the density of connections within communities compared to connections with the rest of the network. 

## The Leiden algorithm enhances community detection in networks through three key steps:

- **Local Moving of Nodes:** It starts by moving nodes between clusters
to achieve a locally optimal grouping.
- **Refinement:** Each cluster is then refined to ensure internal connec-
tivity, preventing the formation of fragmented clusters.
- **Aggregation:** After refinement, clusters are aggregated into super-
nodes, and the algorithm iterates the process on this new network
level. This cycle repeats until it can no longer enhance the network’s
modularity, indicating an optimal community structure has been reached.
The result is a network partitioned into cohesive, well-connected com-
munities.

## Role of Leiden Clustering in UMAP Visulization

Leiden clustering complements UMAP by quantitatively defining clusters within the visually represented data. It’s favored for several reasons:
- **Enhanced Cluster Detection:** Leiden clustering quantifies the clus-ters that UMAP visually uncovers, providing a more definitive cluster identification.
- **Scalability:** The efficiency and scalability of Leiden clustering make it suitable for large datasets that are typically analyzed using UMAP.
- **Accuracy:** Offering greater accuracy and robustness.

The Leiden Clustering in our work is used to plot the UMAP and compare the data distribution with the generated and real data - 
- For Real Data

``` python
#Leiden clustering
import leidenalg
sc.tl.leiden(real,resolution=1.5)
##visualizing clusters
sc.pl.umap(real, color=['leiden'])
```
- For Generated Data

``` python
#Leiden clustering
import leidenalg
sc.tl.leiden(gen,resolution=1.78999)
##visualizing clusters
sc.pl.umap(gen, color=['leiden'])
```
- Clustering images of real & generated data:

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/real_umap_gatgat_yan_k_30_cv2_actual_labels.pdf" alt="Image 1" width="300"/><br>Yan real</td>
    <td><img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/generated_umap_gatgat_yan_k_30_cv2_actual_labels.pdf" alt="Image 2" width="300"/><br>Yan gen</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/real_umap_gatgan_pollen_cv2_actual_label%20(1).pdf" alt="Image 5" width="300"/><br>Pollen real</td>
    <td><img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/generated_umap_gatgat_pollen_k_10_cv2_actual_labels.pdf" alt="Image 6" width="300"/><br>Pollen gen</td>
  </tr>
</table>

we utilized Uniform Manifold Approximation and Projection (UMAP) to visualize the clustering results. UMAP effectively displays the high-dimensional data in a low-dimensional space, preserving both local and global structures. The UMAP plots provide a visual representation of the clustering, with the first plot showing the clusters of real data and the second plot depicting the clusters of generated data. These visualizations highlight the distinct groupings and validate the effectiveness of our feature selection, dimensionality reduction, and clustering approaches. UMAP, when combined with PCA and Leiden clustering, forms a powerful toolset for data visualization and analysis. PCA helps in preprocessing by reducing noise and  dimensionality, making the dataset more manageable for UMAP. UMAP then provides a visual representation that preserves the data’s structure, facilitating the identification of patterns and clusters. Finally, Leiden clustering quantifies these clusters, providing a robust and accurate solution for cluster detection. This integrated approach is particularly valuable in fields dealing with high-dimensional data, such as bioinformatics and image processing. It not only enhances interpretability and efficiency but also provides a comprehensive method for exploring and understanding complex datasets. By following the steps of PCA preprocessing, UMAP transformation, and Leiden clustering, data scientists can achieve meaningful insights and improved model performance, making this combination a critical component in modern data analysis workflows.
