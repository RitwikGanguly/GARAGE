biological_analysis module
===========================

.. automodule:: biological_analysis.biological_validation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: biological_analysis.rare_cell_utility
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: biological_analysis.marker_gene_clustering
   :members:
   :undoc-members:
   :show-inheritance:

Modules Overview
----------------

- ``biological_validation`` — trains the GAT classifier, extracts attention weights, and performs enrichment analysis (Fisher's exact test, Wilcoxon rank-sum, log₂ fold change) for rare-cell marker genes.
- ``rare_cell_utility`` — held-out rare-cell classification experiment: splits data, re-trains GARAGE/GAN/LSH-GAN, generates synthetic rare cells, and evaluates a Random Forest classifier on rare-cell recall and F1.
- ``marker_gene_clustering`` — grid search over clustering parameters (feature selection method × top genes × resolution) for marker-gene-based evaluation.
