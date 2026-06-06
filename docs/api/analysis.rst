analysis module
=================

.. automodule:: analysis.distribution_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.clustering_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.aggregate_losses
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.build_summary_tables
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.sc_specific_benchmark
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.marker_clustering_grid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.plot_wasserstein_vs_leakage
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.mmd_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.swd_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Modules Overview
----------------

- ``distribution_metrics`` — batch computation of MMD and SWD for all methods × datasets.
- ``clustering_evaluation`` — feature selection + clustering across multiple random seeds.
- ``aggregate_losses`` — aggregates per-run GAN loss CSV files into a single record.
- ``build_summary_tables`` — builds mean ± std summary tables for WD, ARI, NMI, F1, MMD, SWD.
- ``sc_specific_benchmark`` — ARI/NMI/F1 for scRNA-seq-specific baselines only.
- ``marker_clustering_grid`` — grid search over clustering parameters.
- ``plot_wasserstein_vs_leakage`` — generates WD vs leakage fraction figure.
- ``mmd_analysis`` — standalone MMD computation and analysis.
- ``swd_analysis`` — standalone Sliced Wasserstein Distance computation and analysis.
