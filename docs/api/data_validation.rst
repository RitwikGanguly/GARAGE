data_validation module
=======================

.. automodule:: data_validation.data_validation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: data_validation.feature_selection
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

The ``data_validation`` module contains:

- ``validate_generated_data()`` — end-to-end validation: loads data, runs feature selection, performs Leiden clustering over a resolution sweep, and reports ARI/NMI/macro-F1.
- ``cv2_selection()`` — selects top ``n_genes`` by coefficient of variation squared.
- ``fano_selection()`` — selects top ``n_genes`` by Fano index.
- ``pca_loading_selection()`` — selects top ``n_genes`` by PCA loading on the first 3 components.

The ``feature_selection`` module provides a standalone Python port of the original
``feature_selection.R`` script (Fano, PCA loading, CV²).

Resolution Sweep
----------------

The Leiden resolution sweep is controlled by ``data_validation.data_validation.RESOLUTION_RANGES``, a dictionary mapping dataset names to ``numpy`` arrays of resolution values.

Reference Notebook
------------------

The original validation notebook is preserved at
``data_validation/data_vaidation_garage.ipynb`` for reference and reproducibility.
