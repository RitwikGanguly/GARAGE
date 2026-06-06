config module
=============

.. automodule:: config
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   This module is the single source of truth for all paths, dataset
   configurations, and default hyper-parameters in GARAGE.  Import it from
   any script to ensure consistency.

Constants
---------

.. data:: DATASET_CONFIG

   Dictionary mapping dataset names to their file paths, loading parameters,
   rare-threshold, and iteration maps.  See :doc:`/prepare_your_data` for
   the full schema.

.. data:: GARAGE_DEFAULTS

   Dictionary of GAT and GAN hyper-parameter defaults (epochs, learning rates,
   hidden-layer sizes, leakage fraction, priority weight, etc.).

.. data:: DATA_DIR, CELL_TYPES_DIR, EXPRESSION_DIR, RESULTS_DIR

   Absolute paths to the data, cell-types, expression-matrix, and results
   directories respectively, derived from the repository root.
