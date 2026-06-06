data_generation module
=======================

.. automodule:: data_generation.garage
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: data_generation.wasserstein_distance
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

The ``garage`` module contains:

- ``load_dataset()`` —    loads expression matrix and labels via ``config.DATASET_CONFIG``.
- ``gat_main()`` — trains the GAT classifier, extracts attention weights, and returns seed cell indices.
- ``Generator`` — MLP that receives the hybrid (noise + seeds) input batch.
- ``Discriminator`` — MLP binary classifier.
- ``sample_Z()`` — constructs the hybrid input batch $(1-\lambda) \cdot z \oplus \lambda \cdot x_{\text{seed}}$.

The ``wasserstein_distance`` module contains:

- ``compute_wasserstein()`` — returns the Earth Mover's Distance between two expression matrices.
