ablation_study module
=======================

.. automodule:: ablation_study.leakage_ablation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ablation_study.multi_seed_synthesis
   :members:
   :undoc-members:
   :show-inheritance:

Modules Overview
----------------

- ``leakage_ablation`` — runs GARAGE on all 4 datasets with varying leakage fractions
  $\lambda \in \{0.0, 0.1, 0.2, 0.3\}$ and logs GAN losses to ``results/rev6_losses.csv``.
- ``multi_seed_synthesis`` — runs GARAGE with 5 different random seeds on all 4 datasets
  to assess reproducibility of the generated data.
