## Computational requirements and reproducibility settings

All experiments were performed using Python-based implementations of the GARAGE
pipeline.  Two dedicated Conda environments---``ritwik_base`` (Python 3.12.5)
and ``scrna`` (Python 3.9.25)---were used to isolate the model-training and
validation workflows, respectively.  The corresponding dependency files are
provided in the GitHub repository as ``requirements_garage.txt``
(model-training environment, including the core GARAGE pipeline, the
state-of-the-art and single-cell-specific benchmarking baselines, the ablation
studies, and the biological analysis modules) and
``requirements_validation.txt`` (validation environment, including feature
selection, Leiden clustering, adjusted Rand index, normalised mutual
information, macro-averaged F1-score, UMAP visualisation, Wasserstein distance,
and marker-gene enrichment analysis).

The model-training environment (``ritwik_base``) was configured with Python
3.12.5, PyTorch 2.7.1 (CUDA 11.8), PyTorch Geometric 2.7.0, NumPy 2.4.4,
pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.9, seaborn
0.13.2, AnnData 0.12.16, and NetworkX 3.6.1.  The validation environment
(``scrna``) was configured with Python 3.9.25, Scanpy 1.10.3, leidenalg 0.12.0,
UMAP-learn 0.5.12, POT 0.9.5, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1,
scikit-learn 1.6.1, Matplotlib 3.9.4, seaborn 0.13.2, and AnnData 0.10.9.

All model-training experiments were performed on a single NVIDIA RTX A6000 GPU
(48~GB VRAM, CUDA 11.8) with an AMD EPYC 7713P 64-core processor (64 threads)
running Ubuntu 24.04 LTS and 314~GB system RAM.  The smaller datasets, Yan
(124~cells) and Pollen (301~cells), completed training on CPU in under five
minutes; the larger datasets, Muraro (2,126~cells) and CBMC (7,895~cells),
required a GPU for efficient training.  Wasserstein distance computations and
clustering evaluations were performed on CPU.

To improve reproducibility, a fixed random seed (``random_seed = 42`` in the
``GARAGE_DEFAULTS`` dictionary of ``config.py``) was used for all stochastic
operations, including PyTorch weight initialisation, NumPy random number
generation, the GAT training loop, GAN generator and discriminator
initialisation, synthetic-data generation, and scikit-learn cross-validation
splits.  All hyper-parameters are enumerated in the repository's ``config.py``
with their default values and a rationale for each choice.  The repository
README and the ReadTheDocs documentation
(\url{https://garage-docs.readthedocs.io/en/latest/}) provide exact setup
commands, per-script usage examples, expected outputs, and a step-by-step
end-to-end tutorial for reproducing the core GARAGE results on the four
built-in datasets (Yan, Pollen, CBMC, and Muraro).  Users who wish to apply
GARAGE to custom single-cell RNA-seq data are referred to the ``Preparing Your
Data'' section of the documentation.
