========
Overview
========

.. raw:: html

   <p align="center">
     <img src="https://raw.githubusercontent.com/RitwikGanguly/GARAGE/refs/heads/main/docs/images/github_title_garage.png" alt="GARAGE" width="680"/>
     <h1 align="center">GARAGE</h1>
     <h3 align="center">Graph-Attentive Rare-cell-Aware single-cell data GEneration</h3>
   </p>

.. raw:: html

   <p align="center">

     <a href="https://www.python.org/downloads/release/python-3125/">
     <img alt="Python" src="https://img.shields.io/badge/python-3.12.5-blue"/>
     </a>

     <a href="https://garage-docs.readthedocs.io/en/latest/">
       <img alt="Documentation Status" src="https://readthedocs.org/projects/garage/badge/?version=latest"/>
     </a>

     <a href="https://github.com/RitwikGanguly/GARAGE/blob/main/LICENSE">
       <img alt="License" src="https://img.shields.io/github/license/RitwikGanguly/GARAGE"/>
     </a>
   </p>

.. raw:: html

   <p align="center">
     <img alt="Docs" src="https://img.shields.io/badge/Docs-Sphinx-blue"/>
     <img alt="Linting" src="https://img.shields.io/badge/Linting-flake8%20black%20mypy-yellow"/>
   </p>

----

.. admonition:: How to read these docs
   :class: tip

   GARAGE's documentation follows the **Diátaxis** framework to serve
   different needs at different points in your journey:

   * **Getting Started** — first-time setup and your first synthetic dataset.
   * **Tutorials** — step-by-step, end-to-end walkthroughs for research use cases.
   * **How‑to Guides** — focused recipes for specific tasks.
   * **Theoretical Background** — understanding the architecture and methods.
   * **API Reference** — function signatures, module-level documentation.
   * **Appendix** — glossary, FAQ, troubleshooting, citation, and changelog.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   summary
   installation
   quickstart
   prepare_your_data

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_end_to_end
   tutorials/02_biological_validation
   tutorials/03_rare_cell_experiment
   tutorials/04_benchmark_against_sota

.. toctree::
   :maxdepth: 2
   :caption: How‑to Guides

   howto/run_garage
   howto/feature_selection
   howto/clustering_validation
   howto/wasserstein_distance
   howto/biological_validation
   howto/benchmarking
   howto/ablations
   howto/interpret_outputs

.. toctree::
   :maxdepth: 2
   :caption: Theoretical Background

   background/motivation
   background/scrnaseq_challenges
   background/garage_architecture
   background/gan
   background/gat
   background/evaluation_metrics
   singlecell
   wasserstein

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/config
   api/data_generation
   api/data_validation
   api/biological_analysis
   api/ablation_study
   api/analysis

.. toctree::
   :maxdepth: 2
   :caption: Appendix

   glossary
   faq
   troubleshooting
   citation
   changelog

.. raw:: html

   <p align="center">
     <small>Image credits: GARAGE architecture image by the authors; 10x Genomics scRNA-seq comparison
     used under fair-use for educational illustration.</small>
   </p>
