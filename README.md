<p align="center">
  <img src="img/github_title_garage.png" alt="GARAGE" width="680"/>
<!--   <h1 align="center">GARAGE</h1> -->
</p>

# GARAGE - A Graph Attentive GAN for Rare Cell Aware single cell RNA-seq Data Generation

A python pipeline for synthetic single cell (scRNA-seq) data generation using graph attention-based GAN approach.

## 🔗 Related Links :
**Docs :** [GARAGE Documentation ☑️](https://garage-docs.readthedocs.io/en/latest/) 

---



<p align="center">

  <a href="https://www.python.org/downloads/release/python-3125/">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12.5-blue"/>
  </a>

  <!-- Documentation (if enabled) -->
  <a href="https://garage-docs.readthedocs.io/en/latest/">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/garage/badge/?version=latest"/>
  </a>

  <!-- License -->
  <a href="https://github.com/RitwikGanguly/GARAGE/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RitwikGanguly/GARAGE"/>
  </a>
  
  
  <!-- Gitter / Community Chat -->
  <a href="https://app.gitter.im/#/room/!FIUyTpwDzJtqorWCMm:gitter.im">
    <img alt="Gitter" src="https://badges.gitter.im/garage/garage.svg"/>
  </a>
</p>

<p align="center">
  <img alt="Poetry" src="https://img.shields.io/badge/Packaging-Poetry-blue"/>
  <img alt="Docs" src="https://img.shields.io/badge/Docs-Mkdocs-red"/>
  <img alt="Linting" src="https://img.shields.io/badge/Linting-flake8%20black%20mypy-yellow"/>
</p>

---

**GARAGE** (**G**raph-**A**ttentive **RA**re-cell aware single-cell data **GE**neration) is a novel deep learning framework for generating high-fidelity synthetic single-cell RNA-seq (scRNA-seq) data.

Traditional Generative Adversarial Networks (GANs) often struggle with the high-dimensional and sparse nature of scRNA-seq data, leading to training instability and a failure to reproduce rare but biologically important cell populations. GARAGE overcomes these challenges with a unique two-stage architecture that intelligently guides the generative process.

---

##  Workflow

The GARAGE framework uses a two-stage process to generate realistic synthetic cells, with a special focus on preserving rare cell types.

<p align="center">
  <img src="img/garage_workflow.jpg" alt="GARAGE" width="680"/>
</p>

*A high-level overview of the GARAGE framework.*

1.  **Stage 1: GAT-based Cell Selection:**
    A Graph Attention Network (GAT) is first trained on a cell-cell neighborhood graph. By leveraging its attention mechanism, the GAT identifies a core set of "archetypal" or high-importance cells that are most influential in defining the data's structure and cell-type identities.

2.  **Stage 2: GAT-Seeded GAN Generation:**
    Instead of receiving only random noise, the GAN's generator is fed a **hybrid input batch**. This batch is a mixture of random vectors and the high-priority "seed" cells selected by the GAT. This "attention-guided leakage" anchors the generator to known, biologically realistic states, stabilizing training and ensuring all cell types are represented.

---

## 🚀 Key Features

*   **GAT-Informed Seeding:** Moves beyond random sampling to intelligently select the most representative cells to guide generation.
*   **Enhanced Rare Cell Generation:** The framework is explicitly designed to better capture and generate samples for rare and underrepresented cell populations.
*   **Improved Stability & Convergence:** The seeded-generation process significantly stabilizes GAN training, reduces mode collapse, and accelerates convergence.
*   **High-Fidelity Synthetic Data:** Produces synthetic datasets ideal for data augmentation, methods benchmarking, and privacy-preserving data sharing.

---

## 📂 Repository Structure

```
├── .github/workflows/       # CI/CD workflows (e.g., for ReadTheDocs)
├── benchmarking/            # Scripts for benchmarking GARAGE against other models
    ├── 5 Models (.py)       # gan, wgan, fgan, lsh-gan, vae SOTA models
├── data/                    # Directory for input data
│   ├── cell_types/          # Folder for cell type label files (.csv)
│   └── expression_data/     # Folder for gene expression matrices (.csv)
├── data_generation/         # Core scripts for the GARAGE pipeline
│   ├── garage.py            # Main script for GAT selection and GAN training
│   └── wasserstein_distance.py     # Script to calculate Wasserstein Distance
├── data_validation/         # Scripts and notebooks for evaluating generated data
│   ├── feature_selection.R         # R script for Feature Selection analysis
│   └── data_vaidation_garage.ipynb # Notebook for ARI, NMI, F1-score, and UMAPs
├── docs/                    # Source files for documentation (ReadTheDocs)
├── img/                     # Images used in the README and docs
├── .gitignore               # Files to be ignored by Git
├── LICENSE                  # Project license (MIT)
├── readthedocs.yaml         # Configuration for ReadTheDocs
├── requirements_benchmarking.txt   # For Benchmarking in py(v3.7.12), packages to be installed
└── requirements_garage.txt  # For GARAGE pipeline in py(3.12.5), python packages to be installed
```
**Importent:** For GARAGE Model (v3.12.5) || For Benchmarking (v3.7.12) || For Data Validation (v3.9.21)

---

## 🛠️ Getting Started

Follow these instructions to set up your environment and run the GARAGE pipeline.

### Prerequisites

*   Git
*   Python (3.7 or higher) [python versions mentioned earlier]
*   R (for feature selection validation scripts)

### 1. Clone the Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/RitwikGanguly/GARAGE.git
cd GARAGE
```

### 2. Set Up the Environment

We strongly recommend using a virtual environment to manage dependencies.

```bash
# Create a virtual environment (Windows)
python -m venv venv_garage

# Create a virtual environment (Linux)
conda create --name venv_garage python=3.12.5
```

```bash
# Activate the environment

# On Windows:
.\venv_garage\Scripts\activate

# On Linux:
conda activate venv_garage
```

### 3. Install Dependencies

This repository contains two requirements files. For running the main pipeline, use `requirements_garage.txt`.

```bash
pip install -r requirements_garage.txt
```
If you also want to run the benchmarking scripts, install the additional dependencies:
```bash
pip install -r requirements_benchmarking.txt
```

**Disclaimer:** For all the code files, at the beginning of all, the required py version, necessary packages to be installed, every required dependencies are mentioned.

---

## ⚙️ Usage: The GARAGE Pipeline

To generate and validate your own synthetic single-cell data, the below steps need to be followed :

- Prepare Your Data 📁
- Run GARAGE 🧠
- Validate the Generated Data 📊

🍁 We have a dedicated documentation for GARAGE at - [GARAGE Documentation ☑️](https://garage-docs.readthedocs.io/en/latest/) 
🙋‍♂️ CALM DOWN❗Have a SEE 👀

## 📜 Citation

If you use **GARAGE** in your research, please cite our paper:

```bibtex
  TO BE ADDED SOON
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.








