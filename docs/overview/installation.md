# Installation

**GARAGE** is a two-staged python pipeline that used the Graph Attention (GAT) and GAN to generate synthetic scRNA-seq data.

---

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
