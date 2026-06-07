# Installation

GARAGE is a two-stage Python pipeline that uses Graph Attention (GAT) and GANs to generate synthetic scRNA-seq data.

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python    | 3.12.5  | Core GARAGE pipeline run |
| Python    | 3.7.12  | TF1.11 reference baselines (optional) |
| R         | any     | Legacy `feature_selection.R` (optional) |
| Git       | any     | Repository clone |

### Hardware Notes

- **GPU recommended** (CUDA-capable with ≥ 8 GB VRAM). The code auto-detects CUDA via `torch.cuda.is_available()`.
- **CPU-only** works for the small datasets (Yan, Pollen) but will be slow for CBMC and Muraro.
- **Apple Silicon** — PyTorch MPS support works; fallback to CPU if issues.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/RitwikGanguly/GARAGE.git
cd GARAGE
```

---

## Step 2: Set Up the Environment

We strongly recommend using a Conda environment:

```bash
conda create --name venv_garage python=3.12.5
conda activate venv_garage
```

Alternatively, use `venv`:

```bash
python -m venv venv_garage

# Linux / macOS:
source venv_garage/bin/activate

# Windows:
venv_garage\Scripts\activate
```

---

## Step 3: Install Dependencies

### Core pipeline (required)

```bash
pip install -r requirements_garage.txt
```

### Benchmarking dependencies (optional)

Use a **separate** environment (different Python version):

```bash
conda create --name venv_benchmarking python=3.7.12
conda activate venv_benchmarking
pip install -r requirements_benchmarking.txt
```

### Validation dependencies (recommended)

Use the ``scrna`` environment for data validation and biological validation:

```bash
conda create --name scrna python=3.9.25
conda activate scrna
pip install -r requirements_validation.txt
```

---

## Step 4: Verify Installation

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from config import DATASET_CONFIG; print('Datasets:', list(DATASET_CONFIG))"
```

---

## Step 5: Run the Pipeline

```bash
# Generate synthetic data (example: Muraro dataset)
python -m data_generation.garage --dataset muraro

# Validate the generated data
python -m data_validation.data_validation --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv

# Compute Wasserstein distance
python -m data_generation.wasserstein_distance --dataset muraro \
    --gen_csv data/gen_data/muraro_data_mixdata_iter3_top_426.csv
```

Proceed to [Quickstart](quickstart) for a detailed walkthrough, or see [Preparing Your Data](prepare_your_data) to plug in your own dataset.
