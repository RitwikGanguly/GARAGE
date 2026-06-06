"""
GARAGE — shared configuration and path constants.

All paths in the project should import from this file so users only need to
update paths in one place.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(REPO_ROOT, "data")
CELL_TYPES_DIR = os.path.join(DATA_DIR, "cell_types")
EXPRESSION_DIR = os.path.join(DATA_DIR, "expression_matrix")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DATASET_CONFIG = {
    "yan": {
        "expression_file": "yan_process.csv",
        "label_file": "yan_celltype.csv",
        "header": None,
        "transpose": True,
        "label_header": None,
        "label_col": 0,
        "rare_threshold": 10,
        "iter_map": [0, 1, 2, 3, 4, 5],
    },
    "pollen": {
        "expression_file": "pollen_process.txt",
        "label_file": "pollenc.txt",
        "header": None,
        "transpose": False,
        "label_header": None,
        "label_col": 0,
        "rare_threshold": 25,
        "iter_map": [0, 1, 2, 3, 4, 5],
    },
    "cbmc": {
        "expression_file": "cbmc_rna_scaled.csv",
        "label_file": "cell_type_cbmc.csv",
        "header": 0,
        "index_col": 0,
        "transpose": True,
        "label_header": 0,
        "label_col": "x",
        "rare_threshold": 200,
        "iter_map": [0, 1, 2, 3, 4, 5],
    },
    "muraro": {
        "expression_file": "muraro_expression_matrix.csv",
        "label_file": "muraro_cell_types.csv",
        "header": 0,
        "transpose": False,
        "label_header": 0,
        "label_col": "cell_type",
        "rare_threshold": 200,
        "iter_map": [0, 1, 2, 3, 4, 5],
    },
}

# ScRNA-seq-specific dataset keys used in benchmarking
SCRNASEQ_DATASETS = ["yan", "pollen", "cbmc", "muraro"]

# Default hyper-parameters for the GARAGE pipeline
GARAGE_DEFAULTS = {
    "gat_epochs": 7501,
    "gan_total_iters": 20001,
    "leakage_fraction": 0.2,
    "nd_steps": 5,
    "ng_steps": 2,
    "g_lr": 0.0002,
    "d_lr": 0.0004,
    "label_smooth_real": 0.9,
    "label_smooth_fake": 0.1,
    "generator_hidden": [1024, 1024],
    "discriminator_hidden": [512, 256],
    "priority_weight": 2,
    "random_seed": 42,
}
