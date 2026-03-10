from pathlib import Path
import sys

import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.int8
STAT_DTYPE = np.float64

DEBUG = False
# DEBUG = True

MAX_SAMPLES = 100 if DEBUG else 2500
MAX_SVC_SAMPLES = 10 if DEBUG else 100
NJOBS = 1 if DEBUG else 23

print(f"DEBUG mode is {'on' if DEBUG else 'off'}.")

TMP_DIR = Path("../tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
#
# if DEBUG:
#     RESULT_DIR = TMP_DIR / "results"
# else:
RESULT_DIR = Path("../results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DIRTS_DIR = RESULT_DIR / "figures"
DIRTS_DIR.mkdir(parents=True, exist_ok=True)

PMLB_DATA_DIR = Path("../pmlb_data")
PMLB_DATA_FILE = "data.parquet"
PMLB_METADATA_PATH = PMLB_DATA_DIR / "metadata.csv"
TARGET_COL = "target"

dataset_scores_name = "dataset_scores.csv"

def get_dataset_dir(dataset_name: str) -> Path:
    return PMLB_DATA_DIR / dataset_name

def get_dataset_file(name: str) -> Path:
    return get_dataset_dir(name) / PMLB_DATA_FILE

# ---------- Naming helpers ----------

def _knn_base(k_label: str) -> str:
    return f"{k_label}-NN"

def get_knn_mcc_name(k_label: str) -> str:
    return f"{_knn_base(k_label)} MCC"

def get_knn_acc_name(k_label: str) -> str:
    return f"{_knn_base(k_label)} Acc"

def get_knn_auroc_name(k_label: str) -> str:
    return f"{_knn_base(k_label)} AUROC"

def get_weighted_knn_name(k_label: str) -> str:
    # Base (no metric suffix); compose with get_knn_*_name afterwards
    return f"Weighted {k_label}-NN"

def get_knn_mcc_name_from_base(base: str) -> str:
    return f"{base} MCC"

def get_knn_acc_name_from_base(base: str) -> str:
    return f"{base} Acc"

def get_knn_auroc_name_from_base(base: str) -> str:
    return f"{base} AUROC"

def knn_k_labels():
    # The complete set requested
    return ("1", "3", "5", "7", "√n", "2%", "5%", "n-1", "n")

def _all_knn_metric_names():
    names = []
    for k in knn_k_labels():
        # Majority (unweighted) k-NN: MCC + Acc
        names.append(get_knn_mcc_name(k))
        names.append(get_knn_acc_name(k))
        # Weighted k-NN: MCC + Acc + AUROC (continuous score)
        wbase = get_weighted_knn_name(k)
        names.append(get_knn_mcc_name_from_base(wbase))
        names.append(get_knn_acc_name_from_base(wbase))
        names.append(get_knn_auroc_name_from_base(wbase))
    return tuple(names)

class Lens:
    # Non-kNN diagnostics
    spectral   = "Spectral Clustering"
    rndldd_auroc = "RNLDD"
    svm        = "SVC"
    svm_loo    = "SVC-LOO"
    agg_cluster = "Aggregate Clustering"
    k_medoids  = "K-Medoids"

    # Canonical “1-NN” convenience names (match generated ones)
    one_nn     = get_knn_mcc_name("1")
    one_nn_acc = get_knn_acc_name("1")

    # Full registry used to initialize result frames
    all_names = (
        spectral, agg_cluster, k_medoids, rndldd_auroc, svm, svm_loo,
        * _all_knn_metric_names()
    )