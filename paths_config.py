import os

# Base project directory
VLM_ROOT = "/path/to/PathKGen"

# Key subdirectories
DATA_DIR = os.path.join(VLM_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CHECKPOINTS_DIR = os.path.join(VLM_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(VLM_ROOT, "results")
