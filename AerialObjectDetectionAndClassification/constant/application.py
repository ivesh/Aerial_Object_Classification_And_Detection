# constant/application.py
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Configuration files
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config.yaml")
PARAMS_FILE_PATH = os.path.join(ROOT_DIR, "params.yaml")
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, "schema.yaml")

# Artifacts
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Data directories
CLASSIFICATION_DATA_DIR = "data/classification_dataset"
DETECTION_DATA_DIR = "data/object_detection_Dataset"

# Timestamp
TIMESTAMP: str = "2024_01_15-10_30_00"  # You can make this dynamic later