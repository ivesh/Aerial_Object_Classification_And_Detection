import os
import yaml
from box import ConfigBox
from pathlib import Path
import json
import joblib
from ensure import ensure_annotations
from typing import Any

# Remove ensure_annotations from read_yaml and create_directories
print("DEBUG: Loading main_utils.py - updated version with Path type")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read yaml file and returns ConfigBox type"""
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories"""
    for path in path_to_directories:
        # Convert to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory at: {path}")

# Keep ensure_annotations for the following functions because they are called with specific types.

@ensure_annotations
def save_json(path: Path, data: dict):
    """Save json data"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load json files data"""
    with open(path) as f:
        content = json.load(f)
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary file"""
    joblib.dump(value=data, filename=path)
    print(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary data"""
    data = joblib.load(path)
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Get size in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"