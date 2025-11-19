# entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    classification_source_URL: str
    detection_source_URL: str
    classification_local_data_file: Path
    detection_local_data_file: Path
    classification_unzip_dir: Path
    detection_unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    classification_data_dir: Path
    detection_data_dir: Path
    validation_status_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    classification_trained_model_path: Path
    detection_trained_model_path: Path
    classification_training_data: Path
    detection_training_data: Path
    classification_params_epochs: int
    classification_params_batch_size: int
    classification_params_image_size: list
    classification_params_learning_rate: float
    classification_params_include_top: bool
    classification_params_weights: str
    detection_params_epochs: int
    detection_params_batch_size: int
    detection_params_image_size: int
    detection_params_pretrained_weights: str