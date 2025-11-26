# configuration/config.py
from AerialObjectDetectionAndClassification.constant.application import *
from AerialObjectDetectionAndClassification.utils.main_utils import read_yaml, create_directories
from AerialObjectDetectionAndClassification.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        # Convert string paths to Path objects
        config_filepath = Path(config_filepath)
        params_filepath = Path(params_filepath)
        schema_filepath = Path(schema_filepath) if schema_filepath else None

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath) if schema_filepath and schema_filepath.exists() else None

        create_directories([Path(self.config.artifacts_root)])
        
    def get_schema_config(self):
        """Get schema configuration for data validation"""
        return self.schema        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            classification_source_URL=config.classification_source_URL,
            detection_source_URL=config.detection_source_URL,
            classification_local_data_file=Path(config.classification_local_data_file),
            detection_local_data_file=Path(config.detection_local_data_file),
            classification_unzip_dir=Path(config.classification_unzip_dir),
            detection_unzip_dir=Path(config.detection_unzip_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([Path(config.root_dir)])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            classification_data_dir=Path(config.classification_data_dir),
            detection_data_dir=Path(config.detection_data_dir),
            validation_status_file=Path(config.validation_status_file)
        )

        return data_validation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.MODEL_TRAINER

        create_directories([Path(config.root_dir)])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            classification_trained_model_path=Path(config.classification_trained_model_path),
            detection_trained_model_path=Path(config.detection_trained_model_path),
            classification_training_data=Path(config.classification_training_data),
            detection_training_data=Path(config.detection_training_data),
            classification_params_epochs=params.classification.epochs,
            classification_params_batch_size=params.classification.batch_size,
            classification_params_image_size=params.classification.image_size,
            classification_params_learning_rate=params.classification.learning_rate,
            classification_params_include_top=params.classification.include_top,
            classification_params_weights=params.classification.weights,
            detection_params_epochs=params.detection.epochs,
            detection_params_batch_size=params.detection.batch_size,
            detection_params_image_size=params.detection.image_size,
            detection_params_pretrained_weights=params.detection.pretrained_weights
        )

        return model_trainer_config