# pipeline/training_pipeline.py
from AerialObjectDetectionAndClassification.configuration.config import ConfigurationManager
from AerialObjectDetectionAndClassification.components.data_ingestion import DataIngestion
from AerialObjectDetectionAndClassification.components.data_validation import DataValidation
from AerialObjectDetectionAndClassification.components.model_trainer import ClassificationModelTrainer

from AerialObjectDetectionAndClassification.components.detection_model_trainer import DetectionModelTrainer

from AerialObjectDetectionAndClassification import logger

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_data_ingestion(self):
        """
        Run data ingestion pipeline
        """
        try:
            logger.info("Starting data ingestion pipeline...")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion pipeline completed successfully!")
            return True
        except Exception as e:
            logger.exception(f"Data ingestion pipeline failed: {e}")
            return False

    def run_data_validation(self):
        """
        Run data validation pipeline
        """
        try:
            logger.info("Starting data validation pipeline...")
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            is_valid = data_validation.initiate_data_validation()
            logger.info(f"Data validation completed. Status: {'Valid' if is_valid else 'Invalid'}")
            return is_valid
        except Exception as e:
            logger.exception(f"Data validation pipeline failed: {e}")
            return False
        
    def run_model_training(self):
        """
        Run model training pipeline
        """
        try:
            logger.info("Starting model training pipeline...")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ClassificationModelTrainer(config=model_trainer_config)
            model_trainer.initiate_classification_training()
            logger.info("Model training pipeline completed successfully!")
            return True
        except Exception as e:
            logger.exception(f"Model training pipeline failed: {e}")
            return False

    def run_detection_training(self):
        """Run detection training pipeline"""
        try:
            logger.info("Starting detection training pipeline...")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = DetectionModelTrainer(config=model_trainer_config)
            results = model_trainer.initiate_detection_training()
            logger.info("Detection training pipeline completed successfully!")
            return results
        except Exception as e:
            logger.exception(f"Detection training pipeline failed: {e}")
            return None            

    def run_pipeline(self):
        """
        Run complete training pipeline
        """
        try:
            # Step 1: Data Ingestion
            if not self.run_data_ingestion():
                raise Exception("Data ingestion failed")
            
            # Step 2: Data Validation
            if not self.run_data_validation():
                raise Exception("Data validation failed")
            
            logger.info("Training pipeline completed successfully!")

            # Step 3: Model Training
            if not self.run_model_training():
                raise Exception("Classification Model training failed")
            
            logger.info("Classification Training pipeline completed successfully!")

            # Step 4: Detection Model Training
            if not self.run_detection_training():
                raise Exception("Detection Model training failed")
            
            logger.info("Detection Training pipeline completed successfully!")
            
            
        except Exception as e:
            logger.exception(f"Training pipeline failed: {e}")
            raise e