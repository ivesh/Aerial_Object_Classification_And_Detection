# components/detection_model_trainer.py - FIXED EVALUATION VERSION
import os
import sys
from pathlib import Path
from AerialObjectDetectionAndClassification import logger, AerialException
from AerialObjectDetectionAndClassification.entity.config_entity import ModelTrainerConfig
import yaml
import shutil
import requests
import numpy as np

class DetectionModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        logger.info(f"Initializing DetectionModelTrainer")

    def prepare_data_yaml(self):
        """Prepare data.yaml file for YOLOv8 training"""
        try:
            detection_data_path = Path(self.config.detection_training_data)
            data_yaml_path = detection_data_path / "data.yaml"
            
            if data_yaml_path.exists():
                logger.info(f"data.yaml already exists at: {data_yaml_path}")
                return data_yaml_path
            
            # Create data.yaml content
            data_config = {
                'path': str(detection_data_path),
                'train': 'train/images',
                'val': 'valid/images', 
                'test': 'test/images',
                'nc': 2,
                'names': ['Bird', 'drone']
            }
            
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            logger.info(f"Created data.yaml at: {data_yaml_path}")
            return data_yaml_path
            
        except Exception as e:
            logger.error(f"Error preparing data.yaml: {e}")
            raise AerialException(e, sys)

    def download_pretrained_weights(self):
        """Download pretrained weights using direct URL to get proper YOLO format"""
        try:
            weights_name = self.config.detection_params_pretrained_weights
            artifacts_weights_path = Path(self.config.root_dir) / weights_name
            
            if artifacts_weights_path.exists():
                logger.info(f"Pretrained weights already exist at: {artifacts_weights_path}")
                return str(artifacts_weights_path)
            
            # YOLOv8 pretrained weights URLs
            weights_urls = {
                'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
                'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
                'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
                'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
                'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt'
            }
            
            if weights_name not in weights_urls:
                raise AerialException(f"Unsupported weights: {weights_name}", sys)
            
            logger.info(f"Downloading {weights_name} from {weights_urls[weights_name]}...")
            
            # Download with progress
            response = requests.get(weights_urls[weights_name], stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(artifacts_weights_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
            
            logger.info(f"Pretrained weights downloaded to: {artifacts_weights_path}")
            return str(artifacts_weights_path)
            
        except Exception as e:
            logger.error(f"Error downloading pretrained weights: {e}")
            raise AerialException(e, sys)

    def train_yolov8_model(self):
        """Train YOLOv8 model using Ultralytics"""
        try:
            from ultralytics import YOLO
            
            # Prepare data.yaml
            data_yaml_path = self.prepare_data_yaml()
            
            # Download pretrained weights (proper YOLO format)
            artifacts_weights_path = self.download_pretrained_weights()
            
            # Load pretrained model - this should now work with proper checkpoint format
            model = YOLO(artifacts_weights_path)
            logger.info(f"Loaded pretrained weights from: {artifacts_weights_path}")
            
            # Windows-compatible training configuration
            train_args = {
                'data': str(data_yaml_path),
                'epochs': self.config.detection_params_epochs,
                'batch': self.config.detection_params_batch_size,
                'imgsz': self.config.detection_params_image_size,
                'project': str(self.config.root_dir),
                'name': 'yolov8_training',
                'exist_ok': True,
                'patience': 10,
                'save': True,
                'verbose': True,
                'workers': 0,  # CRITICAL: Set to 0 for Windows compatibility
                'deterministic': True,
                'cache': False,
            }
            
            # Train the model
            results = model.train(**train_args)
            
            # Get the best model path
            best_model_path = Path(self.config.root_dir) / 'yolov8_training' / 'weights' / 'best.pt'
            
            if not best_model_path.exists():
                # Try alternative path structure
                best_model_path = Path(self.config.root_dir) / 'yolov8_training' / 'best.pt'
            
            if not best_model_path.exists():
                # If still not found, look for any .pt file in the training directory
                training_dir = Path(self.config.root_dir) / 'yolov8_training'
                pt_files = list(training_dir.rglob('*.pt'))
                if pt_files:
                    best_model_path = pt_files[0]
                    logger.info(f"Using found model: {best_model_path}")
                else:
                    raise AerialException("No trained model found after training", sys)
            
            # Copy best model to the specified location
            final_model_path = Path(self.config.detection_trained_model_path)
            os.makedirs(final_model_path.parent, exist_ok=True)
            shutil.copy2(best_model_path, final_model_path)
            
            logger.info(f"YOLOv8 training completed. Best model saved to: {final_model_path}")
            
            return {
                'model_path': str(final_model_path),
                'metrics': results
            }
            
        except ImportError:
            logger.error("Ultralytics YOLO package not found. Please install with: pip install ultralytics")
            raise AerialException("Ultralytics YOLO package not found", sys)
        except Exception as e:
            logger.error(f"Error during YOLOv8 training: {e}")
            raise AerialException(e, sys)

    def evaluate_yolov8_model(self, model_path):
        """Evaluate the trained YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            # Load the trained model
            model = YOLO(model_path)
            
            # Evaluate on validation set
            detection_data_path = Path(self.config.detection_training_data)
            data_yaml_path = detection_data_path / "data.yaml"
            
            metrics = model.val(
                data=str(data_yaml_path),
                split='val',
                verbose=True,
                workers=0  # Windows compatibility
            )
            
            # FIX: Extract and convert metrics properly
            # The metrics might be numpy arrays, so we need to handle them correctly
            map50 = metrics.box.map50
            map_val = metrics.box.map
            precision = metrics.box.p
            recall = metrics.box.r
            
            # Convert to float if they are arrays
            if hasattr(map50, '__iter__'):
                map50 = float(map50.mean()) if len(map50) > 0 else 0.0
            if hasattr(map_val, '__iter__'):
                map_val = float(map_val.mean()) if len(map_val) > 0 else 0.0
            if hasattr(precision, '__iter__'):
                precision = float(precision.mean()) if len(precision) > 0 else 0.0
            if hasattr(recall, '__iter__'):
                recall = float(recall.mean()) if len(recall) > 0 else 0.0
            
            # Convert to Python float if they're numpy types
            map50 = float(map50)
            map_val = float(map_val)
            precision = float(precision)
            recall = float(recall)
            
            # Log key metrics
            logger.info(f"YOLOv8 Evaluation Metrics:")
            logger.info(f"  mAP50: {map50:.4f}")
            logger.info(f"  mAP50-95: {map_val:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            
            return {
                'map50': map50,
                'map': map_val,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            logger.error(f"Error during YOLOv8 evaluation: {e}")
            raise AerialException(e, sys)

    def initiate_detection_training(self):
        """Initiate the object detection model training pipeline"""
        try:
            logger.info("Starting object detection model training...")
            
            # Train YOLOv8 model
            training_results = self.train_yolov8_model()
            
            # Evaluate the model
            evaluation_results = self.evaluate_yolov8_model(training_results['model_path'])
            
            # Combine results
            detection_results = {
                'training': training_results,
                'evaluation': evaluation_results,
                'model_path': training_results['model_path']
            }
            
            # Save results summary
            results_summary_path = Path(self.config.root_dir) / 'detection_results_summary.yaml'
            with open(results_summary_path, 'w') as f:
                yaml.dump(detection_results, f, default_flow_style=False)
            
            logger.info(f"Detection training completed. Results saved to: {results_summary_path}")
            logger.info(f"Final model: {detection_results['model_path']}")
            
            return detection_results
            
        except Exception as e:
            logger.exception(f"Detection model training failed: {e}")
            raise AerialException(e, sys)