# components/data_validation.py
import os
import json
import sys
import datetime
from pathlib import Path
from AerialObjectDetectionAndClassification import logger, AerialException
from AerialObjectDetectionAndClassification.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig, schema):
        self.config = config
        self.schema = schema

    def validate_classification_dataset(self) -> dict:
        """
        Validate the classification dataset structure and contents using schema
        """
        logger.info("Validating classification dataset against schema...")
        validation_report = {}
        
        classification_path = Path(self.config.classification_data_dir)
        
        # Get schema requirements
        required_splits = self.schema.CLASSIFICATION_DATA.required_splits
        required_classes = self.schema.CLASSIFICATION_DATA.required_classes
        image_extensions = self.schema.CLASSIFICATION_DATA.image_extensions
        
        for split in required_splits:
            split_path = classification_path / split
            validation_report[split] = {}
            
            if not split_path.exists():
                validation_report[split]['exists'] = False
                validation_report[split]['error'] = f"Split directory {split} not found (required by schema)"
                continue
            
            validation_report[split]['exists'] = True
            validation_report[split]['classes'] = {}
            
            for class_name in required_classes:
                class_path = split_path / class_name
                validation_report[split]['classes'][class_name] = {}
                
                if not class_path.exists():
                    validation_report[split]['classes'][class_name]['exists'] = False
                    validation_report[split]['classes'][class_name]['error'] = f"Class directory {class_name} not found (required by schema)"
                    continue
                
                validation_report[split]['classes'][class_name]['exists'] = True
                
                # Count images using extensions from schema
                images = []
                for ext in image_extensions:
                    images.extend(list(class_path.glob(f"*{ext}")))
                
                validation_report[split]['classes'][class_name]['image_count'] = len(images)
                validation_report[split]['classes'][class_name]['images_found'] = len(images) > 0
                
                if len(images) == 0:
                    validation_report[split]['classes'][class_name]['warning'] = f"No images found in {class_name}"
        
        return validation_report

    def validate_detection_dataset(self) -> dict:
        """
        Validate the object detection dataset structure and contents using schema
        """
        logger.info("Validating detection dataset against schema...")
        validation_report = {}
        
        detection_path = Path(self.config.detection_data_dir)
        
        # Get schema requirements
        required_dirs = self.schema.DETECTION_DATA.required_directories
        required_files = self.schema.DETECTION_DATA.required_files
        expected_classes = self.schema.DETECTION_DATA.expected_classes
        
        # Check required directories
        for dir_path in required_dirs:
            full_path = detection_path / dir_path
            validation_report[dir_path] = {
                'exists': full_path.exists(),
                'file_count': len(list(full_path.glob('*'))) if full_path.exists() else 0
            }
        
        # Check required files
        for file_name in required_files:
            file_path = detection_path / file_name
            validation_report[file_name] = {
                'exists': file_path.exists(),
                'path': str(file_path)
            }
        
        # Validate data.yaml if it exists
        data_yaml_path = detection_path / 'data.yaml'
        if data_yaml_path.exists():
            try:
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data_yaml_content = yaml.safe_load(f)
                
                validation_report['data.yaml']['content'] = data_yaml_content
                validation_report['data.yaml']['valid'] = True
                
                # Validate classes in data.yaml against schema
                data_yaml_classes = data_yaml_content.get('names', [])
                if set(data_yaml_classes) == set(expected_classes):
                    validation_report['data.yaml']['classes_match'] = True
                else:
                    validation_report['data.yaml']['classes_match'] = False
                    validation_report['data.yaml']['expected_classes'] = expected_classes
                    validation_report['data.yaml']['actual_classes'] = data_yaml_classes
                    
            except Exception as e:
                validation_report['data.yaml']['valid'] = False
                validation_report['data.yaml']['error'] = str(e)
        
        return validation_report

    def is_dataset_valid(self, classification_report: dict, detection_report: dict) -> bool:
        """
        Check if both datasets meet the schema requirements
        """
        # Check classification dataset against schema
        classification_valid = True
        required_splits = self.schema.CLASSIFICATION_DATA.required_splits
        required_classes = self.schema.CLASSIFICATION_DATA.required_classes
        
        for split in required_splits:
            if split not in classification_report or not classification_report[split].get('exists', False):
                classification_valid = False
                break
            
            for class_name in required_classes:
                class_info = classification_report[split]['classes'].get(class_name, {})
                if not class_info.get('exists', False) or not class_info.get('images_found', False):
                    classification_valid = False
                    break
        
        # Check detection dataset against schema
        detection_valid = True
        required_dirs = self.schema.DETECTION_DATA.required_directories
        required_files = self.schema.DETECTION_DATA.required_files
        
        for dir_path in required_dirs:
            if not detection_report.get(dir_path, {}).get('exists', False):
                detection_valid = False
                break
        
        for file_name in required_files:
            if not detection_report.get(file_name, {}).get('exists', False):
                detection_valid = False
                break
        
        # Additional check: data.yaml classes should match schema
        if detection_report.get('data.yaml', {}).get('classes_match', False) == False:
            detection_valid = False
        
        return classification_valid and detection_valid

    def initiate_data_validation(self) -> bool:
        """
        Initiate complete data validation for both datasets using schema
        """
        logger.info("Starting schema-based data validation process...")
        
        try:
            # Validate both datasets against schema
            classification_report = self.validate_classification_dataset()
            detection_report = self.validate_detection_dataset()
            
            # Check if datasets are valid according to schema
            is_valid = self.is_dataset_valid(classification_report, detection_report)
            
            # Save validation reports
            validation_result = {
                'timestamp': str(datetime.datetime.now()),
                'is_valid': is_valid,
                'schema_used': {
                    'classification_requirements': {
                        'splits': self.schema.CLASSIFICATION_DATA.required_splits,
                        'classes': self.schema.CLASSIFICATION_DATA.required_classes
                    },
                    'detection_requirements': {
                        'directories': self.schema.DETECTION_DATA.required_directories,
                        'files': self.schema.DETECTION_DATA.required_files,
                        'expected_classes': self.schema.DETECTION_DATA.expected_classes
                    }
                },
                'classification_report': classification_report,
                'detection_report': detection_report
            }
            
            # Ensure the directory exists
            os.makedirs(Path(self.config.validation_status_file).parent, exist_ok=True)
            
            with open(self.config.validation_status_file, 'w') as f:
                json.dump(validation_result, f, indent=4)
            
            logger.info(f"Schema-based data validation completed. Status: {'VALID' if is_valid else 'INVALID'}")
            
            if is_valid:
                logger.info("Both datasets meet the schema requirements.")
            else:
                logger.warning("One or both datasets do not meet schema requirements. Check the validation report for details.")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Schema-based data validation failed: {str(e)}")
            raise AerialException(e, sys)

    def get_validation_summary(self) -> dict:
        """
        Get a summary of the validation results
        """
        if not Path(self.config.validation_status_file).exists():
            return {"error": "Validation not yet performed"}
        
        with open(self.config.validation_status_file, 'r') as f:
            return json.load(f)