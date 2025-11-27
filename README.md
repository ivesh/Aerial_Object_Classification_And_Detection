## 🎯 PROJECT OVERVIEW

# 🛸 Aerial Object Classification & Detection System

## 📋 Project Objective

This project implements a comprehensive **Computer Vision MLOps pipeline** for aerial object analysis, specifically designed to classify and detect **birds vs drones** in aerial imagery. The system combines:

- **Multi-model classification** (Custom CNN, ResNet50, MobileNet, EfficientNet)
- **YOLOv8 object detection** with bounding boxes
- **End-to-end MLOps pipeline** with data validation, model training, and deployment
- **Streamlit web application** for real-time inference

The solution addresses critical needs in **aerial surveillance, wildlife monitoring, and security applications** by providing accurate, real-time analysis of aerial objects.

## 📊 TECHNICAL ARCHITECTURE ANALYSIS

### **Current Project Structure & Design Patterns**
The project follows a sophisticated MLOps pipeline architecture with:

1. **Modular Component Design** - Clear separation of concerns
2. **Configuration Management** - Centralized config handling via YAML files
3. **Entity Pattern** - Dataclass-based configuration entities
4. **Exception Handling** - Custom exception hierarchy
5. **Logging Infrastructure** - Comprehensive logging with file and console handlers
6. **Pipeline Orchestration** - Sequential execution of ML workflows


### **Phase 1: Data Ingestion ✅ COMPLETED**
- **Implementation**: `DataIngestion` class with Google Drive integration
- **Features**:
  - Dual dataset handling (classification + detection)
  - Download from Google Drive with gdown
  - ZIP extraction with error handling
  - Progress logging and size validation
- **Status**: Tested and functional

### **Phase 2: Data Validation ✅ COMPLETED**
- **Implementation**: Schema-driven validation system
- **Features**:
  - **Classification Validation**:
    - Directory structure validation
    - Class presence verification (bird/drone)
    - Image count validation per split
    - File extension validation
  - **Detection Validation**:
    - YOLOv8 directory structure
    - Annotation file validation
    - data.yaml configuration validation
    - Class matching verification
- **Status**: Comprehensive schema-based validation implemented

### **Phase 3: Model Training (Classification) ✅ PARTIALLY COMPLETED**

#### **Classification Training - IMPLEMENTED & TESTED**
- **Architectures Implemented**:
  1. **Custom CNN** - Lightweight sequential architecture
  2. **Transfer Learning Models**:
     - ResNet50
     - MobileNetV2
     - EfficientNetB0

- **Training Infrastructure**:
  - PyTorch-based training pipeline
  - GPU/CPU device detection
  - Data augmentation pipeline
  - Progressive training with validation
  - Model selection based on test accuracy
  - Training history tracking

- **Performance Results** (from your training logs):
  - **Custom CNN**: 80.93% test accuracy
  - **ResNet50**: 88.84% test accuracy  
  - **MobileNet**: 95.81% test accuracy
  - **EfficientNet**: 97.21% test accuracy ⭐ **BEST PERFORMER**

- **Key Features**:
  - Automatic best model selection
  - JSON serialization fixes implemented
  - Comprehensive logging
  - Model persistence



## 🛠️ Project Dependencies

### Core Technologies
- **Python 3.8+** - Primary programming language
- **PyTorch** - Deep learning framework
- **YOLOv8** - Object detection model
- **Streamlit** - Web application framework
- **OpenCV** - Computer vision operations

### Key Libraries
```yaml
torch: "Model training and inference"
torchvision: "Computer vision models and transforms"
ultralytics: "YOLOv8 implementation"
streamlit: "Web application framework"
opencv-python: "Image processing"
Pillow: "Image manipulation"
numpy: "Numerical operations"
pandas: "Data handling"
```

### MLOps Tools
- **DVC** - Data version control
- **MLflow** - Experiment tracking
- **Docker** - Containerization
- **GCP** - Cloud deployment (future)

## 📁 File Structure

```
AerialObjectDetectionAndClassification/
├── 📁 artifacts/                          # Generated artifacts
│   ├── 📁 data_ingestion/                # Downloaded datasets
│   ├── 📁 data_validation/               # Validation reports
│   └── 📁 model_trainer/                 # Trained models & results
├── 📁 AerialObjectDetectionAndClassification/  # Source code
│   ├── 📁 components/                    # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── 📁 configuration/                 # Configuration management
│   │   └── config.py
│   ├── 📁 constants/                     # Project constants
│   │   └── application.py
│   ├── 📁 entity/                        # Data classes
│   │   └── config_entity.py
│   ├── 📁 pipeline/                      # Training pipelines
│   │   └── training_pipeline.py
│   ├── 📁 utils/                         # Utility functions
│   │   └── main_utils.py
│   ├── exception.py                      # Custom exceptions
│   └── logger.py                         # Logging configuration
├── 📁 research/                          # Experimental notebooks
│   ├── data_ingestion_test.ipynb
│   ├── data_validation_schema_test.ipynb
│   └── model_training_test.ipynb
├── 📁 data/                              # Dataset directory
├── 📁 logs/                              # Application logs
├── 📄 config.yaml                        # Main configuration
├── 📄 params.yaml                        # Hyperparameters
├── 📄 schema.yaml                        # Data schema
├── 📄 app.py                            # Streamlit application
├── 📄 requirements.txt                   # Python dependencies
├── 📄 setup.py                          # Package installation
└── 📄 template.py                       # Project structure generator
```

## 🔄 Workflows

### Step 1: Setup & Initialization

**template.py**
- Auto-generates complete folder structure
- Uses `os` + `logging` for structured creation
- First file executed to set up repository structure

**setup.py**
```python
from setuptools import setup, find_packages

setup(
    name="AerialObjectDetectionAndClassification",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[]  # Populated from requirements.txt
)
```
- Installs `src/` as local package (`-e .`)
- Required for imports: `from AerialObjectDetectionAndClassification.pipeline import training_pipeline`

### Step 2: Configuration & Parameters

**config.yaml**
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  classification_source_URL: "https://drive.google.com/..."
  detection_source_URL: "https://drive.google.com/..."
  classification_local_data_file: artifacts/data_ingestion/classification_data.zip
  detection_local_data_file: artifacts/data_ingestion/detection_data.zip
  classification_unzip_dir: artifacts/data_ingestion
  detection_unzip_dir: artifacts/data_ingestion

data_validation:
  root_dir: artifacts/data_validation
  classification_data_dir: artifacts/data_ingestion/classification_dataset
  detection_data_dir: artifacts/data_ingestion/object_detection_Dataset
  validation_status_file: artifacts/data_validation/status.txt

model_trainer:
  root_dir: artifacts/model_trainer
  classification_trained_model_path: artifacts/model_trainer/classification_model.h5
  detection_trained_model_path: artifacts/model_trainer/detection_model.pt
  classification_training_data: artifacts/data_ingestion/classification_dataset
  detection_training_data: artifacts/data_ingestion/object_detection_Dataset
```

**params.yaml**
```yaml
MODEL_TRAINER:
  classification:
    epochs: 10
    batch_size: 32
    image_size: [224, 224]
    learning_rate: 0.001
    include_top: false
    weights: imagenet
  detection:
    epochs: 50
    batch_size: 16
    image_size: 640
    pretrained_weights: yolov8n.pt
```

**schema.yaml**
```yaml
CLASSIFICATION_DATA:
  required_classes: [bird, drone]
  required_splits: [train, valid, test]
  image_extensions: [.jpg, .jpeg, .png, .bmp]

DETECTION_DATA:
  required_directories: [train/images, train/labels, valid/images, valid/labels, test/images, test/labels]
  required_files: [data.yaml]
  expected_classes: [Bird, drone]
```

**src/AerialObjectDetectionAndClassification/constants**
- Defines constant paths and configuration values
- Centralized location for all project constants

**src/AerialObjectDetectionAndClassification/entity**
```python
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    classification_source_URL: str
    detection_source_URL: str
    classification_local_data_file: Path
    detection_local_data_file: Path
    classification_unzip_dir: Path
    detection_unzip_dir: Path
```
- Contains dataclasses to strongly type configuration sections
- Ensures type safety and clear configuration structure

### Step 3: Utilities & Configuration Management

**src/AerialObjectDetectionAndClassification/utils/main_utils.py**
- File handling utilities
- YAML reading/writing functions
- Directory creation helpers

**src/AerialObjectDetectionAndClassification/configuration/config.py**
```python
class ConfigurationManager:
    def __init__(self, config_filepath, params_filepath, schema_filepath):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            classification_source_URL=config.classification_source_URL,
            # ... other parameters
        )
```
- Centralized configuration management
- Type-safe configuration retrieval

### Step 4: Pipeline Components

#### Data Ingestion (`components/data_ingestion.py`)
```python
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self, url: str, filename: Path):
        """Download datasets from Google Drive"""
        # Uses gdown for Google Drive downloads
    
    def extract_zip_file(self, zip_file_path: Path, unzip_dir: Path):
        """Extract downloaded datasets"""
    
    def initiate_data_ingestion(self):
        """Orchestrate complete data ingestion process"""
```

#### Data Validation (`components/data_validation.py`)
```python
class DataValidation:
    def __init__(self, config: DataValidationConfig, schema):
        self.config = config
        self.schema = schema
    
    def validate_classification_dataset(self) -> dict:
        """Validate classification data against schema"""
    
    def validate_detection_dataset(self) -> dict:
        """Validate detection data against schema"""
    
    def initiate_data_validation(self) -> bool:
        """Complete schema-based validation"""
```

#### Model Trainer (`components/model_trainer.py`)
```python
class ClassificationModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_custom_cnn(self):
        """Build custom CNN architecture"""
    
    def build_transfer_learning_model(self, model_name='resnet50'):
        """Build transfer learning models"""
    
    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs):
        """Training loop with validation"""
    
    def initiate_classification_training(self):
        """Train multiple models and select best performer"""
```

### Step 5: Training Pipeline

**src/AerialObjectDetectionAndClassification/pipeline/training_pipeline.py**
```python
class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
    
    def run_data_ingestion(self):
        """Execute data ingestion pipeline"""
    
    def run_data_validation(self):
        """Execute data validation pipeline"""
    
    def run_model_training(self):
        """Execute model training pipeline"""
    
    def run_pipeline(self):
        """Complete training pipeline orchestration"""
        self.run_data_ingestion()
        self.run_data_validation() 
        self.run_model_training()
```

### Step 6: Streamlit Application

**app.py**
```python
class ClassificationModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def predict(self, image):
        """Classify image as bird or drone"""

class DetectionModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
    
    def predict(self, image):
        """Detect objects with bounding boxes"""
```

### Step 7: DVC Pipeline (Future)

**dvc.yaml**
```yaml
stages:
  data_ingestion:
    cmd: python -c "from AerialObjectDetectionAndClassification.pipeline.training_pipeline import TrainingPipeline; tp = TrainingPipeline(); tp.run_data_ingestion()"
    deps: [data/]
    outs: [artifacts/data_ingestion/]
  
  data_validation:
    cmd: python -c "from AerialObjectDetectionAndClassification.pipeline.training_pipeline import TrainingPipeline; tp = TrainingPipeline(); tp.run_data_validation()"
    deps: [artifacts/data_ingestion/]
    outs: [artifacts/data_validation/status.txt]
  
  model_training:
    cmd: python -c "from AerialObjectDetectionAndClassification.pipeline.training_pipeline import TrainingPipeline; tp = TrainingPipeline(); tp.run_model_training()"
    deps: [artifacts/data_validation/status.txt]
    outs: [artifacts/model_trainer/]
```

## 🔧 Components Deep Dive

### Step 1: Data Ingestion Component
- **Downloads datasets** from Google Drive using `gdown`
- **Extracts zip files** to structured directories
- **Handles both classification and detection** datasets separately
- **Error handling** for network issues and file corruption

### Step 2: Data Validation Component
- **Schema-based validation** against predefined requirements
- **Checks directory structure** for both datasets
- **Validates class distributions** and image formats
- **Generates comprehensive validation reports**
- **Ensures data quality** before model training

### Step 3: Model Training Component
- **Multi-architecture training**: Custom CNN, ResNet50, MobileNet, EfficientNet
- **Transfer learning** with fine-tuning
- **Automatic model selection** based on validation performance
- **Comprehensive logging** of training metrics
- **Model comparison and selection**

### Step 4: Streamlit Application
- **Real-time image classification** and object detection
- **Interactive user interface** with file upload
- **Confidence scoring** and visualization
- **Bounding box display** for object detection
- **Model performance information**

## 🚀 How to Run & Fetch Results

### Installation & Setup
```bash
# Clone the repository
git clone <repository-url>
cd AerialObjectDetectionAndClassification

# Install dependencies
pip install -r requirements.txt

# Install as local package
pip install -e .
```

### Running the Training Pipeline
```bash
# Method 1: Using the training pipeline
python -c "
from AerialObjectDetectionAndClassification.pipeline.training_pipeline import TrainingPipeline
tp = TrainingPipeline()
tp.run_pipeline()
"

# Method 2: Run individual components
python -c "
from AerialObjectDetectionAndClassification.pipeline.training_pipeline import TrainingPipeline
tp = TrainingPipeline()
tp.run_data_ingestion()      # Step 1: Download and extract data
tp.run_data_validation()     # Step 2: Validate data structure
tp.run_model_training()      # Step 3: Train models
"
```

### Running the Streamlit Application
```bash
streamlit run app.py
```
- Open browser to `http://localhost:8501`
- Use sidebar to navigate between Classification and Detection
- Upload images for real-time analysis

![alt text](<Screenshot 2025-11-27 151702.png>)
![alt text](<Screenshot 2025-11-27 190007.png>)
![alt text](<Screenshot 2025-11-27 190049.png>)
![alt text](<Screenshot 2025-11-27 151702.png>)

### Fetching Results
**Training Results:**
- Models saved in `artifacts/model_trainer/`
- Performance metrics in `classification_results_summary.json`
- Training history in `*_history.pkl` files

**Validation Reports:**
- Data validation status in `artifacts/data_validation/status.txt`
- Schema compliance reports

## 🔮 Future Work & Enhancements

### MLOps Integration
1. **MLflow Integration**
   ```python
   import mlflow
   
   with mlflow.start_run():
       mlflow.log_params(params)
       mlflow.log_metric("accuracy", test_accuracy)
       mlflow.pytorch.log_model(model, "model")
   ```
   - Experiment tracking and comparison
   - Model versioning and registry
   - Parameter and metric logging

2. **DVC Pipeline Enhancement**
   ```yaml
   stages:
     data_ingestion:
       cmd: python src/components/data_ingestion.py
       deps: [config.yaml, params.yaml]
       outs: [artifacts/data_ingestion]
       metrics: [artifacts/data_ingestion/metrics.json]
   ```
   - Reproducible pipeline execution
   - Data version control
   - Pipeline dependency management

3. **Docker Containerization**
   ```dockerfile
   FROM python:3.8-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```
   - Consistent development and production environments
   - Easy deployment to cloud platforms

4. **GCP Deployment Architecture**
   - **Cloud Storage**: Model and data storage
   - **AI Platform**: Model serving and training
   - **Cloud Run**: Streamlit application deployment
   - **Cloud Build**: CI/CD pipeline automation

### Technical Enhancements
1. **Advanced Model Architectures**
   - Vision Transformers (ViT) for classification
   - YOLOv9 for improved detection accuracy
   - Ensemble methods for boosted performance

2. **Real-time Video Processing**
   - Live video stream analysis
   - Object tracking across frames
   - Batch processing optimization

3. **Performance Optimization**
   - Model quantization for faster inference
   - GPU acceleration optimization
   - Caching mechanisms for repeated inferences

4. **Extended Functionality**
   - Multi-class classification (different bird species, drone types)
   - Temporal analysis for movement patterns
   - Alert system for security applications
   - API endpoints for integration with other systems

### Monitoring & Maintenance
1. **Model Performance Monitoring**
   - Drift detection for data and concept drift
   - Automated retraining pipelines
   - A/B testing for model updates

2. **Data Pipeline Enhancement**
   - Continuous data collection and annotation
   - Automated data quality checks
   - Active learning for model improvement



### **Minor Issues Addressed:**
- ✅ JSON serialization fixed (NumPy to Python type conversion)
- ✅ File path handling standardized with Pathlib
- ✅ Logging configuration stabilized

### **Recommended Enhancements:**
1. **Hyperparameter Optimization** - Grid search for optimal parameters
2. **Cross-Validation** - More robust model evaluation
3. **Data Imbalance Handling** - Address class distribution issues
4. **Model Interpretability** - Grad-CAM or SHAP explanations
5. **Performance Optimization** - Mixed precision training, distributed training

## 🎯 BUSINESS IMPACT ASSESSMENT

The current implementation already provides significant value:

1. **High Accuracy**: 97.21% classification accuracy meets production standards
2. **Multiple Model Options**: Flexibility in model selection based on deployment constraints
3. **Scalable Architecture**: Pipeline can handle larger datasets
4. **Maintainable Code**: Clear structure for team collaboration

## 📋 RECOMMENDED NEXT STEPS

1. **Immediate**: Implement YOLOv8 detection training
2. **Short-term**: Develop Streamlit deployment interface
3. **Medium-term**: Add comprehensive model evaluation metrics
4. **Long-term**: Implement CI/CD pipeline and cloud deployment

-------------------------

# Object Detection Process:
1. Data Collection: Data can be collected in 3 ways
    - Collect the ready made data available in the internet - (In this project we have already collected data which is annotated as well using roboflow)
    - Have a team of data annotators and send them to the field collect and annotate data
    - Web Scraping script to collect necessary images and annotate the images
2. Data Annotation: We can either annotate the data using labelImg or roboflow.  
    - Annotated the data using roboflow object detection. During data annotation step, we have to provide the images to annotate and in the next page options will be there to create version of the data where we can perform train test split using the images provided, preprocessing steps such as Auto Orient and Resize is available, and data augmentation. 
    - After the version of this data is ready we can train with the model using roboflow(paid feature), however we will be training this images here inside our project workflow.
    - Post theversion of the data is ready we can download it which will contain the images, labels and data.yaml(contains path of the image and number of classes) files.
3. Project Workflow: 
    - 1st Step: Updated custom logging under AerialObjectDetectionAndClassificaation->logger->__init__.py & updated exception under AerialObjectDetectionAndClassificaation->exception->__init__.py
    - 
    - Components: Main components of data ingestion, data validation, model pusher and model trainer components
