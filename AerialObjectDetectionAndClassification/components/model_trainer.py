# components/model_trainer.py - COMPLETELY FIXED VERSION
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path
from AerialObjectDetectionAndClassification import logger, AerialException
from AerialObjectDetectionAndClassification.entity.config_entity import ModelTrainerConfig
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import joblib
import sys
import json

class ClassificationModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def prepare_data(self):
        """Prepare data loaders for training, validation, and testing."""
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.config.classification_params_image_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(self.config.classification_params_image_size[0]),
                transforms.CenterCrop(self.config.classification_params_image_size[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.config.classification_params_image_size[0]),
                transforms.CenterCrop(self.config.classification_params_image_size[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create datasets
        image_datasets = {
            'train': datasets.ImageFolder(
                os.path.join(self.config.classification_training_data, 'train'),
                data_transforms['train']
            ),
            'valid': datasets.ImageFolder(
                os.path.join(self.config.classification_training_data, 'valid'),
                data_transforms['valid']
            ),
            'test': datasets.ImageFolder(
                os.path.join(self.config.classification_training_data, 'test'),
                data_transforms['test']
            )
        }

        # Create data loaders
        dataloaders = {
            'train': DataLoader(
                image_datasets['train'],
                batch_size=self.config.classification_params_batch_size,
                shuffle=True,
                num_workers=4
            ),
            'valid': DataLoader(
                image_datasets['valid'],
                batch_size=self.config.classification_params_batch_size,
                shuffle=False,
                num_workers=4
            ),
            'test': DataLoader(
                image_datasets['test'],
                batch_size=self.config.classification_params_batch_size,
                shuffle=False,
                num_workers=4
            )
        }

        return dataloaders, image_datasets

    def build_custom_cnn(self):
        """Build a custom CNN model."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # bird vs drone
        )
        return model

    def build_transfer_learning_model(self, model_name='resnet50'):
        """Build a transfer learning model."""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, 2)
        elif model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs):
        """Train the model and return the best model and history."""
        model = model.to(self.device)
        best_model_wts = model.state_dict()
        best_acc = 0.0
        # FIXED: Use 'valid' consistently
        history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # FIXED: Use 'valid' consistently
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                # FIXED: Use 'valid' consistently
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.cpu().numpy())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # FIXED: Use 'valid' consistently
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

            print()

        model.load_state_dict(best_model_wts)
        return model, history

    def evaluate_model(self, model, test_loader):
        """Evaluate the model on the test set."""
        model.eval()
        running_corrects = 0

        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / len(test_loader.dataset)
        test_acc_float = test_acc.cpu().item()  # Convert to Python float
        print(f'Test Accuracy: {test_acc_float:.4f}')
        return test_acc_float  
       

    def initiate_classification_training(self):
        """Initiate the classification model training."""
        try:
            logger.info("Starting classification model training...")

            # Prepare data
            dataloaders, image_datasets = self.prepare_data()
            logger.info("Data preparation completed.")

            # Models to train
            models_dict = {
                'custom_cnn': self.build_custom_cnn(),
                'resnet50': self.build_transfer_learning_model('resnet50'),
                'mobilenet': self.build_transfer_learning_model('mobilenet'),
                'efficientnet': self.build_transfer_learning_model('efficientnet')
            }

            # Training results
            results = {}

            for model_name, model in models_dict.items():
                logger.info(f"Training {model_name}...")
                print(f"Training {model_name}...")

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=self.config.classification_params_learning_rate)

                # Train the model
                model, history = self.train_model(
                    model, dataloaders, criterion, optimizer, 
                    num_epochs=self.config.classification_params_epochs
                )

                # Evaluate the model
                test_acc = self.evaluate_model(model, dataloaders['test'])

                # Save the model
                model_save_path = os.path.join(
                    self.config.root_dir, 
                    f"{model_name}_classification_model.pth"
                )
                torch.save(model.state_dict(), model_save_path)

                # Save history
                history_save_path = os.path.join(
                    self.config.root_dir,
                    f"{model_name}_history.pkl"
                )
                joblib.dump(history, history_save_path)

                results[model_name] = {
                    'test_accuracy': test_acc,
                    'model_path': model_save_path,
                    'history_path': history_save_path
                }

                logger.info(f"{model_name} training completed. Test Accuracy: {test_acc}")

            # Compare models and select the best one
            best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
            best_model_path = results[best_model_name]['model_path']

            # Save the best model as the final classification model
            final_model_path = self.config.classification_trained_model_path
            os.rename(best_model_path, final_model_path)

            logger.info(f"Best model is {best_model_name} with accuracy {results[best_model_name]['test_accuracy']}")
            logger.info(f"Final model saved at: {final_model_path}")

            # Save the results comparison
            results_summary = {
                'best_model': best_model_name,
                'best_accuracy': results[best_model_name]['test_accuracy'],
                'all_results': results
            }
            results_summary_path = os.path.join(self.config.root_dir, 'classification_results_summary.json')
            with open(results_summary_path, 'w') as f:
                json.dump(results_summary, f, indent=4)

            logger.info("Classification model training completed successfully.")

        except Exception as e:
            logger.exception(f"Classification model training failed: {e}")
            raise AerialException(e, sys)