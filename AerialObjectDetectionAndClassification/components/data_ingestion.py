# components/data_ingestion.py
import os
import zipfile
import sys
from AerialObjectDetectionAndClassification import logger, AerialException
from AerialObjectDetectionAndClassification.utils.main_utils import get_size
from pathlib import Path
from AerialObjectDetectionAndClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self, url: str, filename: Path):
        """
        Download file from the given URL and save it to the specified filename.
        """
        if not os.path.exists(filename):
            try:
                import gdown
                # Extract file ID from the URL
                file_id = url.split('/')[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix+file_id, str(filename))
                logger.info(f"Downloaded {filename} from {url}")
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                raise AerialException(e, sys)
        else:
            logger.info(f"File already exists of size: {get_size(Path(filename))}")

    def extract_zip_file(self, zip_file_path: Path, unzip_dir: Path):
        """
        Extracts the zip file into the data directory.
        """
        try:
            if not os.path.exists(unzip_dir):
                os.makedirs(unzip_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            logger.info(f"Extracted {zip_file_path} to {unzip_dir}")
        except Exception as e:
            logger.error(f"Error extracting {zip_file_path}: {str(e)}")
            raise AerialException(e, sys)

    def initiate_data_ingestion(self):
        """
        Initiate data ingestion for both classification and detection datasets.
        """
        logger.info("Starting data ingestion process...")
        
        try:
            # Download classification data
            logger.info("Starting classification data download...")
            self.download_file(
                url=self.config.classification_source_URL,
                filename=self.config.classification_local_data_file
            )

            # Download detection data
            logger.info("Starting detection data download...")
            self.download_file(
                url=self.config.detection_source_URL,
                filename=self.config.detection_local_data_file
            )

            # Extract classification data
            logger.info("Extracting classification data...")
            self.extract_zip_file(
                zip_file_path=self.config.classification_local_data_file,
                unzip_dir=self.config.classification_unzip_dir
            )

            # Extract detection data
            logger.info("Extracting detection data...")
            self.extract_zip_file(
                zip_file_path=self.config.detection_local_data_file,
                unzip_dir=self.config.detection_unzip_dir
            )

            logger.info("Data ingestion completed successfully.")

        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise AerialException(e, sys)