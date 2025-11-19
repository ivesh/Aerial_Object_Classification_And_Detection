from AerialObjectDetectionAndClassification.logger import logging
from AerialObjectDetectionAndClassification.exception import Exception_class
import sys

logging.info("Welcome to test part of logging and exception.")

try:
    a=7/0
except Exception as e:
    raise Exception_class(e,sys) from e
