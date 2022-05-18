"""Script for testing model training"""

from src.constants.constants import DEFAULT_CONFIG
from src.transformer.trainClassifier import train_classifier

if __name__ == '__main__':
    train_classifier(DEFAULT_CONFIG, fine_tune='adapter')
