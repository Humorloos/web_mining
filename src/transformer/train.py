"""Script for testing model training"""
import os

from constants import DEFAULT_CONFIG
from trainClassifier import train_classifier

if __name__ == '__main__':
    from utils import get_idle_gpus
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(get_idle_gpus()[0])
    train_classifier(DEFAULT_CONFIG)
