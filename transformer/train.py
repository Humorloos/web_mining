"""Script for testing model training"""
from datetime import timedelta

import torch

from transformer.trainClassifier import train_classifier

MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
PATIENCE = 3  # number of consecutive epochs with validation loss < MIN_DELTA after which to stop early
MAX_TIME = timedelta(hours=3)

config = {
    'batch_size_train': 16,
    # 'batch_size_train': 16,
    # 'num_workers': 8,
    'num_workers': 1,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

if __name__ == '__main__':
    train_classifier(config, max_time=MAX_TIME, min_delta=MIN_DELTA, patience=PATIENCE)
