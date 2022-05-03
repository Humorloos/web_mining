"""Script for testing model training"""

import torch

from src.transformer.trainClassifier import train_classifier

config = {
    'batch_size_train': 16,
    'num_workers': 1,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

if __name__ == '__main__':
    train_classifier(config)
