from math import ceil
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
TRANSFORMER_DIR = DATA_DIR / 'transformer'

# transformer training config
# todo: set this according to dataset size
COMPLETE_DATA_SIZE = 2169661  # size of training + validation set (before split) (2169661 for full 'premade' set)
VAL_SET_SIZE = 1000  # size of validation set
TRAIN_SET_SIZE = COMPLETE_DATA_SIZE - VAL_SET_SIZE  # size of training set
# todo: figure out best MAX_EPOCHS value (e.g., by training one example model until convergence and looking how many
#  EPOCHS that took)
MAX_EPOCHS = 2
PATIENCE = 3  # number of consecutive checks with validation loss < MIN_DELTA after which to stop early
MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
# todo: figure out how large the batches may be so that machine can still handle them
MAX_BATCH_SIZE = 128
# spend 10 times more time on training than on validating
# formula computes interval so that epoch is evenly split
VAL_CHECK_INTERVAL = 1 / (ceil(1 / (10 * VAL_SET_SIZE / TRAIN_SET_SIZE)))
# todo: set this depending on machine (e.g., torch.cuda.device_count())
MAX_GPUS = 1
# todo: set this depending on machine (e.g., os.cpu_count())
MAX_WORKERS = 8
ADAPTER_NAME = 'classification'
DEFAULT_CONFIG = {
    'fine_tune': 'adapter',
    'data_source': 'premade',  # one of 'original', 'crawl', or 'premade'
    'batch_size_train': MAX_BATCH_SIZE,
    'num_workers': MAX_WORKERS,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'dropout_prob': 0.5,
}
