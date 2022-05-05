import os
from math import ceil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
TRANSFORMER_DIR = DATA_DIR / 'transformer'

# transformer training config
# todo: set this according to dataset size
COMPLETE_DATA_SIZE = 359  # size of training + validation set (before split)
VAL_SET_SIZE = 60  # size of validation set
TRAIN_SET_SIZE = COMPLETE_DATA_SIZE - VAL_SET_SIZE  # size of training set
# todo: figure out best MAX_EPOCHS value (e.g., by training one example model until convergence and looking how many
#  EPOCHS that took)
MAX_EPOCHS = 10
PATIENCE = 3  # number of consecutive checks with validation loss < MIN_DELTA after which to stop early
MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
# todo: figure out how large the batches may be so that machine can still handle them
MAX_BATCH_SIZE = 64
# spend 10 times more time on training than on validating
# formula computes interval so that epoch is evenly split
VAL_CHECK_INTERVAL = 1 / (ceil(1 / (10 * VAL_SET_SIZE / TRAIN_SET_SIZE)))
# todo: set this depending on machine (e.g., torch.cuda.device_count())
MAX_GPUS = 0
# todo: set this depending on machine (e.g., os.cpu_count())
MAX_WORKERS = 1
ADAPTER_NAME = 'classification'
