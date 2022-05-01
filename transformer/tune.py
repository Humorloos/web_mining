"""Script for hyperparameter optimization"""

from datetime import timedelta, datetime, timezone

import pandas as pd
import torch
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from constants import DATA_DIR
from transformer.trainClassifier import train_classifier

RAY_RESULTS_DIR = DATA_DIR / 'transformer' / 'ray_results'

local_timezone = datetime.now(timezone(timedelta(0))).astimezone().tzinfo
start_timestamp = pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M')
MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
PATIENCE = 3  # number of consecutive epochs with validation loss < MIN_DELTA after which to stop early
MAX_TIME = timedelta(hours=3)
NUM_SAMPLES = 50
RUN_NAME = "test"
# RESUME = 'LOCAL'  # 'LOCAL' resumes at last checkpoint, False starts new trial
RESUME = False  # 'LOCAL' resumes at last checkpoint, False starts new trial

config = {
    'batch_size_train': 16,
    # 'num_workers': 8,
    'num_workers': 1,
    'optimizer': torch.optim.AdamW,
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-7, 1e-1),
    'dropout_prob': tune.uniform(0.1, 0.5),
}

bohb = TuneBOHB(metric='loss', mode='min')

analysis = tune.run(
    tune.with_parameters(
        train_classifier,
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        max_time=MAX_TIME,
    ),
    metric="loss",
    mode="min",
    config=config,
    num_samples=NUM_SAMPLES,
    scheduler=HyperBandForBOHB(),
    name=RUN_NAME,
    local_dir=RAY_RESULTS_DIR,
    trial_name_creator=lambda x: pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M'),
    trial_dirname_creator=lambda x: pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M'),
    resume=RESUME,
    resources_per_trial={
        'gpu': 0,
        # 'gpu': torch.cuda.device_count(),
        'cpu': 1
        # 'cpu': os.cpu_count()
    },
    search_alg=bohb,
)
