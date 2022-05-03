"""Script for hyperparameter optimization"""
import os
from datetime import timedelta, datetime, timezone

import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from src.constants.constants import TRANSFORMER_DIR, MAX_BATCH_SIZE, VAL_CHECK_INTERVAL, MAX_EPOCHS
from src.transformer.trainClassifier import train_classifier

RAY_RESULTS_DIR = TRANSFORMER_DIR / 'ray_results'

local_timezone = datetime.now(timezone(timedelta(0))).astimezone().tzinfo
start_timestamp = pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M')
MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
# todo: set this depending on machine (e.g., os.cpu_count())
MAX_WORKERS = os.cpu_count()
# todo: set this depending on machine (e.g., torch.cuda.device_count())
MAX_GPUS = 0
NUM_SAMPLES = 50
RUN_NAME = "test"
# RESUME = 'LOCAL'  # 'LOCAL' resumes at last checkpoint, False starts new trial
RESUME = False  # 'LOCAL' resumes at last checkpoint, False starts new trial

config = {
    'batch_size_train': tune.qloguniform(8, MAX_BATCH_SIZE, q=1),
    'num_workers': MAX_WORKERS,
    'optimizer': torch.optim.AdamW,
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-7, 1e-1),
    'dropout_prob': tune.uniform(0.1, 0.5),
}

bohb = TuneBOHB(metric='loss', mode='min')

reporter = CLIReporter(
    parameter_columns=["batch_size_train", "lr", "weight_decay", "dropout_prob"],
    metric_columns=["loss", "mean_accuracy", "training_iteration"])

iterations_per_epoch = 1 / VAL_CHECK_INTERVAL
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    # train for at most the number of iterations that fit into the max number of epochs
    max_t=MAX_EPOCHS * iterations_per_epoch
)


def get_trial_name(trial):
    return f"{pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M')}_{trial.trial_id}"


analysis = tune.run(
    tune.with_parameters(
        train_classifier,
        do_tune=True
    ),
    metric="loss",
    mode="min",
    config=config,
    num_samples=NUM_SAMPLES,
    scheduler=scheduler,
    name=RUN_NAME,
    local_dir=RAY_RESULTS_DIR,
    trial_name_creator=get_trial_name,
    trial_dirname_creator=get_trial_name,
    resume=RESUME,
    resources_per_trial={
        'gpu': MAX_GPUS,
        'cpu': MAX_WORKERS
    },
    search_alg=bohb,
    progress_reporter=reporter,
)
