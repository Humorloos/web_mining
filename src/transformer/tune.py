"""Script for hyperparameter optimization"""
from datetime import timedelta, datetime, timezone

import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from constants import TRANSFORMER_DIR, MAX_BATCH_SIZE, VAL_CHECK_INTERVAL, MAX_EPOCHS, MAX_GPUS, MAX_WORKERS
from trainClassifier import train_classifier

RAY_RESULTS_DIR = TRANSFORMER_DIR / 'ray_results'

local_timezone = datetime.now(timezone(timedelta(0))).astimezone().tzinfo
start_timestamp = pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M')
NUM_SAMPLES = 50
RUN_NAME = "test"
# RESUME = 'LOCAL'  # 'LOCAL' resumes at last checkpoint, False starts new trial
RESUME = False  # 'LOCAL' resumes at last checkpoint, False starts new trial

config = {
    'fine_tune': 'adapter',
    'data_source': 'premade',
    'batch_size_train': tune.qloguniform(2, MAX_BATCH_SIZE, q=1),
    'num_workers': MAX_WORKERS,
    'optimizer': torch.optim.AdamW,
    'lr': tune.loguniform(1e-6, 1e-1),
    'weight_decay': tune.loguniform(1e-7, 1e-1),
    'dropout_prob': tune.uniform(0.1, 0.5),
}

# Reporter for reporting progress in command line
reporter = CLIReporter(
    parameter_columns=["batch_size_train", "lr", "weight_decay", "dropout_prob"],
    metric_columns=["loss", "accuracy", "training_iteration"])

# BOHB search algorithm for finding new hyperparameter configurations
search_alg = TuneBOHB(metric='loss', mode='min')

# BOHB scheduler for scheduling and discarding trials
iterations_per_epoch = 1 / VAL_CHECK_INTERVAL
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    # train for at most the number of iterations that fit into the max number of epochs
    max_t=MAX_EPOCHS * iterations_per_epoch
)


def get_trial_name(trial):
    """Function for generating trial names"""
    return f"{pd.Timestamp.today(tz=local_timezone).strftime('%Y-%m-%d_%H.%M')}_{trial.trial_id}"


# run hyperparameter optimization
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
    search_alg=search_alg,
    progress_reporter=reporter,
)
