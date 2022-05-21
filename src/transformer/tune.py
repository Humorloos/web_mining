"""Script for hyperparameter optimization"""

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from constants import TRANSFORMER_DIR, MAX_BATCH_SIZE, VAL_CHECK_INTERVAL, MAX_EPOCHS, WORKERS_PER_TRIAL
from trainClassifier import train_classifier
from utils import get_timestamp

RAY_RESULTS_DIR = TRANSFORMER_DIR / 'ray_results'

NUM_SAMPLES = 50
RUN_NAME = get_timestamp()
# RESUME = 'LOCAL'  # 'LOCAL' resumes at last checkpoint, False starts new trial
RESUME = False  # 'LOCAL' resumes at last checkpoint, False starts new trial
# if set to a run directory, restores search algorithm state from that run, otherwise initiates new search algorithm
SEARCH_RESTORE_DIR = RAY_RESULTS_DIR / '2022-05-21_00.24'

search_config = {
    'batch_size_train': tune.qloguniform(2, MAX_BATCH_SIZE, q=1),
    'lr': tune.loguniform(1e-6, 1e-1),
    'weight_decay': tune.loguniform(1e-7, 1e-1),
    'dropout_prob': tune.uniform(0.1, 0.5),
}

static_config = {
    'fine_tune': 'adapter',
    'data_source': 'premade',
    'num_workers': WORKERS_PER_TRIAL,
    'optimizer': torch.optim.AdamW,
}

# Reporter for reporting progress in command line
reporter = CLIReporter(
    parameter_columns=["batch_size_train", "lr", "weight_decay", "dropout_prob"],
    metric_columns=["loss", "accuracy", "training_iteration"])

# BOHB search algorithm for finding new hyperparameter configurations
search_alg = TuneBOHB()
if SEARCH_RESTORE_DIR is not None:
    search_alg.restore_from_dir(SEARCH_RESTORE_DIR)
search_alg.set_search_properties(metric='loss', mode='min', config=search_config)

# BOHB scheduler for scheduling and discarding trials
iterations_per_epoch = 1 / VAL_CHECK_INTERVAL
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    # train for at most the number of iterations that fit into the max number of epochs
    max_t=MAX_EPOCHS * iterations_per_epoch
)


def get_trial_name(trial):
    """Function for generating trial names"""
    return f"{get_timestamp()}_{trial.trial_id}"


# run hyperparameter optimization
analysis = tune.run(
    tune.with_parameters(
        train_classifier,
        do_tune=True
    ),
    metric="loss",
    mode="min",
    config=static_config,
    num_samples=NUM_SAMPLES,
    scheduler=scheduler,
    name=RUN_NAME,
    local_dir=RAY_RESULTS_DIR,
    trial_name_creator=get_trial_name,
    trial_dirname_creator=get_trial_name,
    resume=RESUME,
    resources_per_trial={
        'gpu': 1,
        'cpu': WORKERS_PER_TRIAL
    },
    search_alg=search_alg,
    progress_reporter=reporter,
)
