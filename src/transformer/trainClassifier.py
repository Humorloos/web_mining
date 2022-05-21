import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from constants import TRANSFORMER_DIR, MAX_EPOCHS, VAL_CHECK_INTERVAL
from emoBert import EmoBERT
from utils import get_timestamp


def train_classifier(config, checkpoint_dir=None, do_tune=False):
    # initialize model
    if checkpoint_dir:
        logging.info(f'Loading model from checkpoint {checkpoint_dir}')
        # when using ray-tune and resuming training of a previously stopped model, load the model again from the
        # checkpoint provided by ray-tune
        model = EmoBERT.load_from_checkpoint(str(Path(checkpoint_dir) / "checkpoint"))
    else:
        logging.info('Instantiating new EmoBERT model')
        model = EmoBERT(config=config)

    save_dir = TRANSFORMER_DIR / 'trials'

    # callbacks
    callbacks = [
        # EarlyStopping(
        #     monitor='ptl/val_loss',
        #     min_delta=MIN_DELTA,
        #     patience=PATIENCE,
        #     verbose=True,
        #     # run check in each validation, not after training epoch
        #     check_on_train_epoch_end=False)
    ]
    if do_tune:
        from ray import tune
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        callbacks.append(TuneReportCheckpointCallback(
            {'loss': 'ptl/val_loss', 'accuracy': 'ptl/val_accuracy'},
            on='validation_end'))
        save_dir = tune.get_trial_dir()
        wandb_logger = WandbLogger(
            name=tune.get_trial_name(), save_dir=save_dir, id=tune.get_trial_id(), project="web_mining", log_model=True,
            config=config)
    else:
        wandb_logger = WandbLogger(
            name=get_timestamp(), save_dir=save_dir, project="web_mining", log_model=True, config=config)

    logging.info('Instantiating trainer')
    # train model
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        gpus=1,
        max_epochs=MAX_EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
    )
    logging.info('Starting model training')
    trainer.fit(model)
