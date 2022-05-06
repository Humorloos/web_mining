from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers.adapters.configuration import AdapterConfig

from src.constants.constants import TRANSFORMER_DIR, MAX_EPOCHS, PATIENCE, MIN_DELTA, VAL_CHECK_INTERVAL, MAX_GPUS, \
    ADAPTER_NAME
from src.transformer.emoBert import EmoBERT


def train_classifier(config, checkpoint_dir=None, do_tune=False, fine_tune=True):
    # initialize model
    if checkpoint_dir:
        # when using ray-tune and resuming training of a previously stopped model, load the model again from the
        # checkpoint provided by ray-tune
        model = EmoBERT.load_from_checkpoint(str(Path(checkpoint_dir) / "checkpoint"))
    else:
        model = EmoBERT(config=config)

    if not fine_tune:
        # freeze base model (for testing)
        model.base_model.freeze_model()
    elif fine_tune == 'adapter':
        # add adapter to base model
        adapter_config = AdapterConfig.load(
            config='pfeiffer',
            non_linearity='relu',
        )
        model.base_model.add_adapter(ADAPTER_NAME, config=adapter_config)
        model.base_model.train_adapter(ADAPTER_NAME)

    save_dir = TRANSFORMER_DIR / 'trials'

    # callbacks
    callbacks = [EarlyStopping(
        monitor='ptl/val_loss',
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        verbose=True,
        # run check in each validation, not after training epoch
        check_on_train_epoch_end=False)]
    if do_tune:
        from ray import tune
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        callbacks.append(TuneReportCheckpointCallback(
            {'loss': 'ptl/val_loss', 'accuracy': 'ptl/val_accuracy'},
            on='validation_end'))
        save_dir = tune.get_trial_dir()

    # train model
    trainer = pl.Trainer(
        logger=WandbLogger(save_dir=save_dir, project="web_mining"),
        callbacks=callbacks,
        gpus=MAX_GPUS,
        max_epochs=MAX_EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
    )
    trainer.fit(model)
