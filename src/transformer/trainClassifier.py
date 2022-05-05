import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.constants.constants import TRANSFORMER_DIR, MAX_EPOCHS, PATIENCE, MIN_DELTA, VAL_CHECK_INTERVAL, MAX_GPUS
from src.transformer.emoBert import EmoBERT


def train_classifier(config, checkpoint_dir=None, do_tune=False):
    # initialize model
    if checkpoint_dir:
        model = EmoBERT.load_from_checkpoint(checkpoint_dir / "checkpoint")
    else:
        model = EmoBERT(config=config)

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
