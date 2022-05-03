import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.constants.constants import TRANSFORMER_DIR
from src.transformer.emoBert import EmoBERT


def train_classifier(config, max_time, min_delta, patience, checkpoint_dir=None, do_tune=False):
    # initialize model
    if checkpoint_dir:
        model = EmoBERT.load_from_checkpoint(checkpoint_dir / "checkpoint")
    else:
        model = EmoBERT(config=config)

    save_dir = TRANSFORMER_DIR / 'trials'

    # callbacks
    callbacks = [EarlyStopping(
        monitor='val_loss',
        min_delta=min_delta,
        patience=patience,
        verbose=True)]
    if do_tune:
        from ray import tune
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        callbacks.append(TuneReportCheckpointCallback(
            # todo: add validation accuracy
            {'loss': 'ptl/val_loss'},
            on='validation_end'))
        save_dir = tune.get_trial_dir()

    # train model
    trainer = pl.Trainer(
        logger=WandbLogger(save_dir=save_dir, project="web_mining"),
        callbacks=callbacks,
        # gpus=torch.cuda.device_count(),
        gpus=0,
        max_time=max_time,
        # todo: adjust this so that validation is triggered about once every 10? minutes
        val_check_interval=1.0,
    )
    trainer.fit(model)
