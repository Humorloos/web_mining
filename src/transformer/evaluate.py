from pytorch_lightning import Trainer

from constants import DATA_DIR
from datasets.TestDataLoader import get_test_dataloader
from emoBert import EmoBERT

TRIAL_DIR_NAME = '2022-05-06_12.13_255b1c85'  # name of model's ray-tune trial
CHECKPOINT_DIR_NAME = 'checkpoint_epoch=1-step=4'  # name of model's checkpoint
DATA_SOURCE = 'premade'  # dataset to use for evaluation (one of 'sst2', 'original', or 'premade')

if __name__ == '__main__':
    model = EmoBERT.load_from_checkpoint(
        DATA_DIR / 'transformer' / 'ray_results' / 'test' / TRIAL_DIR_NAME / CHECKPOINT_DIR_NAME / 'checkpoint')

    trainer = Trainer()
    trainer.test(model, dataloaders=get_test_dataloader(model=model, source=DATA_SOURCE))
