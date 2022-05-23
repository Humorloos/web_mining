import os

from pytorch_lightning import Trainer

from constants import DATA_DIR
from datasets.TestDataLoader import get_test_dataloader
from emoBert import EmoBERT

TUNE_RUN_DIR_NAME = '2022-05-21_19.18'
TRIAL_DIR_NAME = '2022-05-21_21.23_8d02f3ce'  # name of trial directory
CHECKPOINT_NAME = 'checkpoint_epoch=0-step=11388'  # name of model's checkpoint
DATA_SOURCE = 'reddit'  # dataset to use for evaluation (one of 'sst2', 'original', 'premade', or 'reddit')

if __name__ == '__main__':
    # from utils import get_gpu_with_most_available_memory
    # target_gpu = get_gpu_with_most_available_memory()
    target_gpu = '2'
    # with hpyerparameter-optimization:
    # set TRIAL_DIR_NAME to name of ray-tune directory and CHECKPOINT_NAME to name of checkpoint directory
    model = EmoBERT.load_from_checkpoint(
        DATA_DIR / 'transformer' / 'ray_results' / TUNE_RUN_DIR_NAME / TRIAL_DIR_NAME / CHECKPOINT_NAME / 'checkpoint')
    # without hyperparameter-optimization:
    # set TRIAL_DIR_NAME to training dir name and CHECKPOINT_NAME to name of checkpoint file
    # model = EmoBERT.load_from_checkpoint(
    #     DATA_DIR / 'transformer' / 'trials' / 'web_mining' / TRIAL_DIR_NAME / 'checkpoints' / CHECKPOINT_NAME)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = target_gpu
    trainer = Trainer(gpus=1)
    trainer.test(model, dataloaders=get_test_dataloader(model=model, source=DATA_SOURCE))
