import pandas as pd
from torch.utils.data import DataLoader

from constants import MAX_BATCH_SIZE, MAX_WORKERS, DATA_DIR
from datasets.EmoBertDataset import EmoBertDataset
from datasets.EmoticonDataset import EmoticonDataset
from datasets.SST2TestSet import get_sst2_test_set
from emoBert import EmoBERT


def get_test_dataloader(model: EmoBERT, source: str):
    """
    Gets the test dataloader for evaluating the transformer model on a specific dataset
    :param model: EmoBERT model to get the dataloader for
    :param source: test set to get the data loader for, must be one of 'original' for the author's test set, 'sst2' for
    the SST-2 benchmark test set, or 'premade' for our own crawled test set
    :return: Dataloader loading data from the provided source
    """
    if source == 'sst2':
        test_set = get_sst2_test_set()
    elif source == 'original':
        test_set = EmoticonDataset('test')
    elif source == 'premade':
        test_set = EmoBertDataset(data=pd.read_csv(
            DATA_DIR / 'premade_datasets' / 'evaluation_dataset_non_lemmatized.csv'
        )[['prep_text', 'sentiment']].rename(columns={'prep_text': 'text', 'sentiment': 'polarity'}))
    else:
        raise ValueError("parameter 'source' must be one of 'original', 'sst2', or 'premade'")
    return DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=min(MAX_BATCH_SIZE, test_set.data.shape[0]),
        collate_fn=model.custom_collate,
        num_workers=MAX_WORKERS,
        pin_memory=True
    )
