import pandas as pd
from torch.utils.data import DataLoader

from constants import MAX_BATCH_SIZE, WORKERS_PER_TRIAL, DATA_DIR
from datasets.EmoBertDataset import EmoBertDataset
from datasets.EmoticonDataset import EmoticonDataset
from datasets.SST2TestSet import get_preprocessed_sst2_test_set
from emoBert import EmoBERT


def get_test_dataloader(model: EmoBERT, source: str):
    """
    Gets the test dataloader for evaluating the transformer model on a specific dataset
    :param model: EmoBERT model to get the dataloader for
    :param source: test set to get the data loader for, must be one of 'original' for the author's test set, 'sst2' for
    the SST-2 benchmark test set, 'premade' for our own crawled twitter test set, or 'reddit' for our crawled reddit
    test set
    :return: Dataloader loading data from the provided source
    """
    if source == 'sst2':
        test_set = get_preprocessed_sst2_test_set()
    elif source == 'original':
        test_set = EmoBertDataset.from_preprocessed_dataset(
            pd.read_csv(DATA_DIR / 'emoticon_paper' / 'evaluation_paper.csv')
        )
    elif source == 'premade':
        test_set = EmoBertDataset.from_preprocessed_dataset(
            pd.read_csv(DATA_DIR / 'premade_datasets' / 'evaluation_dataset_non_lemmatized.csv')
        )
    elif source == 'reddit':
        test_set = EmoBertDataset.from_preprocessed_dataset(pd.concat([
            pd.read_csv(DATA_DIR / 'reddit' / f'{polarity}_reddit_annotated_.csv')
            for polarity in ['negative', 'positive']
        ]))
    else:
        raise ValueError("parameter 'source' must be one of 'original', 'sst2', or 'premade'")
    return DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=min(MAX_BATCH_SIZE, test_set.data.shape[0]),
        collate_fn=model.custom_collate,
        num_workers=WORKERS_PER_TRIAL,
        pin_memory=True
    )
