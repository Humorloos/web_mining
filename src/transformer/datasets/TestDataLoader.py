from torch.utils.data import DataLoader

from src.constants.constants import MAX_BATCH_SIZE, MAX_WORKERS
from src.transformer.datasets.EmoticonDataset import EmoticonDataset
from src.transformer.datasets.SST2TestSet import get_sst2_test_set
from src.transformer.emoBert import EmoBERT


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
    else:
        raise ValueError("parameter 'source' must be one of 'original' or 'sst2'")
    return DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=min(MAX_BATCH_SIZE, test_set.data.shape[0]),
        collate_fn=model.custom_collate,
        num_workers=MAX_WORKERS,
        pin_memory=True
    )
