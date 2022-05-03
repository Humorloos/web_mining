from sklearn.model_selection import train_test_split

from src.transformer.datasets.EmoBertDataset import EmoBertDataset
from src.transformer.datasets.EmoticonDataset import EmoticonDataset


def get_emoticon_train_val_split():
    """
    Gets two EmoBertDatasets, one with 10000 stratified random samples from emoticon train set (for validation), and one
    with the rest for training
    """
    data = EmoticonDataset('train').data
    return tuple(EmoBertDataset(split)
                 for split in train_test_split(data, test_size=10000, random_state=420, stratify=data['polarity']))
