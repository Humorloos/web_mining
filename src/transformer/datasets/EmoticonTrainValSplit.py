from sklearn.model_selection import train_test_split

from src.constants.constants import VAL_SET_SIZE, COMPLETE_DATA_SIZE
from src.transformer.datasets.EmoBertDataset import EmoBertDataset
from src.transformer.datasets.EmoticonDataset import EmoticonDataset


def get_emoticon_train_val_split():
    """
    Gets two EmoBertDatasets, one with VAL_SET_SIZE stratified random samples from emoticon train set (for validation), and one
    with the rest for training
    """
    data = EmoticonDataset('train').data.sample(COMPLETE_DATA_SIZE, random_state=420)
    return tuple(EmoBertDataset(split)
                 for split in train_test_split(data, test_size=VAL_SET_SIZE, random_state=420, stratify=data['polarity']))
