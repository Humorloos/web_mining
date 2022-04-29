import pandas as pd
from torch.utils.data import Dataset

from constants import DATA_DIR


class EmoticonDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(
            DATA_DIR / 'emoticon_paper' / 'train.csv',
            encoding='latin-1',
            header=None,
            names=['polarity', 'id', 'date', 'query', 'user', 'text']
        )[['text', 'polarity']]
        # set polarity 4 to 1 to get probability vector
        self.data.loc[self.data['polarity'] == 4, 'polarity'] = 1
        self.data.dtype = 'float32'

    def __getitem__(self, item):
        return self.data.iloc[item]

    def __len__(self):
        return self.data.shape[0]
