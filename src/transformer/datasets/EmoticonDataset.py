"""Generic Dataset for training based on original author's training set"""

import pandas as pd

from constants import DATA_DIR


class EmoticonDataset:
    def __init__(self, purpose):
        self.data = pd.read_csv(
            DATA_DIR / 'emoticon_paper' / f'{purpose}.csv',
            encoding='latin-1',
            header=None,
            names=['polarity', 'id', 'date', 'query', 'user', 'text']
        )[['text', 'polarity']]
        self.data = self.data.loc[self.data['polarity'] != 2]
        # set polarity 4 to 1 to get probability vector
        self.data.loc[self.data['polarity'] == 4, 'polarity'] = 1
