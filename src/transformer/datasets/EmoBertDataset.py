import pandas as pd
from torch.utils.data import Dataset


class EmoBertDataset(Dataset):
    """Generic dataset for EmoBERT model"""

    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    @staticmethod
    def from_crawled_dataset(dataset):
        return EmoBertDataset(
            data=dataset[['prep_text', 'sentiment']].rename(columns={'prep_text': 'text', 'sentiment': 'polarity'})
        )

    def __getitem__(self, item):
        return self.data.iloc[item]

    def __len__(self):
        return self.data.shape[0]
