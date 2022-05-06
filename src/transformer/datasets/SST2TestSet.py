import pandas as pd

from src.constants.constants import DATA_DIR
from src.transformer.datasets.EmoBertDataset import EmoBertDataset

SST2_DIR = DATA_DIR / 'sst2'


def get_sst2_test_set():
    split = pd.read_csv(SST2_DIR / 'datasetSplit.txt')
    dictionary = pd.read_csv(SST2_DIR / 'dictionary.txt', sep='|', header=None,
                             names=['text', 'sentence_index'])
    labels = pd.read_csv(SST2_DIR / 'sentiment_labels.txt', sep='|').rename(columns={
        'phrase ids': 'sentence_index',
        'sentiment values': 'polarity'
    })
    test_split = split.loc[split['splitset_label'] == 2].drop(columns='splitset_label')
    texts = test_split.merge(dictionary)
    texts_with_labels = texts.merge(labels).drop(columns='sentence_index')
    texts_without_neutral = texts_with_labels[
        (texts_with_labels['polarity'] <= 0.4) | (texts_with_labels['polarity'] > 0.6)
        ]
    texts_without_neutral['polarity'] = texts_without_neutral['polarity'].round().astype('int')
    return EmoBertDataset(texts_without_neutral)
