import pandas as pd
from sklearn.model_selection import train_test_split

from constants import DATA_DIR, COMPLETE_DATA_SIZE, VAL_SET_SIZE
from datasets.EmoBertDataset import EmoBertDataset
from datasets.EmoticonDataset import EmoticonDataset


def get_train_val_split(source):
    """
    Gets two EmoBertDatasets, one with VAL_SET_SIZE stratified random samples from emoticon train set (for validation),
    and one with the rest for training
    :param source: 'original' for author's original training set, 'crawl' for our own crawled tweets, 'premade' for
    crawled dataset without tweets from crawled evaluation set
    :return: a tuple with 2 EmoBertDatasets where the first one is the training set and the second one the validation
    set
    """
    if source == 'original':
        data = EmoticonDataset('train').data
    elif source == 'crawl':
        data = pd.concat([
            pd.read_csv(DATA_DIR / 'crawled_tweets' / f'{filename}.csv')
            for filename in ['pos_tweets_590k', 'neg_tweets_600k', 'new_pos_tweets_490k', 'new_neg_tweets_560k']
        ])[['prep_text', 'sentiment']] \
            .rename(columns={'prep_text': 'text', 'sentiment': 'polarity'})
    elif source == 'premade':
        data = pd.read_csv(
            DATA_DIR / 'premade_datasets' / 'training_dataset_non_lemmatized.csv'
        )[['prep_text', 'sentiment']].rename(columns={'prep_text': 'text', 'sentiment': 'polarity'})
    else:
        raise ValueError("parameter 'source' must be one of 'original', 'crawl', or 'premade'")
    data = data.sample(COMPLETE_DATA_SIZE, random_state=420)
    return tuple(
        EmoBertDataset(split) for split in
        train_test_split(
            data,
            test_size=VAL_SET_SIZE,
            random_state=420,
            stratify=data['polarity']
        )
    )
