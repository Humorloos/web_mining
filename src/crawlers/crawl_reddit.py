"""Script for crawling reddit data"""
# %%
import os

import pandas as pd
from psaw import PushshiftAPI

from src.constants.constants import DATA_DIR

# %%
# Setup
# get stopwords for stopword removal

# todo: here you set the number of days before today of the date at which you start crawling the posts.
# The code below will then crawl all tweets from this day until today.
DELTA_DAYS = 1
START_DATE = pd.Timestamp.now() - pd.Timedelta(days=DELTA_DAYS-1)
REDDIT_DATA_DIR = DATA_DIR / 'reddit'

# %%
# Data collection

api = PushshiftAPI()  # pushshift api is a 3rd party service that allows querying reddit with more expressive queries
for i in range(DELTA_DAYS):
    date = START_DATE + pd.Timedelta(days=i)
    file_name = f'{date.date()}.pickle'
    if file_name not in os.listdir(REDDIT_DATA_DIR):
        print(f'Downloading comments for date {date}')
        # Todo: here you can either search for comments (as currently configured) or for posts (api.search_submissions)
        # for search parameter reference, see https://github.com/pushshift/api#search-parameters-for-submissions
        # or https://github.com/pushshift/api#search-parameters-for-comments
        pd.DataFrame([comment.d_ for comment in api.search_comments(
            after=int(date.timestamp()),
            before=int((date + pd.Timedelta(days=1)).timestamp()),
            subreddit='de',
            user_removed=False,
            mod_removed=False,
        )]).to_pickle(REDDIT_DATA_DIR / file_name)
    else:
        print(f'Comments for date {date} already downloaded, '
              f'proceeding to next day')
# %%
# Load comments from disk
comments = pd.concat([
    pd.read_pickle(REDDIT_DATA_DIR / file_name) for file_name in
    os.listdir(REDDIT_DATA_DIR)
], ignore_index=True)
# %%
# Some simple reddit-specific preprocessing steps for removing irrelevant posts

# exclude removed comments
comments = comments[comments['body'] != '[removed]']

# exclude comments by moderators (they are mostly about why a
# comment was removed and create an irrelevant topic)
comments = comments[comments['distinguished'] != 'moderator']
