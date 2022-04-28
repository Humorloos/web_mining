import pandas as pd
import re
import sys
import random

from helper import (
    NEGATIVE_EMOJIS,
    NEGATIVE_EMOTICON_REGEX,
    POSITIVE_EMOJIS,
    POSITIVE_EMOTICON_REGEX,
)
from emoji import EMOJI_DATA

SENTIMENT_EMOJIS = NEGATIVE_EMOJIS + POSITIVE_EMOJIS

NEUTRAL_EMOJIS = [i for i in list(EMOJI_DATA.keys()) if i not in SENTIMENT_EMOJIS]


class TwitterProcessor:
    def remove_repetitions(self, word: str):
        # replace 2+ letters with 2 letters
        # huuuuuuungry -> huungry
        return re.sub(r"(.)\1+", r"\1\1", word)

    def remove_emoticons(self, tweet: str):
        tweet = re.sub(POSITIVE_EMOTICON_REGEX, "  ", tweet)
        tweet = re.sub(NEGATIVE_EMOTICON_REGEX, "  ", tweet)
        return tweet

    def handle_emojis(self, tweet: str):
        # remove positive and negative emojis
        tweet=re.sub(r"({})".format("|".join(SENTIMENT_EMOJIS))," ",tweet)
        # replace other neutral emojis with EMO token
        tweet=re.sub(r"({})".format("|".join(map(re.escape,NEUTRAL_EMOJIS)))," EMO ",tweet)
        return tweet

    def is_valid_word(self, word: str):
        return re.search(r"^[a-zA-Z][a-z0-9A-Z\._]*$", word) is not None

    def preprocess_tweet(self, tweet: str):
        preprocessed_tweet = []

        # to lower case
        tweet = tweet.lower()
        # remove emoticons
        tweet = self.remove_emoticons(tweet)
        # handle emojis
        tweet = self.handle_emojis(tweet)
        # replaces URL with URL token
        tweet = re.sub(r"((www\.[\S]+)|(https?://[\S]+))", "URL", tweet)
        # replace @user with USERNAME token
        tweet = re.sub(r"@[\S]+", "USERNAME", tweet)
        # replace #hashtag with hashtag
        tweet = re.sub(r"#(\S+)", r" \1 ", tweet)
        # remove retweets (RT)
        tweet = re.sub(r"\brt\b", "", tweet)
        # replace 2+ dots with space
        tweet = re.sub(r"\.{2,}", " ", tweet)

        # remove multiple spaces
        tweet = re.sub(r"\s+", " ", tweet)

        for word in tweet.split():
            # remove punctuation
            word = word.strip("'\"?!,.():;")
            # remove repetitions
            word = self.remove_repetitions(word)
            # remove further punctuations
            word = re.sub(r"(-|\'|&)", "", word)

            # check if word valid
            # if self.is_valid_word(word):
            preprocessed_tweet.append(word)

        return " ".join(preprocessed_tweet)


if __name__ == "__main__":
    twitter_processor = TwitterProcessor()
    # read in tweets csv
    df = pd.read_csv(sys.argv[1], engine="python", delimiter=",", usecols=["full_text"])
    # get random tweet
    tweet = random.choice(df["full_text"].to_list())
    print(f'"{tweet}"')
    print(f'"{twitter_processor.preprocess_tweet(tweet)}"')
