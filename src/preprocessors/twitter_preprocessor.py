from emoji import EMOJI_DATA
from constants.glyph_constants import (
    NEGATIVE_EMOJIS,
    NEGATIVE_EMOTICON_REGEX,
    POSITIVE_EMOJIS,
    POSITIVE_EMOTICON_REGEX,
)
import pandas as pd
import re
import sys
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
tqdm.pandas()


SENTIMENT_EMOJIS = NEGATIVE_EMOJIS + POSITIVE_EMOJIS
NEUTRAL_EMOJIS = [i for i in list(
    EMOJI_DATA.keys()) if i not in SENTIMENT_EMOJIS]


class TwitterPreprocessor:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
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
        tweet = re.sub(r"({})".format("|".join(SENTIMENT_EMOJIS)), " ", tweet)
        # replace other neutral emojis with EMO token
        tweet = re.sub(r"({})".format(
            "|".join(map(re.escape, NEUTRAL_EMOJIS))), " EMO ", tweet)
        return tweet

    def is_valid_word(self, word: str):
        return re.search(r"^[a-z0-9A-Z][a-z0-9A-Z\._]*$", word) is not None

    def get_pos(self, word: str):
        # get POS tag of a word
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                    "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize(self, word: str):
        # lemmatize word depending on POS tag
        return self.lemmatizer.lemmatize(word, pos=self.get_pos(word))

    def preprocess_tweet(self, tweet: str, lemmatize=True):
        preprocessed_tweet = []

        # remove emoticons
        tweet = self.remove_emoticons(tweet)
        # to lower case
        tweet = tweet.lower()
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
            word = word.strip("!\"\#$%()*+,./:;<=>?@[\]^_`{|}~…‘’“”«»¿×～")
            # remove repetitions
            word = self.remove_repetitions(word)
            # remove further punctuations
            word = re.sub(r"(-|\'|&)", "", word)

            # check if word valid
            if self.is_valid_word(word):
                # lemmatize word
                if lemmatize==True:
                    word = self.lemmatize(word)
                preprocessed_tweet.append(word)

        return " ".join(preprocessed_tweet)


if __name__ == "__main__":
    twitter_processor = TwitterPreprocessor()
    # command line args
    tweets_file = sys.argv[1]
    preprocessed_file = sys.argv[2]
    lemmatize=sys.argv[3]
    if lemmatize=="y":
        lemmatize=True
    elif lemmatize=="n":
        lemmatize=False
    else:
        sys.exit(f'wrong lemmatize argument provided: {lemmatize}. provide either "y"=yes or "n"=no')
    # read in tweets csv
    df = pd.read_csv(tweets_file, engine="python",
                     delimiter=",", usecols=["full_text","sentiment"])
    # apply preprocessing
    # save preprocessed tweets as pandas series
    preprocessed_text = df["full_text"].progress_apply(
        lambda x: twitter_processor.preprocess_tweet(x, lemmatize=lemmatize))
    # insert series into dataframe
    df.insert(1, "prep_text", preprocessed_text)
    # write series to pickle file
    df.to_csv(preprocessed_file, index=False, quoting=csv.QUOTE_ALL)
