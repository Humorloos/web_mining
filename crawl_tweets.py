#!~/environs/env1/bin/python

import pandas as pd
import csv
import tweepy
import os
import numpy as np
import time
import sys
import re
from datetime import datetime

BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

POSITIVE_EMOTICONS = [
    ":)",
    ": )",
    ":-)",
    "(:",
    "( :",
    "(-:",
    ":')",
    ":D",
    ": D",
    ":-D",
    "xD",
    "x-D",
    "XD",
    "X-D",
    ";-)",
    ";)",
    ";-D",
    ";D",
    "(;",
    "(-;",
    "<3",
    ":*",
]

NEGATIVE_EMOTICONS = [
    ":-(",
    ": (",
    ":(",
    "):",
    ")-:",
    ":'(",
    #':"(',
]

POSITIVE_EMOJIS = [
    "😊",
    "😀",
    "😃",
    "😄",
    "😁",
    "😂",
    "🤣",
    "😍",
    "🥰",
    "😘",
    "♥",
    "❤️",
    "🧡",
    "💛",
    "💚",
    "💙",
    "💜",
    "🖤",
    "🤍",
    "🤎",
    "❤️‍🔥",
    "❤️‍🩹",
    "❣️",
    "💕",
    "💞",
    "💓",
    "💗",
    "💖",
    "💘",
    "💝",
]
NEGATIVE_EMOJIS = [
    "😞",
    "😔",
    "😕",
    "🙁",
    "😭",
    "🥺",
    "😟",
    "😕",
    "🙁",
    "☹️",
    "😣",
    "😖",
    "😫",
    "😩",
    "😢",
    "😤",
    "😠",
    "😡",
    "🤬",
    "😱",
    "😨",
    "😰",
    "😥",
    "😓",
]

POSITIVE_EMOTICON_REGEX=r"(:\s?\)|:-\)|\(\s?:|\(-:|:\s?D|:-D|x-?D|X-?D|;-?\)|;-?D|\(-?;|<3|:\*|&lt;3)"
NEGATIVE_EMOTICON_REGEX=r"(:\s?\(|:-\(|\)\s?:|\)-:|:,\(|:\'\(|:\"\()"

CSV_HEADER = [
    "id",
    "full_text",
    "created_at",
    "sentiment",
    "lang",
    "iso_language_code",
    "result_type",
    "user_id",
    "user_name",
    "user_screen_name",
]

DATA_DICT = {i: "" for i in CSV_HEADER}


def build_query(sentiment: str):
    if sentiment == "p":
        emoticons = POSITIVE_EMOTICONS
        emojis = POSITIVE_EMOJIS
    elif sentiment == "n":
        emoticons = NEGATIVE_EMOTICONS
        emojis = NEGATIVE_EMOJIS
    else:
        sys.exit('provide either "p"=positive or "n"=negative sentiment')
    query = f"""({" OR ".join([f'"{i}"' for i in emoticons])} OR {" OR ".join([f'"{i}"' for i in emojis])}) -filter:retweets -filter:links -filter:media"""
    return query


def authenticate():
    auth = tweepy.OAuth2BearerHandler(BEARER_TOKEN)
    return tweepy.API(auth)


def get_tweets(api: tweepy.API, query: str, number: int):
    tweets = api.search_tweets(
        q=query, lang="en", result_type="recent", count=number, tweet_mode="extended"
    )
    return [tweet._json for tweet in tweets]


def get_unique_tweet_ids(sentiment: str):
    path = None
    tweet_ids = []
    if sentiment == "p":
        path = "data/tweets/positive"
    elif sentiment == "n":
        path = "data/tweets/negative"
    else:
        sys.exit(f"wrong sentiment: {sentiment} - p or n needed.")
    tweet_files = [f"{path}/{f}" for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]

    for p in tweet_files:
        df = pd.read_csv(p, engine="python", delimiter=",")
        tweet_ids += [int(id) for id in df["id"].to_list()]
    return tweet_ids

def check_emoticons(regex: str, text: str):
    return bool(re.search(regex, text))

def check_emojis(emojis: list, text: str):
    for c in text.replace(" ", ""):
        if c in emojis:
            return True
    return False

def check_tweet(sentiment:str,text:str):
    if sentiment not in ["p","n"]:
        sys.exit('provide either "p"=positive or "n"=negative sentiment')
    if (sentiment=="p"):
        if ( check_emoticons(POSITIVE_EMOTICON_REGEX,text=text) or check_emojis(POSITIVE_EMOJIS,text=text) ):
            if ( (not check_emoticons(NEGATIVE_EMOTICON_REGEX,text=text)) and (not check_emojis(NEGATIVE_EMOJIS,text=text)) ):
                return True
            else:
                return False
        else:
            return False
    elif (sentiment=="n"):
        if ( check_emoticons(NEGATIVE_EMOTICON_REGEX,text=text) or check_emojis(NEGATIVE_EMOJIS,text=text) ):
            if ( (not check_emoticons(POSITIVE_EMOTICON_REGEX,text=text)) and (not check_emojis(POSITIVE_EMOJIS,text=text)) ):
                return True
            else:
                return False
        else:
            return False
    else:
        pass


def write_tweets_to_csv(file: str, tweets: list, tweet_ids: list, sentiment: str):
    with open(file, "a", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if os.stat(file).st_size == 0:
            writer.writerow(CSV_HEADER)

        for tweet in tweets:
            data_dict = DATA_DICT
            # id
            if "id" in tweet.keys() and tweet["id"] not in tweet_ids:
                data_dict["id"] = tweet["id"]
                # full_text
                if "full_text" in tweet.keys():
                    data_dict["full_text"] = tweet["full_text"]
                # created_at
                if "created_at" in tweet.keys():
                    data_dict["created_at"] = tweet["created_at"]
                # lang
                if "lang" in tweet.keys():
                    data_dict["lang"] = tweet["lang"]
                if "metadata" in tweet.keys():
                    # iso_language_code
                    if "iso_language_code" in tweet["metadata"].keys():
                        data_dict["iso_language_code"] = tweet["metadata"][
                            "iso_language_code"
                        ]
                    # result_type
                    if "result_type" in tweet["metadata"].keys():
                        data_dict["result_type"] = tweet["metadata"]["result_type"]
                # user data
                if "user" in tweet.keys():
                    # user_id
                    if "id" in tweet["user"].keys():
                        data_dict["user_id"] = tweet["user"]["id"]
                    # user_name
                    if "name" in tweet["user"].keys():
                        data_dict["user_name"] = tweet["user"]["name"]
                    # user_screen_name
                    if "screen_name" in tweet["user"].keys():
                        data_dict["user_screen_name"] = tweet["user"]["screen_name"]
                if sentiment == "p":
                    data_dict["sentiment"] = 1
                elif sentiment == "n":
                    data_dict["sentiment"] = 0
                else:
                    pass

                # cleanse values
                for k, v in data_dict.items():
                    if type(v) == str:
                        val = v.replace("\n", " ").replace("\t", "")
                        val = " ".join(val.split())
                        data_dict[k] = val

                # check if tweet is valid
                if check_tweet(sentiment,text=data_dict["full_text"]):
                    writer.writerow(list(data_dict.values()))
                    tweet_ids.append(tweet["id"])

            else:
                continue
        f.close()


if __name__ == "__main__":
    api = authenticate()
    #
    t0 = time.time()
    path = None
    # command line args
    if len(sys.argv) != 5:
        sys.exit(f"wrong number command line arguments: {sys.argv[1:]}. 4 needed")
    sentiment = str(sys.argv[1])
    hours = int(sys.argv[2])
    csv_file = str(sys.argv[3])
    log_file = f"data/tweets/logs/{sys.argv[4]}"
    if sentiment not in ["p", "n"]:
        sys.exit('provide either "p"=positive or "n"=negative sentiment')
    if sentiment == "p":
        path = "data/tweets/positive"
    elif sentiment == "n":
        path = "data/tweets/negative"

    # build query
    query = build_query(sentiment)

    tweet_ids = get_unique_tweet_ids(sentiment)
    while time.time() - t0 < ((59 * (60 * hours)) if hours >= 1 else 59):
        try:
            tweets = get_tweets(api, query, 100)
        except Exception as e:
            print(e)
            continue
        write_tweets_to_csv(file=f"{path}/{csv_file}",tweets=tweets,tweet_ids=tweet_ids,sentiment=sentiment)
        original_stdout=sys.stdout
        with open(log_file,"a") as f:
            print(f'[{datetime.now().strftime("%H:%M:%S %d.%m.")}]  unique tweets: {len(tweet_ids)} ;'
            +f' all tweets unique: {(np.unique(tweet_ids).size == len(tweet_ids))} ;'
            +f' new tweets: {( (sum(1 for line in open(f"{path}/{csv_file}"))) - 1)}')
            sys.stdout=f
            print(f'[{datetime.now().strftime("%H:%M:%S %d.%m")}]  unique tweets: {len(tweet_ids)} ;'
            +f' all tweets unique: {(np.unique(tweet_ids).size == len(tweet_ids))} ;'
            +f' new tweets: {( (sum(1 for line in open(f"{path}/{csv_file}"))) - 1)}')
            sys.stdout=original_stdout
            f.close()
        time.sleep(10)
