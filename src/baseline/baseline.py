from constants.glyph_constants import *
from preprocessors.twitter_preprocessor import *


def read_sentiment_words_dict(file: str):
    words = []
    with open(file, "r", encoding="iso-8859-1") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                words.append(line)
        words = list(dict.fromkeys(words))
        words = sorted(words)
        return words


def apply_baseline(tweet: str, word_count=False):
    tweet = tweet.strip()
    words = tweet.split()
    negative_count, positive_count = 0, 0
    for word in words:
        if word in NEGATIVE_WORDS:
            negative_count += 1
        elif word in POSITIVE_WORDS:
            positive_count += 1
        else:
            pass
    data_dict={"negative_count": negative_count, "positive_count": positive_count, "baseline": None}
    if negative_count > positive_count:
        data_dict["baseline"]=int(0)
    else:
        data_dict["baseline"]=int(1)
    # return baseline values
    if word_count == False:
        return data_dict["baseline"]
    elif word_count == True:
        return data_dict
    else:
        pass


NEGATIVE_WORDS = read_sentiment_words_dict("src/baseline/data/negative_words.txt")
POSITIVE_WORDS = read_sentiment_words_dict("src/baseline/data/positive_words.txt")

if __name__ == "__main__":
    # command line args
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    # read in tweets csv
    df = pd.read_csv(csv_file, engine="python", delimiter=",", usecols=["full_text","prep_text","sentiment"])
    # baseline prediction
    baseline = df["prep_text"].progress_apply(lambda x: apply_baseline(x))
    # insert series into dataframe
    df.insert(3, "baseline", baseline)
    # write series to csv file
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
