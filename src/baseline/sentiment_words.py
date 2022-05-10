from constants.glyph_constants import *
from preprocessors.twitter_preprocessor import *


def create_sentiment_words_dict(file: str):
    words = []
    with open(file, "r", encoding="iso-8859-1") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # check if line empty or starts with ;
            if line and (not line.startswith(";")):
                # apply same preprocessing as tweets including lemmatizer
                line = TwitterPreprocessor().preprocess_tweet(line)
                if len(line.strip()) > 1:
                    words.append(line.strip())
        # remove duplicates
        words = list(dict.fromkeys(words))
        words = sorted(words)
        return words


if __name__ == "__main__":
    # command line args
    text_file = sys.argv[1]
    output_file = sys.argv[2]
    # read in sentiment words text file
    words = create_sentiment_words_dict(text_file)
    with open(output_file, "w", encoding="iso-8859-1") as f:
        for word in words:
            if word:
                f.write("%s\n" % word)
