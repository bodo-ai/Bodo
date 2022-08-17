"""
See https://github.com/Bodo-inc/examples-dev/blob/main/beer_reviews.py
"""
import sys
import time

import numba
import pandas as pd

import bodo

with open("nltk-stopwords.txt", "r") as fh:
    STOPWORDS = list(map(str.strip, fh.readlines()))


PUNCT_LIST = ["\.", "\-", "\?", "\:", ":", "!", "&", "'", ","]
punc_regex = "|".join([f"({p})" for p in PUNCT_LIST])
stopword_regex = "|".join([f"\\b({s})\\b" for s in STOPWORDS])


@bodo.jit(distributed=["reviews"])
def preprocess(reviews):
    # lowercase and strip
    reviews = reviews.str.lower()
    reviews = reviews.str.strip()

    # remove punctuation and stopwords
    reviews = reviews.str.replace(punc_regex, "", regex=True)
    reviews = reviews.str.replace(stopword_regex, "", regex=True)
    return reviews


@bodo.jit(cache=True)
def find_top_words(review_filename):
    # Load in the data
    t_start = time.time()
    df = pd.read_csv(review_filename, parse_dates=[2])
    print("read time", time.time() - t_start)

    score = df.score
    reviews = df.text

    t1 = time.time()
    reviews = preprocess(reviews)
    print("preprocess time", time.time() - t1)

    t1 = time.time()
    # create low and high score series
    low_threshold = 1.5
    high_threshold = 4.95
    high_reviews = reviews[score > high_threshold]
    low_reviews = reviews[score <= low_threshold]
    high_reviews = high_reviews.dropna()
    low_reviews = low_reviews.dropna()

    high_colsplit = high_reviews.str.split()
    low_colsplit = low_reviews.str.split()
    print("high/low time", time.time() - t1)

    t1 = time.time()
    high_words = high_colsplit.explode()
    low_words = low_colsplit.explode()

    top_words = high_words.value_counts().head(25)
    low_words = low_words.value_counts().head(25)
    print("value_counts time", time.time() - t1)
    print("total time", time.time() - t_start)

    print("TOP WORDS:")
    print(top_words)
    print("LOW WORDS:")
    print(low_words)


if __name__ == "__main__":
    fname = sys.argv[1]
    require_cache = False
    if len(sys.argv) > 2:
        require_cache = bool(sys.argv[2])
    find_top_words(fname)
    if require_cache and isinstance(find_top_words, numba.core.dispatcher.Dispatcher):
        assert (
            find_top_words._cache_hits[find_top_words.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"
