"""
Source: https://medium.com/rapids-ai/real-data-has-strings-now-so-do-gpus-994497d55f8e
Analyze beer reviews to find most common words used in positive and negative reviews.
    Usage:
        python beer-reviews.py

    Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.

This example uses a sample reviews from S3 bucket (s3://bodo-example-data/beer/reviews_sample.csv).
Fulldataset is available in S3 bucket (s3://bodo-example-data/beer/reviews.csv)
or from Kaggle (https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews)

"""
import pandas as pd
import time
import bodo
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
# Create lists of stopwords and punctuation that will be removed
with open(f"{dir_path}/nltk-stopwords.txt", "r") as fh:
    STOPWORDS = list(map(str.strip, fh.readlines()))
PUNCT_LIST = ["\.", "\-", "\?", "\:", ":", "!", "&", "'", ","]

# Define regex that will be used to remove these punctuation and stopwords from the reviews.
punc_regex = "|".join([f"({p})" for p in PUNCT_LIST])
stopword_regex = "|".join([f"\\b({s})\\b" for s in STOPWORDS])


@bodo.jit
def preprocess(reviews):
    # lowercase and strip
    reviews = reviews.str.lower()
    reviews = reviews.str.strip()

    # remove punctuation and stopwords
    reviews = reviews.str.replace(punc_regex, "", regex=True)
    reviews = reviews.str.replace(stopword_regex, "", regex=True)
    return reviews


@bodo.jit
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

    print("Top words: ")
    print(top_words)
    print("Low words: ")
    print(low_words)


find_top_words("s3://bodo-example-data/beer/reviews_sample.csv")
