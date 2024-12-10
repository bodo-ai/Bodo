"""
Simple word count example which returns a sorted count of words after basic processing

    Usage:
        python beer-reviews.py

    Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.
"""

import time
import argparse
import pandas as pd
import bodo


@bodo.jit
def lexi_main(fname):
    """
    Simple word count example which returns a sorted count of words in "text" column of
    a dataframe
    """

    t0 = time.time()
    case_notes = pd.read_csv(fname, usecols=["text"])
    print("I/O time:", time.time() - t0)


    t1 = time.time()
    case_notes["text"] = case_notes["text"].str.lower()
    text_split_series = case_notes["text"].str.split()
    str_series = text_split_series.explode()
    str_series = str_series[str_series.str.len() < 100]
    word_count = str_series.value_counts()

    print("Finished counts after", time.time() - t1)
    print(word_count.head(25))


def main():
    """
    Read input file name from command line and run lexi_main
    """

    parser = argparse.ArgumentParser(description='word count')
    parser.add_argument('--file', dest='file', type=str, default="s3://bodo-example-data/beer/reviews_sample.csv")
    args = parser.parse_args()

    datafile = args.file
    lexi_main(datafile)


if __name__ == '__main__':
    main()
