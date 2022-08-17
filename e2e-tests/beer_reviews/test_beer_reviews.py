import os
import re
import shutil

from utils.utils import run_cmd

BUCKET_NAME = "bodotest-customer-data"


# -------- oracle --------
top_words = {}
top_words["beer"] = 333
top_words["one"] = 158
top_words["taste"] = 140
top_words["head"] = 119
top_words["like"] = 117
top_words["dark"] = 90
top_words["chocolate"] = 90
top_words["great"] = 86
top_words["perfect"] = 80
top_words["good"] = 79


low_words = {}
low_words["beer"] = 239
low_words["like"] = 109
low_words["taste"] = 104
low_words["head"] = 69
low_words["light"] = 65
low_words["one"] = 65
low_words["smell"] = 57
low_words["bad"] = 53
low_words["bottle"] = 52
low_words["really"] = 49
# ------------------------


def process_output(output):
    regexp = re.compile(r"(\S+)\s+(\d+)")
    top_words = {}
    low_words = {}
    reading_top_words = False
    reading_low_words = False
    for l in output.splitlines():
        if l.startswith("TOP WORDS:"):
            reading_top_words = True
        elif l.startswith("LOW WORDS:"):
            reading_top_words = False
            reading_low_words = True
        if reading_top_words or reading_low_words:
            m = regexp.match(l)
            if m:
                word = m.group(1)
                count = int(m.group(2))
                if reading_top_words:
                    top_words[word] = count
                elif reading_low_words:
                    low_words[word] = count
    return top_words, low_words


def check_result(top_words_out, low_words_out):
    assert len(set(top_words.keys()).intersection(top_words_out.keys())) == len(
        top_words
    )
    assert len(set(low_words.keys()).intersection(low_words_out.keys())) == len(
        low_words
    )
    for word, count in top_words.items():
        assert top_words_out[word] == count
    for word, count in low_words.items():
        assert low_words_out[word] == count


def test_beer_reviews():
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))

        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        beers_fn = "s3://{}/beer_reviews/reviews_sample.csv".format(BUCKET_NAME)

        # --------- run with pandas first ---------
        os.environ["NUMBA_DISABLE_JIT"] = "1"

        cmd = ["python", "-u", "beer_reviews.py", beers_fn]
        top_words_out, low_words_out = process_output(run_cmd(cmd))
        check_result(top_words_out, low_words_out)

        os.environ["NUMBA_DISABLE_JIT"] = "0"

        # --------- now run with Bodo ---------

        # run with 1 process first and generate cache
        top_words_out, low_words_out = process_output(run_cmd(cmd))
        check_result(top_words_out, low_words_out)

        for num_processes in (4,):
            cmd = [
                "mpiexec",
                "-n",
                str(num_processes),
                "python",
                "-u",
                "beer_reviews.py",
                beers_fn,
                "True",  # tell script to make sure we load from cache, or fail
            ]
            top_words_out, low_words_out = process_output(run_cmd(cmd))
            check_result(top_words_out, low_words_out)

    finally:
        # make sure all state is restored even in the case of exceptions
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        os.chdir(pytest_working_dir)
