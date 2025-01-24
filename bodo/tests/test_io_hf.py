"""Test connector for Hugging Face."""

import pandas as pd

from bodo.tests.utils import check_func


def test_read_csv_hf(datapath, memory_leak_check):
    """Test read_csv from Hugging Face"""

    def test_impl():
        return pd.read_csv("hf://datasets/fka/awesome-chatgpt-prompts/prompts.csv")

    check_func(test_impl, ())
