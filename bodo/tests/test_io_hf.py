"""Test connector for Hugging Face."""

import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.mark.slow
def test_read_csv_hf(datapath, memory_leak_check):
    """Test read_csv from Hugging Face"""

    def test_impl():
        return pd.read_csv("hf://datasets/fka/awesome-chatgpt-prompts/prompts.csv")

    check_func(test_impl, ())


@pytest.mark.slow
def test_read_json_hf(datapath, memory_leak_check):
    """Test read_json from Hugging Face"""

    def test_impl():
        return pd.read_json(
            "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True
        )

    check_func(test_impl, ())


@pytest.mark.slow
def test_read_parquet_hf(datapath, memory_leak_check):
    """Test read_parquet from Hugging Face"""

    def test_impl():
        return pd.read_parquet(
            "hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet"
        )

    check_func(test_impl, ())
