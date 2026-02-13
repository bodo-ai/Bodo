"""Test connector for Hugging Face."""

import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.mark.slow
def test_read_csv_hf(datapath, memory_leak_check):
    """Test read_csv from Hugging Face"""

    def test_impl():
        return pd.read_csv(
            "hf://datasets/domenicrosati/TruthfulQA/train.csv", dtype_backend="pyarrow"
        )

    check_func(test_impl, ())


@pytest.mark.slow
def test_read_json_hf(datapath, memory_leak_check):
    """Test read_json from Hugging Face"""

    def test_impl():
        return pd.read_json(
            "hf://datasets/craigwu/vstar_bench/test_questions.jsonl",
            lines=True,
        )

    check_func(test_impl, ())


@pytest.mark.slow
def test_read_parquet_hf(datapath, memory_leak_check):
    """Test read_parquet from Hugging Face"""

    def test_impl():
        return pd.read_parquet(
            "hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet",
            dtype_backend="pyarrow",
        )

    check_func(test_impl, ())


@pytest.mark.slow
def test_read_parquet_hf_split(datapath, memory_leak_check):
    """Test read_parquet from Hugging Face when a split is specified
    (similar to Hugging Face online code examples)"""

    def test_impl():
        splits = {
            "train": "main/train-00000-of-00001.parquet",
            "test": "main/test-00000-of-00001.parquet",
        }
        return pd.read_parquet(
            "hf://datasets/openai/gsm8k/" + splits["train"], dtype_backend="pyarrow"
        )

    check_func(test_impl, ())
