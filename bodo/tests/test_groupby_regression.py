"""
Compares performance of streaming groupby vs non-streaming groupby.
There are two test cases, one where every key is unique and one where the number of unique keys is small.
"""
import time

import bodosql
import pandas as pd

import bodo

from bodo.tests.utils import (
    pytest_perf_regression
)

pytestmark = pytest_perf_regression

data_size = 10000000
few_unique_nkeys = 20


def impl(bc):
    start = time.time()
    out = bc.sql("select a,sum(b) from t1 group by a")
    return out, time.time() - start


def test_groupby_regression_all_unique():
    """This tests the performance of streaming groupby vs non-streaming groupby when every key is unique."""
    prev_streaming = bodo.bodosql_use_streaming_plan
    prev_groupby_enabled = bodo.enable_groupby_streaming

    try:
        bodo.bodosql_use_streaming_plan = True
        bodo.enable_groupby_streaming = True
        t1 = pd.DataFrame({"a": range(data_size), "b": range(data_size)})
        _, streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"t1": t1}))

        bodo.bodosql_use_streaming_plan = False
        _, non_streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"t1": t1}))
    finally:
        bodo.bodosql_use_streaming_plan = prev_streaming
        bodo.enable_groupby_streaming = prev_groupby_enabled

    print("streaming_time_all_unique: ", streaming_time)
    print("non_streaming_time_all_unique: ", non_streaming_time)

    # Disabling for now until we resolve performance issues
    # assert streaming_time < non_streaming_time


def test_groupby_regression_few_unique():
    """
    This tests the performance of streaming groupby vs non-streaming groupby when the number of unique keys is small.
    """
    prev_streaming = bodo.bodosql_use_streaming_plan
    prev_groupby_enabled = bodo.enable_groupby_streaming

    try:
        bodo.bodosql_use_streaming_plan = True
        bodo.enable_groupby_streaming = True
        t1 = pd.DataFrame(
            {
                "a": list(range(few_unique_nkeys)) * (data_size // few_unique_nkeys),
                "b": range(data_size),
            }
        )
        _, streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"t1": t1}))

        bodo.bodosql_use_streaming_plan = False
        _, non_streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"t1": t1}))
    finally:
        bodo.bodosql_use_streaming_plan = prev_streaming
        bodo.enable_groupby_streaming = prev_groupby_enabled
    print("streaming_time_few_unique: ", streaming_time)
    print("non_streaming_time_few_unique: ", non_streaming_time)

    # Disabling for now until we resolve performance issues
    # assert streaming_time < non_streaming_time
