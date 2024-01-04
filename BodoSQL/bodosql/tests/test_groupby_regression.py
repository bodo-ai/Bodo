"""
Compares performance of streaming groupby vs non-streaming groupby.
There are two test cases, one where every key is unique and one where the number of unique keys is small.
"""
import time

import pandas as pd

import bodo
import bodosql
from bodo.tests.utils import pytest_perf_regression

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

    try:
        bodo.bodosql_use_streaming_plan = True
        t1 = pd.DataFrame({"A": range(data_size), "B": range(data_size)})
        _, streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"T1": t1}))

        bodo.bodosql_use_streaming_plan = False
        _, non_streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"T1": t1}))
    finally:
        bodo.bodosql_use_streaming_plan = prev_streaming

    print("streaming_time_all_unique: ", streaming_time)
    print("non_streaming_time_all_unique: ", non_streaming_time)

    # Disabling for now until we resolve performance issues
    # assert streaming_time < non_streaming_time


def test_groupby_regression_few_unique():
    """
    This tests the performance of streaming groupby vs non-streaming groupby when the number of unique keys is small.
    """
    prev_streaming = bodo.bodosql_use_streaming_plan

    try:
        bodo.bodosql_use_streaming_plan = True
        t1 = pd.DataFrame(
            {
                "A": list(range(few_unique_nkeys)) * (data_size // few_unique_nkeys),
                "B": range(data_size),
            }
        )
        _, streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"T1": t1}))

        bodo.bodosql_use_streaming_plan = False
        _, non_streaming_time = bodo.jit(impl)(bodosql.BodoSQLContext({"T1": t1}))
    finally:
        bodo.bodosql_use_streaming_plan = prev_streaming
    print("streaming_time_few_unique: ", streaming_time)
    print("non_streaming_time_few_unique: ", non_streaming_time)

    # Disabling for now until we resolve performance issues
    # assert streaming_time < non_streaming_time
