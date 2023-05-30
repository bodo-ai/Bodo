# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of nested data functions with BodoSQL
"""
import datetime

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

import bodo


def test_to_array_scalars(basic_df, memory_leak_check):
    """Test TO_ARRAY works correctly with scalar inputs"""
    query_fmt = "TO_ARRAY({!s})"
    scalars = [
        "123",
        "456.789",
        "null",
        "'asdafa'",
        "true",
        "to_time('05:34:51')",
        "to_date('2023-05-18')",
        "to_timestamp('2024-06-29 17:00:00')",
    ]
    selects = []
    for scalar in scalars:
        selects.append(query_fmt.format(scalar))
    query = f"SELECT {', '.join(selects)}"
    py_output = pd.DataFrame(
        {
            "int": pd.Series([pd.array([123])]),
            "float": pd.Series([pd.array([456.789])]),
            "null": pd.Series([None]),
            "string": pd.Series([pd.array(["asdafa"])]),
            "bool": pd.Series([pd.array([True])]),
            "time": pd.Series([pd.array([bodo.Time(5, 34, 51)])]),
            "date": pd.Series([pd.array([datetime.date(2023, 5, 18)])]),
            "timestamp": pd.Series([pd.array([pd.Timestamp("2024-06-29 17:00:00")])]),
        }
    )
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series([423, 647, 0, 51, -425] * 4, dtype='int'),
                pd.Series(
                    [pd.array([423]), pd.array([647]), pd.array([0]), pd.array([51]), pd.array([-425])] * 4
                )
            ),
            id="integer",
        ),
        pytest.param(
            (
                pd.Series([4.23, 64.7, None, 0.51, -425.0] * 4),
                pd.Series(
                    [pd.array([4.23]), pd.array([64.7]), None, pd.array([0.51]), pd.array([-425.0])] * 4
                )
            ),
            id="float",
        ),
        pytest.param(
            (
                pd.Series(["ksef", "$@#%", None, "0.51", "1d$g"] * 4),
                pd.Series(
                    [pd.array(["ksef"]), pd.array(["$@#%"]), None, pd.array(["0.51"]), pd.array(["1d$g"])] * 4
                )
            ),
            id="string",
        ),
        pytest.param(
            (
                pd.Series([True, None, False, False, True] * 4),
                pd.Series(
                    [pd.array([True]), None, pd.array([False]), pd.array([False]), pd.array([True])] * 4
                )
            ),
            id="bool",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        bodo.Time(11, 19, 34),
                        bodo.Time(12, 30, 15),
                        bodo.Time(12, 34, 56, 78, 12),
                        None,
                        bodo.Time(12, 34, 56, 78, 12, 34),
                    ] * 4
                ),
                pd.Series(
                    [
                        pd.array([bodo.Time(11, 19, 34)]),
                        pd.array([bodo.Time(12, 30, 15)]),
                        pd.array([bodo.Time(12, 34, 56, 78, 12)]),
                        None,
                        pd.array([bodo.Time(12, 34, 56, 78, 12, 34)]),
                    ] * 4
                )
            ),
            id="time",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        datetime.date(2020, 1, 4),
                        datetime.date(1999, 5, 2),
                        datetime.date(1970, 1, 1),
                        datetime.date(2020, 11, 23),
                        None,
                    ] * 4
                ),
                pd.Series(
                    [
                        pd.array([datetime.date(2020, 1, 4)]),
                        pd.array([datetime.date(1999, 5, 2)]),
                        pd.array([datetime.date(1970, 1, 1)]),
                        pd.array([datetime.date(2020, 11, 23)]),
                        None,
                    ] * 4
                )
            ),
            id="date",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None,
                        pd.Timestamp("2020-01-01 22:00:00"),
                        pd.Timestamp("2019-1-24"),
                        pd.Timestamp("2023-7-18"),
                        pd.Timestamp("2020-01-02 01:23:42.728347"),
                    ] * 4
                ),
                pd.Series(
                    [
                        None,
                        pd.array([pd.Timestamp("2020-01-01 22:00:00")]),
                        pd.array([pd.Timestamp("2019-1-24")]),
                        pd.array([pd.Timestamp("2023-7-18")]),
                        pd.array([pd.Timestamp("2020-01-02 01:23:42.728347")]),
                    ] * 4
                )
            ),
            id="timestamp",
        ),
    ]
)
def to_array_columns_data(request):
    """input data for TO_ARRAY column tests"""
    return request.param


def test_to_array_columns(to_array_columns_data, memory_leak_check):
    """Test TO_ARRAY works correctly with column inputs"""
    query = "SELECT TO_ARRAY(A) FROM TABLE1"
    data, answer = to_array_columns_data
    py_output = pd.DataFrame({"A": answer})
    ctx = {"table1": pd.DataFrame({"A": data})}
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_to_array_arrays(to_array_columns_data, memory_leak_check):
    """tests TO_ARRAY return the same array when input is array"""
    query = "SELECT TO_ARRAY(TO_ARRAY(A)) FROM TABLE1"
    data, answer = to_array_columns_data
    py_output = pd.DataFrame({"A": answer})
    ctx = {"table1": pd.DataFrame({"A": data})}
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )
