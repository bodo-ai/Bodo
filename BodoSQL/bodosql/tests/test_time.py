# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test SQL `time` support
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

import bodo


@pytest.mark.parametrize(
    "precision",
    [
        0,
        3,
        6,
        9,
    ],
)
def test_time_array_box_unbox(precision, memory_leak_check):
    query = "select * from table1"
    df = pd.DataFrame(
        {
            "A": [bodo.Time(0, i, 0, precision=precision) for i in range(15)],
            "B": [bodo.Time(1, i, 1, precision=precision) for i in range(15)],
            "C": [bodo.Time(2, i, 2, precision=precision) for i in range(15)],
        },
    )
    ctx = {"table1": df}
    expected_output = df
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(0, marks=pytest.mark.slow),
        pytest.param(3, marks=pytest.mark.slow),
        pytest.param(6, marks=pytest.mark.slow),
        pytest.param(9),
    ],
)
def test_time_box_array_unbox(precision, memory_leak_check):
    query = "select B from table1"
    df = pd.DataFrame(
        {
            "A": [bodo.Time(0, i, 0, precision=precision) for i in range(15)],
            "B": [bodo.Time(1, i, 1, precision=precision) for i in range(15)],
            "C": [bodo.Time(2, i, 2, precision=precision) for i in range(15)],
        },
    )
    ctx = {"table1": df}
    expected_output = df[["B"]]
    check_query(query, ctx, None, expected_output=expected_output)


to_time_fn_names = pytest.mark.parametrize(
    "fn_name",
    [
        pytest.param(
            "time", marks=pytest.mark.skip(reason="waiting for calcite support")
        ),
        "to_time",
    ],
)

to_time_in_outs = pytest.mark.parametrize(
    "input,output",
    [
        pytest.param(
            "'01:23:45'",
            bodo.Time(1, 23, 45, precision=9),
            id="time_from_string",
        ),
        pytest.param(
            100,
            bodo.Time(0, 0, 100, precision=9),
            id="time_from_int",
        ),
    ],
)


@to_time_fn_names
@to_time_in_outs
def test_time_to_time(fn_name, input, output, memory_leak_check):
    query = f"select {fn_name}({input}) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": output}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output, run_dist_tests=False)


@to_time_fn_names
@to_time_in_outs
def test_time_to_time_case(fn_name, input, output, memory_leak_check):
    query = f"select case when B is null then {fn_name}({input}) else {fn_name}({input}) end as A from table1"
    ctx = {"table1": pd.DataFrame({"B": [None]})}
    expected_output = pd.DataFrame({"A": output}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output, run_dist_tests=False)


@to_time_fn_names
@pytest.mark.parametrize(
    "input,output",
    [
        pytest.param(
            pd.Series(list(range(15))),
            pd.Series([bodo.Time(0, 0, i, precision=9) for i in range(15)]),
            id="time_from_int_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(list(range(14)) + [None], dtype="Int64"),
            pd.Series([bodo.Time(0, 0, i, precision=9) for i in range(14)] + [None]),
            id="time_from_int_vector_nulls",
        ),
    ],
)
def test_time_to_time_vector(fn_name, input, output, memory_leak_check):
    query = "select to_time(A) as A from table1"
    ctx = {"table1": pd.DataFrame({"A": input})}
    expected_output = pd.DataFrame({"A": output}, index=np.arange(len(input)))
    check_query(query, ctx, None, expected_output=expected_output, run_dist_tests=False)


@pytest.mark.parametrize(
    "args,value",
    [
        pytest.param(
            "1, 2, 3",
            bodo.Time(1, 2, 3, precision=9),
            id="no_nanoseconds",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "1, 2, 3, 4",
            bodo.Time(1, 2, 3, 4, precision=9),
            id="nanoseconds",
        ),
    ],
)
def test_time_from_parts(args, value, memory_leak_check):
    query = f"select time_from_parts({args}) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": value}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output, run_dist_tests=False)


@pytest.mark.parametrize(
    "part,value",
    [
        pytest.param(
            "hour",
            1,
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "minute",
            2,
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "second",
            3,
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "millisecond",
            4,
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "microsecond",
            5,
            id="microsecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "nanosecond",
            6,
            id="nanosecond",
        ),
    ],
)
def test_time_extract(part, value, memory_leak_check):
    query = f"select extract({part} from to_time('01:02:03.004005006')) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": value}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output)
