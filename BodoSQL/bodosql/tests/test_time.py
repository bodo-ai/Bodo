# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test SQL `time` support
"""
import numpy as np
import pandas as pd
import pytest
from bodo.tests.conftest import time_df, time_part_strings, day_part_strings
from bodo.tests.timezone_common import date_sub_unit_time_fn
from bodosql.context import BodoSQLContext
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
            pd.Series(
                [
                    "11:23:45.999",
                    "01:23:45",
                    "01:23:45.67588",
                    "01:00:02",
                    "11:59:59",
                    "01:23:45.67588833",
                ]
            ),
            pd.Series(
                [
                    bodo.Time(11, 23, 45, 999, precision=9),
                    bodo.Time(1, 23, 45, precision=9),
                    bodo.Time(1, 23, 45, 675, 88, precision=9),
                    bodo.Time(1, 0, 2, precision=9),
                    bodo.Time(11, 59, 59, precision=9),
                    bodo.Time(1, 23, 45, 675, 888, 33, precision=9),
                ]
            ),
            id="time_from_string",
        ),
        pytest.param(
            pd.Series([100, 1000, 1, 4240, 450, 1200]),
            pd.Series(
                [
                    bodo.Time(0, 0, 100, precision=9),
                    bodo.Time(0, 0, 1000, precision=9),
                    bodo.Time(0, 0, 1, precision=9),
                    bodo.Time(1, 10, 40, precision=9),
                    bodo.Time(0, 0, 450, precision=9),
                    bodo.Time(0, 0, 1200, precision=9),
                ]
            ),
            id="time_from_int",
        ),
    ],
)


@to_time_fn_names
@to_time_in_outs
def test_time_to_time(fn_name, input, output, memory_leak_check):
    query = f"select {fn_name}(S) as A FROM table1"
    ctx = {"table1": pd.DataFrame({"S": input})}
    expected_output = pd.DataFrame({"A": output})
    check_query(query, ctx, None, expected_output=expected_output)


@to_time_fn_names
@to_time_in_outs
def test_time_to_time_case(fn_name, input, output, memory_leak_check):
    query = f"select case when B is null then {fn_name}(S) else {fn_name}(S) end as A from table1"
    ctx = {
        "table1": pd.DataFrame({"B": [None, None, None, None, None, None], "S": input})
    }
    expected_output = pd.DataFrame({"A": output}, index=np.arange(6))
    # check_typing_issues=False since column 'B' is intentionally empty
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_typing_issues=False,
    )


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
    check_query(query, ctx, None, expected_output=expected_output)


time_from_parts_fn_names = pytest.mark.parametrize(
    "fn_name", ["timefromparts", "time_from_parts"]
)
time_from_parts_in_outs = pytest.mark.parametrize(
    "args,value",
    [
        pytest.param(
            "1, 2, 3",
            bodo.Time(1, 2, 3, precision=9),
            id="scalar_no_ns",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "1, 2, 3, 4",
            bodo.Time(1, 2, 3, nanosecond=4, precision=9),
            id="scalar_ns",
        ),
    ],
)


@time_from_parts_fn_names
@time_from_parts_in_outs
@pytest.mark.parametrize("wrap_case", [True, False])
def test_time_from_parts(fn_name, wrap_case, args, value, memory_leak_check):
    ctx = {"table1": pd.DataFrame({"B": [None, "", "", None, ""]})}
    default = bodo.Time(0, 0, 0)
    if wrap_case:
        query = f"select case when B is null then {fn_name}(0, 0, 0) else {fn_name}({args}) end as A from table1"
        expected_output = pd.DataFrame({"A": [default, value, value, default, value]})
    else:
        query = f"select {fn_name}({args}) as A from table1"
        expected_output = pd.DataFrame({"A": [value] * 5})
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_typing_issues=False,
        check_dtype=False,
    )


@pytest.fixture
def time_from_parts_bounds_data():
    hours = [0, 12, 25, 23, 24, 0, 0]
    minutes = [100, 0, 30, -1, 0, 0, -3]
    seconds = [0, 12345, 0, 0, 0, 0, 65]
    nanoseconds = [0, 0, 0, 0, 0, -1, 0]
    answer = [
        bodo.Time(1, 40, 0, precision=9),
        bodo.Time(15, 25, 45, precision=9),
        bodo.Time(1, 30, 0, precision=9),
        bodo.Time(22, 59, 0, precision=9),
        bodo.Time(0, 0, 0, precision=9),
        bodo.Time(23, 59, 59, nanosecond=999999999, precision=9),
        bodo.Time(23, 58, 5, precision=9),
    ]
    return hours, minutes, seconds, nanoseconds, answer


def test_time_from_parts_outside_range(time_from_parts_bounds_data, memory_leak_check):
    hours, minutes, seconds, nanoseconds, answer = time_from_parts_bounds_data
    query = f"select time_from_parts(H, M, S, N) as A from table1"
    ctx = {
        "table1": pd.DataFrame(
            {
                "H": hours,
                "M": minutes,
                "S": seconds,
                "N": nanoseconds,
            }
        )
    }
    expected_output = pd.DataFrame({"A": answer})
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


@time_from_parts_fn_names
@pytest.mark.parametrize(
    "args_a, args_b, args_c, args_d, value",
    [
        pytest.param(
            pd.Series([1] * 15),
            pd.Series([2] * 15),
            pd.Series([3] * 15),
            pd.Series(list(range(15))),
            pd.Series(
                [bodo.Time(1, 2, 3, nanosecond=i, precision=9) for i in range(15)]
            ),
            id="time_from_parts_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([1] * 14 + [None], dtype="Int64"),
            pd.Series([2] * 14 + [None], dtype="Int64"),
            pd.Series([3] * 14 + [None], dtype="Int64"),
            pd.Series(list(range(14)) + [None], dtype="Int64"),
            pd.Series(
                [bodo.Time(1, 2, 3, nanosecond=i, precision=9) for i in range(14)]
                + [None]
            ),
            id="time_from_parts_vector_nulls",
        ),
    ],
)
def test_time_from_parts_vector(
    fn_name, args_a, args_b, args_c, args_d, value, memory_leak_check
):
    query = "select time_from_parts(A, B, C, D) as A from table1"
    ctx = {"table1": pd.DataFrame({"A": args_a, "B": args_b, "C": args_c, "D": args_d})}
    expected_output = pd.DataFrame({"A": value}, index=np.arange(len(args_a)))
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "unit, test_fn_type, answer",
    [
        pytest.param(
            "hour",
            "DATE_PART",
            pd.Series([12, 1, 9, 20, 23]),
            id="valid-hour-date_part",
        ),
        pytest.param(
            "minute",
            "MINUTE",
            pd.Series([30, 2, 59, 45, 50]),
            id="valid-minute-regular",
        ),
        pytest.param(
            "second",
            "DATE_PART",
            pd.Series([15, 3, 0, 1, 59]),
            id="valid-second-date_part",
        ),
        pytest.param(
            "millisecond",
            "EXTRACT",
            pd.Series([0, 4, 100, 123, 500]),
            id="valid-millisecond-extract",
        ),
        pytest.param(
            "microsecond",
            "MICROSECOND",
            pd.Series([0, 0, 250, 456, 0]),
            id="valid-microsecond-regular",
        ),
        pytest.param(
            "nanosecond",
            "EXTRACT",
            pd.Series([0, 0, 0, 789, 999]),
            id="valid-nanosecond-extract",
        ),
        pytest.param(
            "day",
            "DATE_PART",
            None,
            id="invalid-day-date_part",
        ),
        pytest.param(
            "dayofyear",
            "DAYOFYEAR",
            None,
            id="invalid-dayofyear-regular",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "dow",
            "EXTRACT",
            None,
            id="invalid-dow-extract",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "week",
            "WEEK",
            None,
            id="invalid-week-regular",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "weekiso",
            "DATE_PART",
            None,
            id="invalid-weekiso-date_part",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "month",
            "EXTRACT",
            None,
            id="invalid-month-extract",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "quarter",
            "DATE_PART",
            None,
            id="invalid-quarter-date_part",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "year",
            "YEAR",
            None,
            id="invalid-year-regular",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_time_extract(unit, answer, test_fn_type, memory_leak_check):
    """Tests EXTRACT and EXTRACT-like functions on time data, checking that
    values larger than HOUR raise an exception"""
    if test_fn_type == "EXTRACT":
        query = f"SELECT EXTRACT({unit} FROM T) AS U FROM table1"
    elif test_fn_type == "DATE_PART":
        query = f"SELECT DATE_PART('{unit}', T) AS U FROM table1"
    else:
        query = f"SELECT {test_fn_type}(T) AS U FROM table1"
    ctx = {
        "table1": pd.DataFrame(
            {
                "T": pd.Series(
                    [
                        bodo.Time(12, 30, 15, precision=0),
                        bodo.Time(1, 2, 3, 4, precision=3),
                        bodo.Time(9, 59, 0, 100, 250, precision=6),
                        bodo.Time(20, 45, 1, 123, 456, 789, precision=9),
                        bodo.Time(23, 50, 59, 500, 0, 999, precision=9),
                    ]
                )
            }
        )
    }
    if answer is None:
        bc = BodoSQLContext(ctx)
        with pytest.raises(
            Exception, match=r"Cannot extract unit \w+ from TIME values"
        ):
            bc.sql(query)
    else:
        expected_output = pd.DataFrame({"U": answer})
        check_query(
            query, ctx, None, expected_output=expected_output, check_dtype=False
        )


def test_timestampdiff_time_columns(time_df, time_part_strings, memory_leak_check):
    """
    Checks that calling TIMESTAMPDIFF on columns behaves as expected
    """
    query = (
        f"SELECT TIMESTAMPDIFF('{time_part_strings}', A, B) as output from table1"
    )
    output = pd.DataFrame(
        {"output": [
            date_sub_unit_time_fn(time_part_strings,
                                  time_df["table1"]["A"][i],
                                  time_df["table1"]["B"][i]
                                  ) for i in range(len(time_df["table1"]["A"]))]}
    )
    check_query(
        query, time_df, None, check_names=False, check_dtype=False, expected_output=output
    )


def test_timestampdiff_time_day_part_handling(time_df, day_part_strings, memory_leak_check):
    """
    Checks that TIMESTAMPDIFF throws an error when a date part is passed in as the unit
    """
    query = (
        f"SELECT TIMESTAMPDIFF('{day_part_strings}', A, B) as output from table1"
    )
    output = pd.DataFrame(
        {"output": []}
    )
    with pytest.raises(Exception, match=
        f"Unsupported TIMESTAMPDIFF unit for TIME input: \"{day_part_strings}\""):
        check_query(
            query,
            time_df,
            None,
            check_names=False,
            check_dtype=False,
            expected_output=output
        )