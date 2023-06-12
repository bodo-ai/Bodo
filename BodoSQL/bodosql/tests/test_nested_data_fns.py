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
        is_out_distributed=False,
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


@pytest.fixture
def array_df():
    return {
        "table1": pd.DataFrame(
            {
                "int_col": pd.Series(
                    [
                        pd.array([4234, -123, 0]),
                        [],
                        None,
                        [86956, -958, -345, 49, 2],
                        [-4, 50, -15, 941, 252, -404, 1399],
                    ] * 4
                ),
                "float_col": pd.Series(
                    [
                        [],
                        [42.34, -1.23, 0.0],
                        None,
                        [8.6956, -0.958, -34.5, 4.9, 20.0],
                        [-1.4, 5.0, -15.15, 9.41, 25.2, -40.4, 0.1399],
                    ] * 4
                ),
                "bool_col": pd.Series(
                    [
                        [True, False, False, True, True, True],
                        [],
                        [False, False, True],
                        None,
                        [False, True, False, True, False],
                    ] * 4
                ),
                "string_col": pd.Series(
                    [
                        ["True", "False", "and", "or", "not", "xor"],
                        ["kgspoas", "0q3e0j", ";.2qe"],
                        None,
                        [],
                        [" ", "^#%&", "VCX:>?", "3ews", "zxcv"],
                    ] * 4
                ),
                "date_col": pd.Series(
                    [
                        [
                            datetime.date(2018, 1, 24),
                            datetime.date(1983, 1, 3),
                            datetime.date(1966, 4, 27),
                            datetime.date(1999, 12, 7),
                            datetime.date(2020, 11, 17),
                            datetime.date(2008, 1, 19),
                        ],
                        [datetime.date(1966, 4, 27), datetime.date(2004, 7, 8)],
                        None,
                        [],
                        [
                            datetime.date(2012, 1, 1),
                            datetime.date(2011, 3, 3),
                            datetime.date(1999, 5, 2),
                            datetime.date(1981, 8, 31),
                            datetime.date(2019, 11, 12),
                        ],
                    ] * 4
                ),
                "time_col": pd.Series(
                    [
                        None,
                        [
                            bodo.Time(12, 0),
                            bodo.Time(1, 1, 3, 1),
                            bodo.Time(2),
                            bodo.Time(15, 0, 50, 10, 100,),
                            bodo.Time(9, 1, 3, 10),
                        ],
                        [],
                        [bodo.Time(6, 11, 3, 1), bodo.Time(12, 30, 42, 64), bodo.Time(4, 5, 6)],
                        [
                            bodo.Time(5, 6, 7, 8),
                            bodo.Time(12, 13, 14, 15, 16, 17),
                            bodo.Time(17, 33, 26, 91, 8, 79),
                            bodo.Time(0, 24, 43, 365, 18, 74),
                            bodo.Time(3, 59, 6, 25, 757, 3),
                            bodo.Time(11, 59, 59, 100, 100, 50),
                        ],
                    ] * 4
                ),
                "timestamp_col": pd.Series(
                    [
                        [],
                        [
                            pd.Timestamp("2021-12-08"),
                            pd.Timestamp("2020-03-14T15:32:52.192548651"),
                            pd.Timestamp("2016-02-28 12:23:33"),
                            pd.Timestamp("2005-01-01"),
                            pd.Timestamp("1999-10-31 12:23:33"),
                            pd.Timestamp("2020-01-01"),
                        ],
                        [pd.Timestamp("2021-10-14"), pd.Timestamp("2017-01-05")],
                        [
                            pd.Timestamp("2017-01-11"),
                            pd.Timestamp("2022-11-06 11:30:15"),
                            pd.Timestamp("2030-01-01 15:23:42.728347"),
                            pd.Timestamp("1981-08-31"),
                            pd.Timestamp("2019-11-12"),
                        ],
                        None,
                    ] * 4
                ),
                "nested_array_col": pd.Series(
                    [
                        [[], [], None],
                        [[1, 2, 3], None, [4, 5, 6]],
                        [[7, 8, 9], [10, 11]],
                        None,
                        [[12, 13, 14, 15, 16], [17, 18]],
                    ] * 4
                ),
            }
        )
    }


@pytest.mark.parametrize(
    "col_name",
    [
        "int_col",
        "float_col",
        "bool_col",
        "string_col",
        "date_col",
        "time_col",
        "timestamp_col",
        "nested_array_col",
    ]
)
def test_array_item_array_boxing(array_df, col_name, memory_leak_check):
    """Test reading ArrayItemArray"""
    query = "SELECT " + col_name + " from table1"
    py_output = pd.DataFrame({"A": array_df["table1"][col_name]})
    if col_name == "timestamp_col":
        for i in range(len(py_output["A"])):
            if py_output["A"][i] is not None:
                py_output["A"][i] = list(
                    map(lambda x: None if x is None else np.datetime64(x, "ns"), py_output["A"][i])
                )
    check_query(
        query,
        array_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "col_name",
    [
        "int_col",
        "float_col",
        "bool_col",
        "string_col",
        "date_col",
        "time_col",
        "timestamp_col",
        "nested_array_col",
    ]
)
@pytest.mark.slow
def test_array_column_type(array_df, col_name, memory_leak_check):
    """Test BodoSQL can infer ARRAY column type correctly"""
    query = "SELECT TO_ARRAY(" + col_name + ") from table1"
    py_output = pd.DataFrame({"A": array_df["table1"][col_name]})
    if col_name == "timestamp_col":
        for i in range(len(py_output["A"])):
            if py_output["A"][i] is not None:
                py_output["A"][i] = list(
                    map(lambda x: None if x is None else np.datetime64(x, "ns"), py_output["A"][i])
                )
    check_query(
        query,
        array_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )
