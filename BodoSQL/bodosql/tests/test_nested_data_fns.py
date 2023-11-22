# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of nested data functions with BodoSQL
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import pytest_slow_unless_codegen
from bodo.utils.typing import BodoError
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ]
)
def use_case(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            [5, 25, 625],
            id="integers",
        ),
        pytest.param(
            [2.71828, 1024.5, -3.1415],
            id="floats",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [
                pd.Timestamp(s)
                for s in [
                    "1999-12-31 23:59:59.99925",
                    "2023-10-31 18:30:00",
                    "2018-4-1",
                ]
            ],
            id="timestamp_ntz",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [
                pd.Timestamp(s, tz="US/Eastern")
                for s in [
                    "2022-2-28",
                    "2021-11-12 13:14:15",
                    "2015-3-14 9:26:53",
                ]
            ],
            id="timestamp_ltz",
        ),
        pytest.param(
            [
                [0],
                [0, 1, 2],
                [1, 2],
            ],
            id="list_integer",
        ),
        pytest.param(
            [
                [["A"]],
                [[]],
                [],
            ],
            id="list_list_string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [
                {"name": "Zuko", "nation": "Fire"},
                {"name": "Katara", "nation": "Water"},
                {"name": "Sokka", "nation": "Water"},
            ],
            id="struct",
        ),
    ],
)
def value_pool(request):
    """
    Returns a list of 3 values of a certain type used to construct
    nested arrays of various patterns using said type.
    """
    return request.param


def test_to_array_scalars(basic_df, memory_leak_check):
    """Test TO_ARRAY works correctly with scalar inputs"""
    query_fmt = "TO_ARRAY({!s})"
    scalars = [
        "123",
        "456.789",
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
            "string": pd.Series([pd.array(["asdafa"], "string[pyarrow]")]),
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
                pd.Series([423, 647, 0, 51, -425] * 4, dtype="int"),
                pd.Series(
                    [
                        pd.array([423]),
                        pd.array([647]),
                        pd.array([0]),
                        pd.array([51]),
                        pd.array([-425]),
                    ]
                    * 4
                ),
            ),
            id="integer",
        ),
        pytest.param(
            (
                pd.Series([4.23, 64.7, None, 0.51, -425.0] * 4),
                pd.Series(
                    [
                        pd.array([4.23]),
                        pd.array([64.7]),
                        None,
                        pd.array([0.51]),
                        pd.array([-425.0]),
                    ]
                    * 4
                ),
            ),
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(["ksef", "$@#%", None, "0.51", "1d$g"] * 4),
                pd.Series(
                    [
                        pd.array(["ksef"], "string[pyarrow]"),
                        pd.array(["$@#%"], "string[pyarrow]"),
                        None,
                        pd.array(["0.51"], "string[pyarrow]"),
                        pd.array(["1d$g"], "string[pyarrow]"),
                    ]
                    * 4
                ),
            ),
            id="string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([True, None, False, False, True] * 4),
                pd.Series(
                    [
                        pd.array([True]),
                        None,
                        pd.array([False]),
                        pd.array([False]),
                        pd.array([True]),
                    ]
                    * 4
                ),
            ),
            id="bool",
            marks=pytest.mark.slow,
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
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        pd.array([bodo.Time(11, 19, 34)]),
                        pd.array([bodo.Time(12, 30, 15)]),
                        pd.array([bodo.Time(12, 34, 56, 78, 12)]),
                        None,
                        pd.array([bodo.Time(12, 34, 56, 78, 12, 34)]),
                    ]
                    * 4
                ),
            ),
            id="time",
            marks=pytest.mark.slow,
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
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        pd.array([datetime.date(2020, 1, 4)]),
                        pd.array([datetime.date(1999, 5, 2)]),
                        pd.array([datetime.date(1970, 1, 1)]),
                        pd.array([datetime.date(2020, 11, 23)]),
                        None,
                    ]
                    * 4
                ),
            ),
            id="date",
            marks=pytest.mark.slow,
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
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        None,
                        pd.array([pd.Timestamp("2020-01-01 22:00:00")]),
                        pd.array([pd.Timestamp("2019-1-24")]),
                        pd.array([pd.Timestamp("2023-7-18")]),
                        pd.array([pd.Timestamp("2020-01-02 01:23:42.728347")]),
                    ]
                    * 4
                ),
            ),
            id="timestamp",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None,
                        [1, 3, 4],
                        [2, 5],
                        [6, None],
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        None,
                        [1, 3, 4],
                        [2, 5],
                        [6, None],
                    ]
                    * 4
                ),
            ),
            id="integer_array",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {"i": 3, "s": 9.9},
                        {"i": 4, "s": 16.3},
                        {"i": 5, "s": 25.0},
                        None,
                        {"i": 6, "s": -36.41},
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        [{"i": 3, "s": 9.9}],
                        [{"i": 4, "s": 16.3}],
                        [{"i": 5, "s": 25.0}],
                        None,
                        [{"i": 6, "s": -36.41}],
                    ]
                    * 4
                ),
            ),
            id="struct",
            marks=pytest.mark.skip(
                reason="TODO: Make coerce_to_array support struct array"
            ),
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
        # Passing this since _use_dict_str_type=True causes gatherv to fail internally
        # and is not needed since the output of the actual test is regular string array
        # (see https://bodo.atlassian.net/browse/BSE-1256)
        use_dict_encoded_strings=False,
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
        # Passing this since _use_dict_str_type=True causes gatherv to fail internally
        # and is not needed since the output of the actual test is regular string array
        # (see https://bodo.atlassian.net/browse/BSE-1256)
        use_dict_encoded_strings=False,
    )


@pytest.mark.parametrize(
    "query, expected",
    [
        pytest.param(
            "SELECT CASE WHEN int_col IS NOT NULL THEN TO_ARRAY(int_col) ELSE TO_ARRAY(int_col) END FROM TABLE1",
            [[1], [2], [3], [4]] * 3 + [None] + [[5], [6], [7]] * 2,
            id="int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT CASE WHEN string_col IS NOT NULL THEN TO_ARRAY(string_col) ELSE TO_ARRAY(string_col) END FROM TABLE1",
            ["1", "2", "3", "4"] * 3 + [None] + ["5", "6", "7"] * 2,
            id="string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT CASE WHEN int_col IS NOT NULL THEN TO_ARRAY(ARRAY_CONSTRUCT(int_col)) ELSE TO_ARRAY(ARRAY_CONSTRUCT(int_col)) END FROM TABLE1",
            [[1], [2], [3], [4]] * 3 + [[None]] + [[5], [6], [7]] * 2,
            id="int_array",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_to_array_case(query, expected, memory_leak_check):
    """tests TO_ARRAY works correctly in a case statement"""
    ctx = {
        "table1": pd.DataFrame(
            {
                "int_col": pd.array(
                    [1, 2, 3, 4] * 3 + [None] + [5, 6, 7] * 2, pd.Int64Dtype()
                ),
                "string_col": pd.array(
                    ["1", "2", "3", "4"] * 3 + [None] + ["5", "6", "7"] * 2,
                    pd.StringDtype(),
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=pd.DataFrame({"EXPR$0": expected}),
        use_dict_encoded_strings=False,
    )


@pytest.fixture
def array_df():
    return {
        "table1": pd.DataFrame(
            {
                "idx_col": pd.Series(range(20)),
                "int_col": pd.Series(
                    [
                        pd.array([4234, -123, 0]),
                        [],
                        None,
                        [86956, -958, -345, 49, 2],
                        [-4, 50, -15, 941, 252, -404, 1399],
                    ]
                    * 4
                ),
                "float_col": pd.Series(
                    [
                        [],
                        [42.34, -1.23, 0.0],
                        None,
                        [8.6956, -0.958, -34.5, 4.9, 20.0],
                        [-1.4, 5.0, -15.15, 9.41, 25.2, -40.4, 0.1399],
                    ]
                    * 4
                ),
                "bool_col": pd.Series(
                    [
                        [True, False, False, True, True, True],
                        [],
                        [False, False, True],
                        None,
                        [False, True, False, True, False],
                    ]
                    * 4
                ),
                "string_col": pd.Series(
                    [
                        ["True", "False", "and", "or", "not", "xor"],
                        ["kgspoas", "0q3e0j", ";.2qe"],
                        None,
                        [],
                        [" ", "^#%&", "VCX:>?", "3ews", "zxcv"],
                    ]
                    * 4
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
                    ]
                    * 4
                ),
                "time_col": pd.Series(
                    [
                        None,
                        [
                            bodo.Time(12, 0),
                            bodo.Time(1, 1, 3, 1),
                            bodo.Time(2),
                            bodo.Time(
                                15,
                                0,
                                50,
                                10,
                                100,
                            ),
                            bodo.Time(9, 1, 3, 10),
                        ],
                        [],
                        [
                            bodo.Time(6, 11, 3, 1),
                            bodo.Time(12, 30, 42, 64),
                            bodo.Time(4, 5, 6),
                        ],
                        [
                            bodo.Time(5, 6, 7, 8),
                            bodo.Time(12, 13, 14, 15, 16, 17),
                            bodo.Time(17, 33, 26, 91, 8, 79),
                            bodo.Time(0, 24, 43, 365, 18, 74),
                            bodo.Time(3, 59, 6, 25, 757, 3),
                            bodo.Time(11, 59, 59, 100, 100, 50),
                        ],
                    ]
                    * 4
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
                    ]
                    * 4
                ),
                "nested_array_col": pd.Series(
                    [
                        [[], [], None],
                        [[1, 2, 3], None, [4, 5, 6]],
                        [[7, 8, 9], [10, 11]],
                        None,
                        [[12, 13, 14, 15, 16], [17, 18]],
                    ]
                    * 4
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
    ],
)
def test_array_item_array_boxing(array_df, col_name, memory_leak_check):
    """Test reading ArrayItemArray"""
    query = "SELECT " + col_name + " from table1"
    py_output = pd.DataFrame({"A": array_df["table1"][col_name]})
    if col_name == "timestamp_col":
        for i in range(len(py_output["A"])):
            if py_output["A"][i] is not None:
                py_output["A"][i] = list(
                    map(
                        lambda x: None if x is None else np.datetime64(x, "ns"),
                        py_output["A"][i],
                    )
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
    ],
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
                    map(
                        lambda x: None if x is None else np.datetime64(x, "ns"),
                        py_output["A"][i],
                    )
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
    "data_values, use_map",
    [
        pytest.param(
            [0, 1, -2, 4],
            False,
            id="integers",
        ),
        pytest.param(
            ["", "Alphabet Soup", "#WEATTACKATDAWN", "Infinity(∞)"],
            False,
            id="strings",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [[0, 1, None, 2], [], [2, 3, 5, 7], [4, 9, 16]],
            False,
            id="nested_array",
            marks=pytest.mark.skip(
                reason="[BSE-1780] TODO: fix array_construct when inputs are multiple arrays"
            ),
        ),
        pytest.param(
            [
                {"i": 3, "s": 9.9},
                {"i": 4, "s": 16.3},
                {"i": 5, "s": 25.0},
                {"i": 6, "s": -36.41},
            ],
            False,
            id="struct",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [{"A": 0, "B": 1}, {"A": 2, "B": 3, "C": 4}, {"B": 5}, {"C": 6, "A": 7}],
            True,
            id="map",
            marks=pytest.mark.skip(
                reason="[BSE-1782] TODO: fix array_construct when inputs are map arrays with simple keys"
            ),
        ),
    ],
)
def test_array_construct(data_values, use_case, use_map, memory_leak_check):
    """
    Test ARRAY_CONSTRUCT works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 4 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then uses them to
    build a column of arrays.
    """
    if any(isinstance(elem, dict) for elem in data_values) and use_case:
        pytest.skip(
            reason="[BSE-1889] TODO: support returning JSON or arrays of JSON in CASE statements"
        )
    if use_case:
        query = "SELECT CASE WHEN C THEN ARRAY_CONSTRUCT(A, B) ELSE ARRAY_CONSTRUCT(B, A) END FROM table1"
    else:
        query = "SELECT ARRAY_CONSTRUCT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else data_values[idx] for idx in L]

    pattern_a = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [None] * 5
    pattern_b = [0, 1, 2, 3, None] * 5
    pattern_answer = [[a, b] for a, b in zip(pattern_a, pattern_b)]
    vals_a = make_vals(pattern_a)
    vals_b = make_vals(pattern_b)
    answer = [make_vals(row) for row in pattern_answer]
    ctx = {
        "table1": pd.DataFrame({"A": vals_a, "B": vals_b, "C": [True] * len(pattern_a)})
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: answer}),
        sort_output=False,
        use_map_arrays=use_map,
        # Can't use check_python because of intricacies of unboxing map arrays
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT ARRAY_TO_STRING(int_col, ',') from table1",
            pd.Series(
                [
                    "4234,-123,0",
                    "",
                    None,
                    "86956,-958,-345,49,2",
                    "-4,50,-15,941,252,-404,1399",
                ]
                * 4
            ),
            id="int",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(float_col, ', ') from table1",
            pd.Series(
                [
                    "",
                    "42.340000, -1.230000, 0.0",
                    None,
                    "8.695600, -0.958000, -34.500000, 4.900000, 20.000000",
                    "-1.400000, 5.000000, -15.150000, 9.410000, 25.200000, -40.400000, 0.139900",
                ]
                * 4
            ),
            id="float",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(bool_col, '.') from table1",
            pd.Series(
                [
                    "true.false.false.true.true.true",
                    "",
                    "false.false.true",
                    None,
                    "false.true.false.true.false",
                ]
                * 4
            ),
            id="bool",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(string_col, '| ') from table1",
            pd.Series(
                [
                    "True| False| and| or| not| xor",
                    "kgspoas| 0q3e0j| ;.2qe",
                    None,
                    "",
                    " | ^#%&| VCX:>?| 3ews| zxcv",
                ]
                * 4
            ),
            id="string",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(date_col, '-') from table1",
            pd.Series(
                [
                    "2018-01-24-1983-01-03-1966-04-27-1999-12-07-2020-11-17-2008-01-19",
                    "1966-04-27-2004-07-08",
                    None,
                    "",
                    "2012-01-01-2011-03-03-1999-05-02-1981-08-31-2019-11-12",
                ]
                * 4
            ),
            id="date",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(time_col, '| ') from table1",
            pd.Series(
                [
                    "true| false| and| or| not| xor",
                    "kgspoas| 0q3e0j| ;.2qe",
                    None,
                    "",
                    " | ^#%&| VCX:>?| 3ews| zxcv",
                ]
                * 4
            ),
            id="time",
            marks=pytest.mark.skip(reason="TODO: Support str() for time type."),
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(timestamp_col, '-*-') from table1",
            pd.Series(
                [
                    "",
                    "2021-12-08T00:00:00-*-2020-03-14T15:32:52.192548-*-2016-02-28T12:23:33"
                    "-*-2005-01-01T00:00:00-*-1999-10-31T12:23:33-*-2020-01-01T00:00:00",
                    "2021-10-14T00:00:00-*-2017-01-05T00:00:00",
                    "2017-01-11T00:00:00-*-2022-11-06T11:30:15-*-2030-01-01T15:23:42.728347"
                    "-*-1981-08-31T00:00:00-*-2019-11-12T00:00:00",
                    None,
                ]
                * 4
            ),
            id="timestamp",
            marks=pytest.mark.skip(reason="TODO: Support TO_VARCHAR for time type."),
        ),
    ],
)
def test_array_to_string_column(array_df, query, answer, memory_leak_check):
    """
    Test ARRAY_TO_STRING works correctly with different data type columns
    """
    py_output = pd.DataFrame({"A": answer})
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
    "input, answer",
    [
        pytest.param(
            "2395",
            "2395",
            id="int",
        ),
        pytest.param(
            "12.482",
            "12.482000",
            id="float",
        ),
        pytest.param(
            "true",
            "true",
            id="bool",
        ),
        pytest.param(
            "'koagri'",
            "koagri",
            id="string",
        ),
        pytest.param(
            "TO_DATE('2019-06-12')",
            "2019-06-12",
            id="date",
        ),
        pytest.param(
            "TO_TIME('16:47:23')",
            "16:47:23",
            id="time",
            marks=pytest.mark.skip(reason="TODO: Support str() for time type."),
        ),
        pytest.param(
            "TO_TIMESTAMP('2023-06-13 16:49:50')",
            "2023-06-13 16:49:50",
            id="timestamp",
        ),
    ],
)
def test_array_to_string_scalar(basic_df, input, answer, memory_leak_check):
    """
    Test ARRAY_TO_STRING works correctly with different data type scalars
    """
    query = f"SELECT ARRAY_TO_STRING(TO_ARRAY({input}), ', ')"
    py_output = pd.DataFrame({"A": pd.Series([answer])})
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


def test_arrays_overlap(value_pool, use_case, memory_leak_check):
    """
    Test ARRAYS_OVERLAP works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    if use_case:
        query = (
            "SELECT I, CASE WHEN ARRAYS_OVERLAP(A, B) THEN 'Y' ELSE 'N' END FROM table1"
        )
    else:
        query = "SELECT I, ARRAYS_OVERLAP(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = (
        [[0], [1], [0], [1]] * 25
        + [[0, 1, 2]] * 3
        + [[0, 1, None, 0]] * 3
        + [None] * 3
        + [[0, 2, 0, 2]] * 3
    )
    pattern_b = [[0], [0], [1], [1]] * 25 + [
        [2, None],
        [1],
        [],
        [2, None, 2],
        [2, 2, 2, 2, 2, 2],
        [None, None, None],
        None,
        [0],
        [None],
        [0, 2, 0, 2],
        [1, 1, 1, 1],
        [1, None, 1, 2],
    ]
    vals_a = [make_vals(row) for row in pattern_a]
    vals_b = [make_vals(row) for row in pattern_b]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    answer = pd.Series(
        [True, False, False, True] * 25
        + [True, True, False]
        + [True, False, True]
        + [None, None, None]
        + [True, False, True]
    )
    if use_case:
        answer = answer.apply(lambda x: "Y" if x else "N")
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: list(range(len(answer))), 1: answer}),
    )


def test_array_contains(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_CONTAINS works correctly with different data type columns
    and with/without case statements.
    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    if use_case:
        query = "SELECT I, CASE WHEN I IS NULL THEN ARRAY_CONTAINS(A, B) ELSE ARRAY_CONTAINS(A, B) END FROM table1"
    else:
        query = "SELECT I, ARRAY_CONTAINS(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [0] * 40 + [0, 1, 2, None] * 3
    pattern_b = [[0]] * 40 + [[1, None]] * 4 + [[], None, None, []] + [[0, 2]] * 4
    vals_a = make_vals(pattern_a)
    vals_b = [make_vals(row) for row in pattern_b]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    expected = pd.Series(
        [True] * 40
        + [False, True, False, True]
        + [False, None, None, False]
        + [True, False, True, False],
        dtype=pd.BooleanDtype(),
    )
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame(
            {"I": list(range(len(vals_a))), "EXPR$1": expected}
        ),
    )


def test_array_position(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_POSITION works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE ARRAY_POSITION(A, B) END FROM table1"
    else:
        query = "SELECT I, ARRAY_POSITION(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [0] * 40 + [0, 1, 2, None] * 4
    pattern_b = (
        [[0]] * 40
        + [[0, 1, 2, None]] * 4
        + [[0, None] * 3] * 4
        + [[]] * 4
        + [[2, 1, None, 1, 2, None, None, 0]] * 4
    )
    vals_a = make_vals(pattern_a)
    vals_b = [make_vals(row) for row in pattern_b]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    answer = pd.Series(
        [0] * 40
        + [0, 1, 2, 3]
        + [0, None, None, 1]
        + [None, None, None, None]
        + [7, 1, 0, 2],
        dtype=pd.Int32Dtype(),
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: list(range(len(vals_a))), 1: answer}),
    )


def test_array_except(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_EXCEPT works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_EXCEPT. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.

    For case statements, uses the size of the output since BodoSQL is currently
    unable to infer the inner array dtype for case statements.
    """
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_EXCEPT(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_EXCEPT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [[0, 0, 1, 0]] * 40 + [
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 0, None, None, 0, 1, None],
        [0, 0, None, None, 0, 1, None],
        [0, 0, None, None, 0, 1, None],
        [0, 0, None, None, 0, 1, None],
    ]
    pattern_b = [[0, 1]] * 40 + [
        [0],
        None,
        [0, 1, 2],
        [],
        [0, 0, 0, 1, 1, 2],
        [1, None],
        [1, None, 1, None, 1],
        [0, 1, None] * 5,
        [0, 2] * 5,
    ]
    answer_pattern = [[0, 0]] * 40 + [
        [1, 2, 0, 1, 2, 0, 1, 2],
        None,
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [2, 1, 2],
        [0, 0, None, 0, None],
        [0, 0, 0, None],
        [],
        [None, None, 1, None],
    ]
    vals_a = [make_vals(row) for row in pattern_a]
    vals_b = [make_vals(row) for row in pattern_b]
    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = [make_vals(row) for row in answer_pattern]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: list(range(len(vals_a))), 1: answer}),
        sort_output=False,
    )


def test_array_intersection(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_INTERSECTION works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_INTERSECTION. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.

    For case statements, uses the size of the output since BodoSQL is currently
    unable to infer the inner array dtype for case statements.
    """
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_INTERSECTION(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_INTERSECTION(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [[0, 0, 1, 0, 1]] * 40 + [
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 0, None, None, 0, 1, None],
        [0, 0, None, None, 0, 1, None],
        [0, 0, None, None, 0, 1, None],
    ]
    pattern_b = [[0, 1, 1, 1]] * 40 + [
        [],
        None,
        [0, 1, 1, 2, 2, 2, None, None, None, None],
        [],
        None,
        [0, 1, 1, 2, 2, 2, None, None, None, None, None],
    ]
    answer_pattern = [[0, 1, 1]] * 40 + [
        [],
        None,
        [0, 1, 2, 1, 2, 2],
        [],
        None,
        [0, None, None, 1, None],
    ]
    vals_a = [make_vals(row) for row in pattern_a]
    vals_b = [make_vals(row) for row in pattern_b]
    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = [make_vals(row) for row in answer_pattern]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: list(range(len(vals_a))), 1: answer}),
        sort_output=False,
    )


def test_array_cat(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_CAT works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_CAT. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.

    For case statements, uses the size of the output since BodoSQL is currently
    unable to infer the inner array dtype for case statements.
    """
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_CAT(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_CAT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [[0, None, 1]] * 40 + [
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [],
        [],
        None,
        [None],
    ]
    pattern_b = [[0, 2]] * 40 + [
        [],
        None,
        [None],
        [0, 0, 1, None, 0],
        [],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    answer_pattern = [[0, None, 1, 0, 2]] * 40 + [
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        None,
        [0, 0, 1, 0, 0, 1, 2, 1, 0, None],
        [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, None, 0],
        [],
        [0, 1],
        None,
        [None, 0, 1],
    ]
    vals_a = [make_vals(row) for row in pattern_a]
    vals_b = [make_vals(row) for row in pattern_b]
    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = [make_vals(row) for row in answer_pattern]
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": list(range(len(vals_a))),
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: list(range(len(vals_a))), 1: answer}),
        sort_output=False,
    )


def test_array_compact(value_pool, use_case, memory_leak_check):
    """
    Test ARRAY_COMPACT works correctly with different data type columns
    """
    if use_case:
        query = "SELECT CASE WHEN A IS NULL THEN ARRAY_COMPACT(A) ELSE ARRAY_COMPACT(A) END FROM table1"
    else:
        query = "SELECT ARRAY_COMPACT(A) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    data_pattern = [
        [None, 0, 1, None, 2, None, None],
        [0, None, None, 2],
        [None, None, None],
        [],
        None,
    ] * 2
    expected_pattern = [[0, 1, 2], [0, 2], [], [], None] * 2
    data = [make_vals(row) for row in data_pattern]
    expected = [make_vals(row) for row in expected_pattern]
    ctx = {"table1": pd.DataFrame({"A": data})}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=pd.DataFrame({"EXPR$0": expected}),
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT ARRAY_SIZE(int_col) FROM table1",
            pd.Series([3, 0, None, 5, 7] * 4),
            id="int",
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(float_col) FROM table1",
            pd.Series([0, 3, None, 5, 7] * 4),
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(bool_col) FROM table1",
            pd.Series(
                [
                    6,
                    0,
                    3,
                    None,
                    5,
                ]
                * 4
            ),
            id="bool",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(string_col) FROM table1",
            pd.Series(
                [
                    6,
                    3,
                    None,
                    0,
                    5,
                ]
                * 4
            ),
            id="string",
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(date_col) FROM table1",
            pd.Series(
                [
                    6,
                    2,
                    None,
                    0,
                    5,
                ]
                * 4
            ),
            id="date",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(time_col) FROM table1",
            pd.Series(
                [
                    None,
                    5,
                    0,
                    3,
                    6,
                ]
                * 4
            ),
            id="time",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(timestamp_col) FROM table1",
            pd.Series(
                [
                    0,
                    6,
                    2,
                    5,
                    None,
                ]
                * 4
            ),
            id="timestamp",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT ARRAY_SIZE(nested_array_col) FROM table1",
            pd.Series(
                [
                    3,
                    3,
                    2,
                    None,
                    2,
                ]
                * 4
            ),
            id="nested_array",
        ),
        pytest.param(
            "SELECT CASE WHEN idx_col < 10 THEN ARRAY_SIZE(int_col) ELSE NULL END FROM table1",
            pd.Series(
                [
                    3,
                    0,
                    None,
                    5,
                    7,
                ]
                * 2
                + [None] * 10
            ),
            id="case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT CASE WHEN idx_col < 10 THEN ARRAY_SIZE(nested_array_col) ELSE NULL END FROM table1",
            pd.Series(
                [
                    3,
                    3,
                    2,
                    None,
                    2,
                ]
                * 2
                + [None] * 10
            ),
            id="case_nested",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_array_size_column(array_df, query, answer, memory_leak_check):
    """
    Test ARRAY_SIZE works correctly with different data type columns
    """
    py_output = pd.DataFrame({"A": answer})
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
    "input",
    [
        pytest.param(
            "2395",
            id="int",
        ),
        pytest.param(
            "12.482",
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "True",
            id="bool",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "'koagri'",
            id="string",
        ),
        pytest.param(
            "TO_DATE('2019-06-12')",
            id="date",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TO_TIME('16:47:23')",
            id="time",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TO_TIMESTAMP('2023-06-13 16:49:50')",
            id="timestamp",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TO_ARRAY(1)",
            id="nested_array",
        ),
    ],
)
def test_array_size_scalar(basic_df, input, memory_leak_check):
    """
    Test ARRAY_SIZE works correctly with different data type scalars
    """
    query = f"SELECT ARRAY_SIZE(TO_ARRAY({input}))"
    py_output = pd.DataFrame({"A": pd.Series(1)})
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ("1", 1),
        ),
        pytest.param(("'1'", "1"), marks=pytest.mark.slow),
        pytest.param(("True", True), marks=pytest.mark.slow),
    ],
)
def test_array_index_scalar(args, memory_leak_check):
    """
    Test Array indexing works correctly with different data type scalars
    """

    expr = args[0]

    query = f"SELECT CASE WHEN ARRAY_CONSTRUCT({expr})[0] = {expr} THEN ARRAY_CONSTRUCT({expr})[0] ELSE NULL END as out_col FROM TABLE1"
    ctx = {"table1": pd.DataFrame({"unused_col": list(range(10))})}
    py_output = pd.DataFrame({"out_col": [args[1]] * 10})

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("1", 1), marks=pytest.mark.slow),
        pytest.param(("'hello world'", "hello world"), marks=pytest.mark.slow),
        pytest.param(
            ("TIMESTAMP '1969-07-20 20:17:40'", pd.Timestamp("1969-07-20 20:17:40")),
        ),
    ],
)
def test_array_index_column(args, memory_leak_check):
    """
    Test Array indexing works correctly with different data type
    """

    expr = args[0]

    query = f"SELECT arr_col[0] as out_col from (SELECT ARRAY_CONSTRUCT({expr}) as arr_col FROM TABLE1)"
    ctx = {"table1": pd.DataFrame({"unused_col": list(range(10))})}
    py_output = pd.DataFrame({"out_col": [args[1]] * 10})

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow()
def test_array_index_stress_test(memory_leak_check):
    """
    Stress test for Test Array indexing works correctly with different data type
    """

    query = f"""
    SELECT
        ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(0)[0])[ARRAY_CONSTRUCT(0)[0]])[0] as out_col_1,
        arr_col[arr_col[arr_col[arr_col[arr_col[0]]]] + 1] as out_col_2
    from
    (SELECT ARRAY_CONSTRUCT(int_col_0, int_col_1, int_col_2) as arr_col FROM TABLE1)
    """

    ctx = {
        "table1": pd.DataFrame(
            {
                "int_col_0": [0] * 12,
                "int_col_1": list(range(12)),
                "int_col_2": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "out_col_1": [0] * 12,
            "out_col_2": list(range(12)),
        }
    )

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow()
@pytest.mark.skip("https://bodo.atlassian.net/browse/BSE-2111")
def test_array_index_empty(memory_leak_check):
    """
    Test that indexing into an empty array throws a reasonable error
    """

    query = f"""
    SELECT
        arr_col[0] as out_col
    from
    (SELECT ARRAY_CONSTRUCT() as arr_col FROM TABLE1)
    """

    ctx = {
        "table1": pd.DataFrame(
            {
                "input_col": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "out_col": pd.Series([None] * 12, dtype=pd.Int64Dtype()),
        }
    )

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow()
def test_array_index_out_of_bounds(memory_leak_check):
    """
    Test that indexing out of bounds throws a reasonable error.
    """

    query = f"""
    SELECT
        arr_col[2] as out_col
    from
    (SELECT ARRAY_CONSTRUCT(input_col) as arr_col FROM TABLE1)
    """

    ctx = {
        "table1": pd.DataFrame(
            {
                "input_col": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "out_col": pd.Series([None] * 12, dtype=pd.Int64Dtype()),
        }
    )

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )
