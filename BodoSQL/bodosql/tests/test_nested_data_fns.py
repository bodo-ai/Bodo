"""
Test correctness of nested data functions with BodoSQL
"""

import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    pytest_mark_one_rank,
    pytest_slow_unless_codegen,
)
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
            ([5, 25, 625], pa.int16()),
            id="integers",
        ),
        pytest.param(
            ([2.71828, 1024.5, -3.1415], pa.float64()),
            id="floats",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                [
                    pd.Timestamp(s)
                    for s in [
                        "1999-12-31 23:59:59.99925",
                        "2023-10-31 18:30:00",
                        "2018-4-1",
                    ]
                ],
                pa.timestamp("ns"),
            ),
            id="timestamp_ntz",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                [
                    pd.Timestamp(s, tz="US/Eastern")
                    for s in [
                        "2022-2-28",
                        "2021-11-12 13:14:15",
                        "2015-3-14 9:26:53",
                    ]
                ],
                pa.timestamp("ns", "US/Eastern"),
            ),
            id="timestamp_ltz",
        ),
        pytest.param(
            (
                [
                    [0],
                    [0, 1, 2],
                    [1, 2],
                ],
                pa.large_list(pa.int8()),
            ),
            id="list_integer",
        ),
        pytest.param(
            (
                [
                    [["A"]],
                    [[]],
                    [],
                ],
                pa.large_list(pa.large_list(pa.string())),
            ),
            id="list_list_string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                [
                    {"name": "Zuko", "nation": "Fire"},
                    {"name": "Katara", "nation": "Water"},
                    {"name": "Sokka", "nation": "Water"},
                ],
                pa.struct(
                    [pa.field("name", pa.string()), pa.field("nation", pa.string())]
                ),
            ),
            id="struct",
        ),
        pytest.param(
            (
                [
                    {"A": 65},
                    {"B": 66, "?": None, "a": 97},
                    {},
                ],
                pa.map_(pa.string(), pa.int32()),
            ),
            id="map",
        ),
    ],
)
def array_values(request):
    """
    Returns a list of 3 values of a certain type used to construct
    nested arrays of various patterns using said type, as well
    as the corresponding pyarrow type.
    """
    return request.param


# Used in the fixture below
mixed_struct_type = pa.struct(
    [
        pa.field("X", pa.string()),
        pa.field("Y", pa.large_list(pa.float32())),
        pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
        pa.field(
            "W",
            pa.struct(
                pa.struct(
                    [
                        pa.field(
                            "A",
                            pa.int32(),
                            pa.field("B", pa.string()),
                        )
                    ]
                )
            ),
        ),
    ]
)


def wrapped_in_id_typ(typ):
    return pa.struct(
        [
            pa.field("id", pa.int32()),
            pa.field("value", typ),
        ]
    )


nested_nested_df = pd.DataFrame(
    {
        "A": pd.array(
            [
                {
                    "id": 1,
                    "value": {
                        "A": {
                            "X": "AB",
                            "Y": [1.1, 2.2],
                            "Z": [[1], None, [3, None]],
                            "W": {"A": 1, "B": "A"},
                        },
                        "B": {
                            "X": "C",
                            "Y": [1.1],
                            "Z": [[11], None],
                            "W": {"A": 1, "B": "ABC"},
                        },
                    },
                },
                {
                    "id": 1,
                    "value": {
                        "A": {
                            "X": "D",
                            "Y": [4.0, np.nan],
                            "Z": [[1], None],
                            "W": {"A": 1, "B": ""},
                        },
                        "B": {
                            "X": "VFD",
                            "Y": [1.2],
                            "Z": [[], [3, 1]],
                            "W": {"A": 1, "B": "AA"},
                        },
                    },
                },
                None,
            ]
            * 2,
            dtype=pd.ArrowDtype(
                wrapped_in_id_typ(
                    pa.struct(
                        [
                            pa.field("A", mixed_struct_type),
                            pa.field("B", mixed_struct_type),
                        ]
                    )
                )
            ),
        ),
        "B": pd.Series([None, "random_key", None, "id", "value", None]),
    }
)

"""
These list contain pytest params containing a table with a two columns,
"A" that is a SQL object array, and "B",
which is a string column.
the SQL object type in A will always have two fields, "id" and "value".
id is always an integer, but value may vary between the returned tables.
"""

# List of data types represented as a pa map array
map_object_params = [
    pytest.param(
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            {"id": 1, "value": 10},
                            {"id": 1, "value": 20},
                        ],
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                    ),
                    "B": pd.Series(["id", "value"]),
                }
            ),
        },
        id="int_map",
    ),
    pytest.param(
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            {"id": 1, "value": 10},
                            None,
                            {"id": 1, "value": -10},
                            {"id": 1, "value": 0},
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                    ),
                    "B": pd.Series(["random_key", "id", "value", None] * 2),
                }
            )
        },
        id="map_array",
    ),
]

# List of data types represented as a pa struct array
struct_object_params = [
    pytest.param(
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            {"id": 1, "value": 10},
                            {"id": 1, "value": 20},
                        ],
                        dtype=pd.ArrowDtype(wrapped_in_id_typ(pa.int32())),
                    ),
                    "B": pd.Series(["id", "value"]),
                }
            ),
        },
        id="int_struct",
    ),
    pytest.param(
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            {
                                "id": 1,
                                "value": pd.Timestamp("2020-01-01"),
                            },
                            {
                                "id": 1,
                                "value": pd.Timestamp("2020-01-02"),
                            },
                        ],
                        dtype=pd.ArrowDtype(wrapped_in_id_typ(pa.timestamp("ns"))),
                    ),
                    "B": pd.Series(["id", "value"]),
                }
            ),
        },
        id="timestamp",
    ),
    pytest.param(
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            {
                                "id": 1,
                                "value": {
                                    "X": "VFD",
                                    "Y": [1.2],
                                    "Z": [[], [3, 1]],
                                    "W": {"A": 1, "B": "AA"},
                                },
                            },
                            {
                                "id": 1,
                                "value": {
                                    "X": "D",
                                    "Y": [4.0, np.nan],
                                    "Z": [[1], None],
                                    "W": {"A": 1, "B": ""},
                                },
                            },
                        ],
                        dtype=pd.ArrowDtype(wrapped_in_id_typ(mixed_struct_type)),
                    ),
                    "B": pd.Series(["id", "value"]),
                }
            ),
        },
        id="nested_nested_struct",
    ),
    pytest.param(
        {
            "TABLE1": nested_nested_df,
        },
        id="nested_nested_struct",
        marks=pytest.mark.slow,
    ),
]


@pytest.fixture(params=map_object_params + struct_object_params)
def sql_object_array_values(request):
    "Tables containing sql object types that can either be struct or map internally"
    return request.param


@pytest.fixture(params=map_object_params)
def map_array_values(request):
    "Tables containing sql object types that are map internally"
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            ([0, 1, -2, 4], pd.Int32Dtype()),
            id="integers",
        ),
        pytest.param(
            (["", "Alphabet Soup", "#WEATTACKATDAWN", "Infinity(∞)"], None),
            id="strings",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                [[0, 1, None, 2], [], [2, 3, 5, 7], [4, 9, 16]],
                pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="nested_array",
        ),
        pytest.param(
            (
                [
                    {"i": 3, "s": 9.9},
                    {"i": 4, "s": 16.3},
                    {"i": 5, "s": 25.0},
                    {"i": 6, "s": -36.41},
                ],
                pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("i", pa.int8()),
                            pa.field("s", pa.float64()),
                        ]
                    )
                ),
            ),
            id="struct",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                [
                    {"A": 0, "B": 1},
                    {"A": 2, "B": 3, "C": 4},
                    {"B": 5},
                    {"C": 6, "A": 7},
                ],
                pd.ArrowDtype(pa.map_(pa.large_string(), pa.int16())),
            ),
            id="map",
        ),
    ],
)
def data_values(request):
    """
    Returns a list of 4 values of a certain type used to construct
    nested arrays of various patterns using said type, as well
    as the corresponding pyarrow type.
    Only used by test_array_construct and test_array_construct_compact
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
            "INT": pd.Series([pd.array([123])]),
            "FLOAT": pd.Series([pd.array([456.789])]),
            "STRING": pd.Series([pd.array(["asdafa"], "string[pyarrow]")]),
            "BOOL": pd.Series([pd.array([True])]),
            "TIME": pd.Series([pd.array([bodo.types.Time(5, 34, 51)])]),
            "DATE": pd.Series([pd.array([datetime.date(2023, 5, 18)])]),
            "TIMESTAMP": pd.Series(
                [
                    pd.array(
                        [pd.Timestamp("2024-06-29 17:00:00")], dtype="datetime64[ns]"
                    )
                ]
            ),
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


def is_array_dtype(pa_dtype):
    """
    Returns true if the input pyarrow dtype is a list dtype.
    """
    return pa_dtype.id in (pa.list_(pa.int32()).id, pa.large_list(pa.int32()).id)


def is_struct_dtype(pa_dtype):
    """
    Returns true if the input pyarrow dtype is a struct dtype.
    """
    return pa_dtype.id == pa.struct([]).id


def is_map_dtype(pa_dtype):
    """
    Returns true if the input pyarrow dtype is a map dtype.
    """
    return pa_dtype.id == pa.map_(pa.int32(), pa.int32()).id


def ignore_scalar_dtype(pa_dtype):
    """
    Returns a pyarrow dtype if it is a semi-structured dtype, otherwise returns null.
    Does so by comparing the id of the dtype to the id of lists/structs/maps with
    dummy inner dtypes.
    """
    if is_array_dtype(pa_dtype) or is_struct_dtype(pa_dtype) or is_map_dtype(pa_dtype):
        return pd.ArrowDtype(pa_dtype)
    return None


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
                        bodo.types.Time(11, 19, 34),
                        bodo.types.Time(12, 30, 15),
                        bodo.types.Time(12, 34, 56),
                        None,
                        bodo.types.Time(12, 34, 56),
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        pd.array([bodo.types.Time(11, 19, 34)]),
                        pd.array([bodo.types.Time(12, 30, 15)]),
                        pd.array([bodo.types.Time(12, 34, 56)]),
                        None,
                        pd.array([bodo.types.Time(12, 34, 56)]),
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
                    * 4,
                    dtype="datetime64[ns]",
                ),
                pd.Series(
                    [
                        None,
                        pd.array(
                            [pd.Timestamp("2020-01-01 22:00:00")],
                            dtype="datetime64[ns]",
                        ),
                        pd.array([pd.Timestamp("2019-1-24")], dtype="datetime64[ns]"),
                        pd.array([pd.Timestamp("2023-7-18")], dtype="datetime64[ns]"),
                        pd.array(
                            [pd.Timestamp("2020-01-02 01:23:42.728347")],
                            dtype="datetime64[ns]",
                        ),
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
    ctx = {"TABLE1": pd.DataFrame({"A": data})}
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
    ctx = {"TABLE1": pd.DataFrame({"A": data})}
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
            pd.Series(
                [["1"], ["2"], ["3"], ["4"]] * 3 + [None] + [["5"], ["6"], ["7"]] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
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
        "TABLE1": pd.DataFrame(
            {
                "INT_COL": pd.array(
                    [1, 2, 3, 4] * 3 + [None] + [5, 6, 7] * 2, pd.Int64Dtype()
                ),
                "STRING_COL": pd.array(
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
        "TABLE1": pd.DataFrame(
            {
                "IDX_COL": pd.Series(range(20)),
                "INT_COL": pd.Series(
                    [
                        pd.array([4234, -123, 0]),
                        [],
                        None,
                        [86956, -958, -345, 49, 2],
                        [-4, 50, -15, 941, 252, -404, 1399],
                    ]
                    * 4
                ),
                "FLOAT_COL": pd.Series(
                    [
                        [],
                        [42.34, -1.23, 0.0],
                        None,
                        [8.6956, -0.958, -34.5, 4.9, 20.0],
                        [-1.4, 5.0, -15.15, 9.41, 25.2, -40.4, 0.1399],
                    ]
                    * 4
                ),
                "BOOL_COL": pd.Series(
                    [
                        [True, False, False, True, True, True],
                        [],
                        [False, False, True],
                        None,
                        [False, True, False, True, False],
                    ]
                    * 4
                ),
                "STRING_COL": pd.Series(
                    [
                        ["True", "False", "and", "or", "not", "xor"],
                        ["kgspoas", "0q3e0j", ";.2qe"],
                        None,
                        [],
                        [" ", "^#%&", "VCX:>?", "3ews", "zxcv"],
                    ]
                    * 4
                ),
                "DATE_COL": pd.Series(
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
                "TIME_COL": pd.Series(
                    [
                        None,
                        [
                            bodo.types.Time(12, 0),
                            bodo.types.Time(1, 1, 3),
                            bodo.types.Time(2),
                            bodo.types.Time(
                                15,
                                0,
                                50,
                            ),
                            bodo.types.Time(9, 1, 3),
                        ],
                        [],
                        [
                            bodo.types.Time(6, 11, 3),
                            bodo.types.Time(12, 30, 42),
                            bodo.types.Time(4, 5, 6),
                        ],
                        [
                            bodo.types.Time(5, 6, 7),
                            bodo.types.Time(12, 13, 14),
                            bodo.types.Time(17, 33, 26),
                            bodo.types.Time(0, 24, 43),
                            bodo.types.Time(3, 59, 6),
                            bodo.types.Time(11, 59, 59),
                        ],
                    ]
                    * 4
                ),
                "TIMESTAMP_COL": pd.Series(
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
                    * 4,
                    dtype=pd.ArrowDtype(pa.list_(pa.timestamp("ns"))),
                ),
                "NESTED_ARRAY_COL": pd.Series(
                    [
                        [[], [], None],
                        [[1, 2, 3], None, [4, 5, 6]],
                        [[7, 8, 9], [10, 11]],
                        None,
                        [[12, 13, 14, 15, 16], [17, 18]],
                    ]
                    * 4,
                    dtype=pd.ArrowDtype(pa.list_(pa.list_(pa.int64()))),
                ),
            }
        )
    }


@pytest.mark.parametrize(
    "col_name",
    [
        "INT_COL",
        "FLOAT_COL",
        "BOOL_COL",
        "STRING_COL",
        "DATE_COL",
        "TIME_COL",
        "TIMESTAMP_COL",
        "NESTED_ARRAY_COL",
    ],
)
def test_array_item_array_boxing(array_df, col_name, memory_leak_check):
    """Test reading ArrayItemArray"""
    query = "SELECT " + col_name + " from table1"
    py_output = pd.DataFrame({"A": array_df["TABLE1"][col_name]})

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
        "INT_COL",
        "FLOAT_COL",
        "BOOL_COL",
        "STRING_COL",
        "DATE_COL",
        "TIME_COL",
        "TIMESTAMP_COL",
        "NESTED_ARRAY_COL",
    ],
)
@pytest.mark.slow
def test_array_column_type(array_df, col_name, memory_leak_check):
    """Test BodoSQL can infer ARRAY column type correctly"""
    query = "SELECT TO_ARRAY(" + col_name + ") from table1"
    py_output = pd.DataFrame({"A": array_df["TABLE1"][col_name]})

    check_query(
        query,
        array_df,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


def test_array_construct(data_values, use_case, memory_leak_check):
    """
    Test ARRAY_CONSTRUCT works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 4 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then uses them to
    build a column of arrays.
    """
    data_values, dtype = data_values

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
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(vals_a, dtype=dtype),
                "B": pd.Series(vals_b, dtype=dtype),
                "C": [True] * len(pattern_a),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: answer}),
        sort_output=False,
        convert_columns_to_pandas=True,
    )


def test_array_construct_compact(data_values, use_case, memory_leak_check):
    """
    Test ARRAY_CONSTRUCT_COMPACT works correctly with different data
    type columns and with/without case statements.

    Takes in a list of 4 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then uses them to
    build a column of arrays.
    """
    data_values, dtype = data_values

    if use_case:
        query = "SELECT CASE WHEN C THEN ARRAY_CONSTRUCT_COMPACT(A, B) ELSE ARRAY_CONSTRUCT_COMPACT(B, A) END FROM table1"
    else:
        query = "SELECT ARRAY_CONSTRUCT_COMPACT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else data_values[idx] for idx in L]

    def make_ans(A, B):
        ret = []
        if A is not None:
            ret.append(A)
        if B is not None:
            ret.append(B)
        return ret

    pattern_a = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [None] * 5
    pattern_b = [0, 1, 2, 3, None] * 5
    pattern_answer = [make_ans(a, b) for a, b in zip(pattern_a, pattern_b)]
    vals_a = make_vals(pattern_a)
    vals_b = make_vals(pattern_b)
    answer = [make_vals(row) for row in pattern_answer]
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(vals_a, dtype=dtype),
                "B": pd.Series(vals_b, dtype=dtype),
                "C": [True] * len(pattern_a),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"EXPR$0": answer}),
        sort_output=False,
        convert_columns_to_pandas=True,
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
                    None,
                    "12:00:00| 01:01:03| 02:00:00| 15:00:50| 09:01:03",
                    "",
                    "06:11:03| 12:30:42| 04:05:06",
                    "05:06:07| 12:13:14| 17:33:26| 00:24:43| 03:59:06| 11:59:59",
                ]
                * 4
            ),
            id="time",
        ),
        pytest.param(
            "SELECT ARRAY_TO_STRING(timestamp_col, '-*-') from table1",
            pd.Series(
                [
                    "",
                    "2021-12-08 00:00:00-*-2020-03-14 15:32:52.192548651-*-2016-02-28 12:23:33-*-2005-01-01 00:00:00-*-1999-10-31 12:23:33-*-2020-01-01 00:00:00",
                    "2021-10-14 00:00:00-*-2017-01-05 00:00:00",
                    "2017-01-11 00:00:00-*-2022-11-06 11:30:15-*-2030-01-01 15:23:42.728347-*-1981-08-31 00:00:00-*-2019-11-12 00:00:00",
                    None,
                ]
                * 4
            ),
            id="timestamp",
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


def test_arrays_overlap(array_values, use_case, memory_leak_check):
    """
    Test ARRAYS_OVERLAP works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    value_pool, dtype = array_values
    if use_case:
        query = (
            "SELECT I, CASE WHEN ARRAYS_OVERLAP(A, B) THEN 'Y' ELSE 'N' END FROM table1"
        )
    else:
        query = "SELECT I, ARRAYS_OVERLAP(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

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
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    ctx = {
        "TABLE1": pd.DataFrame(
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


def test_array_contains(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_CONTAINS works correctly with different data type columns
    and with/without case statements.
    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT I, CASE WHEN I IS NULL THEN ARRAY_CONTAINS(A, B) ELSE ARRAY_CONTAINS(A, B) END FROM table1"
    else:
        query = "SELECT I, ARRAY_CONTAINS(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

    pattern_a = [0] * 40 + [0, 1, 2, None] * 3
    pattern_b = [[0]] * 40 + [[1, None]] * 4 + [[], None, None, []] + [[0, 2]] * 4
    dtype_a = ignore_scalar_dtype(dtype)
    vals_a = pd.array(make_vals(pattern_a), dtype=dtype_a)
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {"I": list(range(len(vals_a))), "A": vals_a, "B": vals_b}
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


def test_array_position(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_POSITION works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then compares them
    using ARRAYS_OVERLAP. Since the values are always placed in the columns
    in the same permutations, the answers should always be the same.
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE ARRAY_POSITION(A, B) END FROM table1"
    else:
        query = "SELECT I, ARRAY_POSITION(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

    pattern_a = [0] * 40 + [0, 1, 2, None] * 4
    pattern_b = (
        [[0]] * 40
        + [[0, 1, 2, None]] * 4
        + [[0, None] * 3] * 4
        + [[]] * 4
        + [[2, 1, None, 1, 2, None, None, 0]] * 4
    )
    dtype_a = ignore_scalar_dtype(dtype)
    vals_a = pd.array(make_vals(pattern_a), dtype=dtype_a)
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    ctx = {
        "TABLE1": pd.DataFrame(
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


def test_array_except(array_values, use_case, memory_leak_check):
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
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_EXCEPT(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_EXCEPT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

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
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )

    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = pd.array(
            [make_vals(row) for row in answer_pattern],
            dtype=pd.ArrowDtype(pa.large_list(dtype)),
        )
    ctx = {
        "TABLE1": pd.DataFrame(
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


def test_array_remove(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_REMOVE works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_EXCEPT. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT CASE WHEN A IS NULL THEN ARRAY_REMOVE(A, B) ELSE ARRAY_REMOVE(A, B) END FROM table1"
    else:
        query = "SELECT ARRAY_REMOVE(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, None],
        [],
        [0, 0, 0, 0, 0],
        [],
        None,
        [0, 1, 2, 0, 1, 2, None],
    ] * 2
    pattern_b = [1, 0, 1, 1, 0, 2, 2, None] * 2
    expected_pattern = [
        [0, 0, 0, 0, 0],
        [1, 2, 1, 2],
        [0, 2, 0, 2, None],
        [],
        [],
        [],
        None,
        None,
    ] * 2
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array(make_vals(pattern_b), dtype=ignore_scalar_dtype(dtype))
    expected = pd.array(
        [make_vals(row) for row in expected_pattern],
        dtype=pd.ArrowDtype(pa.large_list(dtype)),
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"EXPR$0": expected}),
        sort_output=False,
    )


def test_array_remove_at(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_REMOVE_AT works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_EXCEPT. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT CASE WHEN A IS NULL THEN ARRAY_REMOVE_AT(A, B) ELSE ARRAY_REMOVE_AT(A, B) END FROM table1"
    else:
        query = "SELECT ARRAY_REMOVE_AT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    pattern_a = [
        [0, 1, 2, None],
        [0, 1, 2, None],
        [0, 1, 2, None],
        [0, None, 1, 2],
        [0, None, 1, 2],
        [],
        [],
        None,
        [0, 1, 2, None],
    ]
    expected_pattern = [
        [1, 2, None],
        [0, 2, None],
        [0, 1, 2],
        [0, None, 1, 2],
        [0, None, 1, 2],
        [],
        [],
        None,
        None,
    ]
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array([0, 1, -1, -7, 8, 0, -1, 0, None], dtype=pd.Int32Dtype())
    expected = pd.array(
        [make_vals(row) for row in expected_pattern],
        dtype=pd.ArrowDtype(pa.large_list(dtype)),
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": vals_a,
                "B": vals_b,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"EXPR$0": expected}),
        sort_output=False,
    )


def test_array_intersection(array_values, use_case, memory_leak_check):
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
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_INTERSECTION(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_INTERSECTION(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

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
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = pd.array(
            [make_vals(row) for row in answer_pattern],
            dtype=pd.ArrowDtype(pa.large_list(dtype)),
        )
    ctx = {
        "TABLE1": pd.DataFrame(
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


def test_array_cat(array_values, use_case, memory_leak_check):
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
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT I, CASE WHEN I < 0 THEN -1 ELSE NVL(ARRAY_SIZE(ARRAY_CAT(A, B)), -1) END FROM table1"
    else:
        query = "SELECT I, ARRAY_CAT(A, B) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        out = [None if idx is None else value_pool[idx] for idx in L]
        return out

    pattern_a = [
        [0, None, 1],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        [],
        [],
        None,
        [None],
    ]
    pattern_b = [
        [0, 2],
        [],
        None,
        [None],
        [0, 0, 1, None, 0],
        [],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    answer_pattern = [
        [0, None, 1, 0, 2],
        [0, 0, 1, 0, 0, 1, 2, 1, 0],
        None,
        [0, 0, 1, 0, 0, 1, 2, 1, 0, None],
        [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, None, 0],
        [],
        [0, 1],
        None,
        [None, 0, 1],
    ]
    vals_a = pd.array(
        [make_vals(row) for row in pattern_a], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    vals_b = pd.array(
        [make_vals(row) for row in pattern_b], dtype=pd.ArrowDtype(pa.large_list(dtype))
    )
    if use_case:
        answer = [-1 if ans is None else len(ans) for ans in answer_pattern]
    else:
        answer = pd.array(
            [make_vals(row) for row in answer_pattern],
            dtype=pd.ArrowDtype(pa.large_list(dtype)),
        )
    ctx = {
        "TABLE1": pd.DataFrame(
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
        convert_columns_to_pandas=True,
    )


def test_array_compact(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_COMPACT works correctly with different data type columns
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT CASE WHEN ARRAY_SIZE(A) IS NULL THEN -1 ELSE ARRAY_SIZE(ARRAY_COMPACT(A)) END AS res FROM table1"
    else:
        query = "SELECT ARRAY_COMPACT(A) AS res FROM table1"

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
    data = pd.array(
        [make_vals(row) for row in data_pattern],
        dtype=pd.ArrowDtype(pa.large_list(dtype)),
    )
    expected = [make_vals(row) for row in expected_pattern]
    # Returning arrays/json from case statements not fully supported, so
    # using a function like array_size to ensure the return type is different.
    if use_case:
        expected = [-1 if row is None else len(row) for row in expected]
    ctx = {"TABLE1": pd.DataFrame({"A": data})}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=pd.DataFrame({"res": expected}),
        convert_columns_to_pandas=True,
    )


def test_array_slice(array_values, use_case, memory_leak_check):
    """
    Test ARRAY_SLICE works correctly with different data type columns
    and with/without case statements.

    Takes in a list of 3 distinct values of the desired type and uses them
    to construct 2 columns of arrays of these values, then modifies them
    using ARRAY_SLICE. Since the values are always placed in the columns
    in the same permutations, the answers should always be the orderings of
    the original values.
    """
    value_pool, dtype = array_values
    if use_case:
        query = "SELECT CASE WHEN A IS NULL THEN ARRAY_SLICE(A, 2, 4) ELSE ARRAY_SLICE(A, 2, 4) END FROM table1"
    else:
        query = "SELECT ARRAY_SLICE(A, 2, 4) FROM table1"

    def make_vals(L):
        if L is None:
            return None
        return [None if idx is None else value_pool[idx] for idx in L]

    data_pattern = [
        [1, 2, 0, 1, 2, 0, 1, 2],
        None,
        [0, 1, 2, 0, 1, 2],
        [0, 1, None, 0, 1, 2, 0, 1, 2],
        [2, 1, 2],
        [0, 0, None, None, 0],
        [0, 0, 0, None],
        [],
        [None, None, 1, None],
    ]
    expected_pattern = [
        [0, 1],
        None,
        [2, 0],
        [None, 0],
        [2],
        [None, None],
        [0, None],
        [],
        [1, None],
    ]
    data = pd.array(
        [make_vals(row) for row in data_pattern],
        dtype=pd.ArrowDtype(pa.large_list(dtype)),
    )
    expected = pd.array(
        [make_vals(row) for row in expected_pattern],
        dtype=pd.ArrowDtype(pa.large_list(dtype)),
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": data,
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
    "syntax",
    [
        pytest.param(True, id="Index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="GET_syntax"),
    ],
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
def test_array_index_scalar(args, syntax, memory_leak_check):
    """
    Test Array indexing works correctly with different data type scalars
    """

    expr = args[0]

    # TO_VARIANT isn't needed, but wanted to confirm variant behavior.
    # TO_VARIANT is used in the scalar test with the opposite syntax,
    # to make sure that's no weird syntax specific bugs.
    arr_expr = f"ARRAY_CONSTRUCT({expr})"
    idx_call = f"{arr_expr}[0]" if syntax else f"GET(TO_VARIANT({arr_expr}), 0)"

    query = f"SELECT CASE WHEN {idx_call} = {expr} THEN {expr} ELSE NULL END as out_col FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"UNUSED_COL": list(range(10))})}
    py_output = pd.DataFrame({"OUT_COL": [args[1]] * 10})

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
    "syntax",
    [
        pytest.param(True, id="Index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="GET_syntax"),
    ],
)
def test_map_index_column_scalar(sql_object_array_values, syntax, memory_leak_check):
    """
    Test map indexing works correctly with different data type
    """

    # TO_VARIANT isn't needed, but wanted to confirm variant behavior.
    # TO_VARIANT is used in the scalar test with the opposite syntax,
    # to make sure that's no weird syntax specific bugs.
    idx_call = "TO_VARIANT(A)['value']" if syntax else "GET(A, 'value')"
    non_existent_idx_call = (
        "TO_VARIANT(A)['non_existent_key']" if syntax else "GET(A, 'non_existent_key')"
    )
    query = (
        f"SELECT {idx_call} as out_col, {non_existent_idx_call} as null_col FROM TABLE1"
    )

    input_col = sql_object_array_values["TABLE1"]["A"]
    output_col = []

    if isinstance(input_col.dtype.pyarrow_dtype, pa.MapType):
        for row in input_col:
            if (
                row is None
                or (isinstance(row, list) and pd.isna(row).all())
                or (not isinstance(row, list) and pd.isna(row))
            ):
                output_col.append(None)
            else:
                output_col.append(row[1][1])
    else:
        for row in input_col:
            if row is None or pd.isna(row):
                output_col.append(None)
            else:
                output_col.append(row["value"])

    py_out = pd.DataFrame(
        {
            "OUT_COL": pd.array(output_col),
            "NULL_COL": pd.array(
                [None] * len(output_col), dtype=pd.ArrowDtype(pa.null())
            ),
        }
    )

    check_query(
        query,
        sql_object_array_values,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )


@pytest.mark.slow()
def test_array_index_stress_test(memory_leak_check):
    """
    Stress test for Test Array indexing works correctly with different data type
    """

    query = """
    SELECT
        GET(ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(GET(ARRAY_CONSTRUCT(0), 0))[ARRAY_CONSTRUCT(0)[0]]), 0) as out_col_1,
        arr_col[arr_col[arr_col[arr_col[arr_col[0]]]] + 1] as out_col_2,
        GET(arr_col, GET(arr_col, GET(arr_col, GET(arr_col, GET(arr_col, 0)))) + 1) as out_col_3
    from
    (SELECT ARRAY_CONSTRUCT(int_col_0, int_col_1, int_col_2) as arr_col FROM TABLE1)
    """

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "INT_COL_0": [0] * 12,
                "INT_COL_1": list(range(12)),
                "INT_COL_2": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "OUT_COL_1": [0] * 12,
            "OUT_COL_2": list(range(12)),
            "OUT_COL_3": list(range(12)),
        }
    )

    check_query(
        query,
        ctx,
        None,
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

    query = """
    SELECT
        arr_col[0] as out_col
    from
    (SELECT ARRAY_CONSTRUCT() as arr_col FROM TABLE1)
    """

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "INPUT_COL": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "OUT_COL": pd.Series([None] * 12, dtype=pd.Int64Dtype()),
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

    query = """
    SELECT
        arr_col[2] as out_col
    from
    (SELECT ARRAY_CONSTRUCT(input_col) as arr_col FROM TABLE1)
    """

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "INPUT_COL": list(range(12)),
            }
        )
    }

    py_output = pd.DataFrame(
        {
            "OUT_COL": pd.Series([None] * 12, dtype=pd.Int64Dtype()),
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


@pytest.mark.parametrize(
    "syntax",
    [
        pytest.param(True, id="index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="get_syntax"),
    ],
)
@pytest.mark.parametrize(
    "expr",
    [
        pytest.param("1", id="int"),
        pytest.param("1.23", id="float"),
        pytest.param("'1'", id="string", marks=pytest.mark.slow),
        pytest.param("True", id="bool", marks=pytest.mark.slow),
    ],
)
def test_index_variant_invalid(expr, syntax, memory_leak_check):
    """
    Test indexing works correctly on variant data that is neither
    struct nor array (should return null).
    """

    arr_expr = "TO_VARIANT(input_col)"
    idx_call = f"{arr_expr}[{expr}]" if syntax else f"GET({arr_expr}, {expr})"

    query = f"SELECT CASE WHEN {idx_call} IS NULL THEN 'A' ELSE 'B' END as OUT_COL FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"INPUT_COL": list(range(10))})}
    py_output = pd.DataFrame({"OUT_COL": ["A"] * 10})

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "syntax",
    [
        pytest.param(True, id="Index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="GET_syntax"),
    ],
)
def test_map_index_scalar_scalar(sql_object_array_values, syntax, memory_leak_check):
    """
    Test Array indexing works correctly with different data type, when indexing with a scalar
    """

    idx_call = "TO_VARIANT(A)['value']" if syntax else "GET(A, 'value')"
    query = f"SELECT CASE WHEN A['id'] > 0 THEN {idx_call} ELSE NULL END as out_col FROM TABLE1"

    input_col = sql_object_array_values["TABLE1"]["A"]
    output_col = []

    if isinstance(input_col.dtype.pyarrow_dtype, pa.MapType):
        for row in input_col:
            if (
                row is None
                or (isinstance(row, list) and pd.isna(row).all())
                or (not isinstance(row, list) and pd.isna(row))
            ):
                output_col.append(None)
            else:
                output_col.append(row[1][1])
    else:
        for row in input_col:
            if row is None or pd.isna(row):
                output_col.append(None)
            else:
                output_col.append(row["value"])

    py_out = pd.DataFrame({"out_col": pd.array(output_col)})

    check_query(
        query,
        sql_object_array_values,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )


@pytest.mark.parametrize(
    "syntax",
    [
        pytest.param(True, id="Index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="GET_syntax"),
    ],
)
def test_map_index_scalar_column(map_array_values, syntax, memory_leak_check):
    """
    Test Array indexing works correctly with different data type, when indexing with a column.
    Currently, this can only be done on sql objects that are internally represented with
    the map array type.
    """

    idx_call = "TO_VARIANT(A)[B]" if syntax else "GET(A, B)"
    query = f"SELECT CASE WHEN A['id'] > 0 THEN {idx_call} ELSE NULL END as out_col FROM TABLE1"

    input_table = map_array_values["TABLE1"]

    def lambda_fn(val):
        if val is None or pd.isna(val).any():
            return None
        if val.B == "id":
            return val.A[0][1]
        elif val.B == "value":
            return val.A[1][1]
        else:
            return None

    output_col = input_table.apply(lambda x: lambda_fn(x), axis=1)

    py_out = pd.DataFrame({"out_col": pd.array(output_col)})

    check_query(
        query,
        map_array_values,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )


@pytest.mark.parametrize(
    "syntax",
    [
        pytest.param(True, id="Index_syntax", marks=pytest.mark.slow),
        pytest.param(False, id="GET_syntax"),
    ],
)
def test_map_index_column_column(map_array_values, syntax, memory_leak_check):
    """
    Test Array indexing works correctly with different data type, when indexing with a column.
    Currently, this can only be done on sql objects that are internally represented with
    the map array type.
    """

    idx_call = "TO_VARIANT(A)[B]" if syntax else "GET(A, B)"
    query = f"SELECT {idx_call} as out_col FROM TABLE1"

    input_table = map_array_values["TABLE1"]

    def lambda_fn(val):
        if val is None or pd.isna(val).any():
            return None
        if val.B == "id":
            return val.A[0][1]
        elif val.B == "value":
            return val.A[1][1]
        else:
            return None

    output_col = input_table.apply(lambda x: lambda_fn(x), axis=1)

    py_out = pd.DataFrame({"out_col": pd.array(output_col)})

    check_query(
        query,
        map_array_values,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )


@pytest.mark.slow()
def test_map_index_nested(memory_leak_check):
    """
    Making sure that map indexing works for nested maps
    """

    sql_object_array_values = {
        "TABLE1": nested_nested_df,
    }

    query = """
    SELECT
        GET(A, 'value')['A']['X'] as out_col_1,
        GET(A['value']['A'], 'W') as out_col_2
    from
    TABLE1
    """

    input_col = sql_object_array_values["TABLE1"]["A"]
    output_col_1 = []
    output_col_2 = []

    for row in input_col:
        if row is None or pd.isna(row):
            output_col_1.append(None)
            output_col_2.append(None)
        else:
            firstVal = row["value"]
            if firstVal is None or pd.isna(firstVal):
                output_col_1.append(None)
                output_col_2.append(None)
            else:
                secondVal = firstVal["A"]
                if secondVal is None or pd.isna(secondVal):
                    output_col_1.append(None)
                    output_col_2.append(None)
                else:
                    output_col_1.append(secondVal["X"])
                    output_col_2.append(secondVal["W"])

    py_out = pd.DataFrame(
        {
            "out_col_1": pd.array(output_col_1, dtype=pd.ArrowDtype(pa.string())),
            "out_col_2": pd.array(
                output_col_2,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        pa.struct(
                            [
                                pa.field(
                                    "A",
                                    pa.int32(),
                                    pa.field("B", pa.string()),
                                )
                            ]
                        )
                    )
                ),
            ),
        }
    )

    check_query(
        query,
        sql_object_array_values,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )


# Don't want to test on more than one rank to avoid empty ranks
@pytest_mark_one_rank
@pytest.mark.parametrize("use_struct", [True, False])
def test_map_index_nested_2(use_struct, memory_leak_check):
    # Stress test to insure that GET can handle arbitrary nesting

    def make_nested_struct_df(n_layers):
        if n_layers <= 0:
            return ({"A": 1}, pa.struct([pa.field("A", pa.int32())]))
        else:
            inner_value, inner_type = make_nested_struct_df(n_layers - 1)
            new_typ = pa.struct([pa.field("A", inner_type)])
            return ({"A": inner_value}, new_typ)

    def make_nested_map_df(n_layers):
        if n_layers <= 0:
            return ({"A": 1}, pa.map_(pa.string(), pa.int32()))
        else:
            inner_value, inner_type = make_nested_map_df(n_layers - 1)
            new_typ = pa.map_(pa.string(), inner_type)
            return ({"A": inner_value}, new_typ)

    if use_struct:
        n = 10
        value, typ = make_nested_struct_df(n)
    else:
        # map version takes a significant amount of time to run for higher levels
        # of nesting, exact reason is unknown.
        # https://bodo.atlassian.net/browse/BSE-2561
        n = 3
        value, typ = make_nested_map_df(n)

    ctx = {"TABLE1": pd.DataFrame({"A": pd.array([value], dtype=pd.ArrowDtype(typ))})}

    idx_str = "['A']" * (n + 1)
    query = f"""
    SELECT
        A{idx_str} as out_col
    from
    TABLE1
    """

    py_out = pd.DataFrame({"OUT_COL": [1]})

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=py_out,
    )


@pytest.mark.parametrize(
    "with_case",
    [
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
        pytest.param(False, id="without_case"),
    ],
)
def test_get_ignore_case(sql_object_array_values, with_case, memory_leak_check):
    """
    Test Array indexing works correctly with different data types in a case statement
    """

    if with_case:
        query = "SELECT CASE WHEN A['id'] > 0 THEN GET_IGNORE_CASE(A, 'VaLUe') ELSE NULL END as out_col FROM TABLE1"
    else:
        query = "SELECT GET_IGNORE_CASE(A, 'vAlUE') as out_col FROM TABLE1"

    input_col = sql_object_array_values["TABLE1"]["A"]
    output_col = []

    if isinstance(input_col.dtype.pyarrow_dtype, pa.MapType):
        for row in input_col:
            if (
                row is None
                or (isinstance(row, list) and pd.isna(row).all())
                or (not isinstance(row, list) and pd.isna(row))
            ):
                output_col.append(None)
            else:
                output_col.append(row[1][1])
    else:
        for row in input_col:
            if row is None or pd.isna(row):
                output_col.append(None)
            else:
                output_col.append(row["value"])

    py_out = pd.DataFrame({"out_col": pd.array(output_col)})

    check_query(
        query,
        sql_object_array_values,
        None,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_out,
    )
