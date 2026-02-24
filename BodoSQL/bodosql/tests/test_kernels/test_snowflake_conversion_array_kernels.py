"""Test Bodo's array kernel utilities for BodoSQL Snowflake conversion functions"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.api.types import is_float_dtype

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    pytest_slow_unless_codegen,
    temp_config_override,
)
from bodosql.kernels.array_kernel_utils import vectorized_sol

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen

true_vals = {"y", "yes", "t", "true", "on", "1"}
false_vals = {"n", "no", "f", "false", "off", "0"}


def str_to_bool(s):
    s = s.strip().lower() if not pd.isna(s) else s
    if s in true_vals or s in false_vals:
        return s in true_vals
    else:
        return np.nan


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    "y",
                    "yes",
                    "t",
                    "true",
                    "on",
                    "1",
                    "n",
                    "no",
                    "f",
                    "false",
                    "off",
                    "0",
                ]
                * 3
            ),
            id="valid_to_boolean_strings",
        ),
        pytest.param(
            pd.Series(
                [
                    "Y",
                    "yes",
                    "T",
                    "TRUE",
                    "on",
                    "1",
                    "n",
                    "NO",
                    "f",
                    "FALSE",
                    "OFF",
                    "0",
                ]
                * 3
            ),
            id="valid_to_boolean_strings_mixed_case",
        ),
        pytest.param(
            pd.Series([1, 0, 1, 0, 0, -2, -400] * 3), id="valid_to_boolean_ints"
        ),
        pytest.param(
            pd.Series([1.1, 0.0, 1.0, 0.1, 0, -2, -400] * 3),
            id="valid_to_boolean_floats",
        ),
        pytest.param(
            pd.Series(["t", "a", "b", "y", "f"] * 3),
            id="invalid_to_boolean_strings",
        ),
        pytest.param(
            pd.Series([1.1, 0.0, np.inf, 0.1, np.nan, -2, -400] * 3),
            id="invalid_to_boolean_floats",
        ),
    ]
)
def to_boolean_test_arrs(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series([1, None, 1, 0, None, -2, -400] * 3, dtype="Int64"),
            id="ints_with_nulls",
        ),
        pytest.param(
            pd.Series([1.1, 0.0, np.nan, 0.1, 0, None, -400] * 3),
            id="floats_with_nulls",
        ),
        pytest.param(
            pd.Series(
                ["t", None, None, "y", "f"] * 3,
            ),
            id="strings_with_nulls",
        ),
    ]
)
def to_boolean_test_arrs_null(request):
    return request.param


def test_to_boolean(to_boolean_test_arrs):
    def impl(arr):
        return pd.Series(bodosql.kernels.to_boolean(arr))

    arr = to_boolean_test_arrs
    to_bool_scalar_fn = lambda x: np.nan if pd.isna(x) else bool(x)
    run_check = True
    if pd._libs.lib.infer_dtype(arr, skipna=True) == "string":
        if arr.apply(
            lambda x: pd.isna(x) or x.lower() in true_vals or x.lower() in false_vals
        ).all():
            to_bool_scalar_fn = str_to_bool
        else:
            with pytest.raises(ValueError, match="string must be one of"):
                bodo.jit(impl)(arr)
            run_check = False
    elif is_float_dtype(arr):
        if np.isinf(arr).any():
            with pytest.raises(
                ValueError, match="value must be a valid numeric expression"
            ):
                bodo.jit(impl)(arr)
            run_check = False
    if run_check:
        py_output = vectorized_sol((arr,), to_bool_scalar_fn, "boolean")
        check_func(
            impl,
            (arr,),
            py_output=py_output,
            check_dtype=True,
            reset_index=True,
            check_names=False,
        )


def test_try_to_boolean(to_boolean_test_arrs):
    def impl(arr):
        return pd.Series(bodosql.kernels.try_to_boolean(arr))

    arr = to_boolean_test_arrs
    if pd._libs.lib.infer_dtype(arr, skipna=True) == "string":
        to_bool_scalar_fn = str_to_bool
    elif is_float_dtype(arr):
        to_bool_scalar_fn = lambda x: np.nan if np.isnan(x) or np.isinf(x) else bool(x)
    else:
        to_bool_scalar_fn = lambda x: np.nan if pd.isna(x) else bool(x)
    py_output = vectorized_sol((arr,), to_bool_scalar_fn, "boolean")
    check_func(
        impl,
        (arr,),
        py_output=py_output,
        check_dtype=True,
        reset_index=True,
        check_names=False,
    )


def test_try_to_boolean_opt(to_boolean_test_arrs_null):
    def impl(arr):
        return pd.Series(bodosql.kernels.try_to_boolean(arr))

    arr = to_boolean_test_arrs_null
    if pd._libs.lib.infer_dtype(arr, skipna=True) == "string":
        py_output = vectorized_sol((arr,), str_to_bool, "boolean")
    elif is_float_dtype(arr):
        py_output = vectorized_sol(
            (arr,),
            lambda x: np.nan if np.isnan(x) or np.isinf(x) else bool(x),
            "boolean",
        )
    else:
        py_output = vectorized_sol(
            (arr,), lambda x: np.nan if pd.isna(x) else bool(x), "boolean"
        )
    check_func(
        impl,
        (arr,),
        py_output=py_output,
        check_dtype=True,
        reset_index=True,
        check_names=False,
    )


def test_to_boolean_opt(to_boolean_test_arrs_null):
    def impl(arr):
        return pd.Series(bodosql.kernels.to_boolean(arr))

    arr = to_boolean_test_arrs_null
    if pd._libs.lib.infer_dtype(arr, skipna=True) == "string":
        py_output = vectorized_sol((arr,), str_to_bool, "boolean")
    elif is_float_dtype(arr):
        py_output = vectorized_sol(
            (arr,),
            lambda x: np.nan if np.isnan(x) or np.isinf(x) else bool(x),
            "boolean",
        )
    else:
        py_output = vectorized_sol(
            (arr,), lambda x: np.nan if pd.isna(x) else bool(x), "boolean"
        )
    check_func(
        impl,
        (arr,),
        py_output=py_output,
        check_dtype=True,
        reset_index=True,
        check_names=False,
    )


_dates = pd.Series(
    pd.date_range("2010-1-1", periods=10, freq="841D", unit="ns"),
    dtype="datetime64[ns]",
).apply(lambda x: x.date())
_timestamps = pd.Series(
    pd.date_range("20130101", periods=10, freq="h", unit="ns"), dtype="datetime64[ns]"
)
_dates_nans = _dates.copy()
_timestamps_nans = _timestamps.copy()
_dates_nans[4] = _dates_nans[7] = np.nan
_timestamps_nans[2] = _timestamps_nans[7] = np.nan


@pytest.mark.parametrize(
    "input, is_scalar, expected",
    [
        pytest.param(
            0,
            True,
            "0",
            id="int-scalar",
        ),
        pytest.param(
            pd.array(
                [-2147483648, -999, -1, 0, 1, 99, 2147483647, None], pd.Int32Dtype()
            ),
            False,
            pd.array(
                ["-2147483648", "-999", "-1", "0", "1", "99", "2147483647", None],
                pd.StringDtype(),
            ),
            id="int-vector",
        ),
        pytest.param(
            pd.array([None, None, None], pd.ArrowDtype(pa.null())),
            False,
            pd.array([None, None, None], pd.ArrowDtype(pa.null())),
            id="null-vector",
        ),
        pytest.param(
            pd.array(
                [np.inf, -20.5, 1.1, 10.1, 101.99, None, 20000.000, -np.inf],
                pd.Float32Dtype(),
            ),
            False,
            # TODO(Yipeng): Match Snowflake's behavior on float operands
            pd.array(
                [
                    "inf",
                    "-20.500000",
                    "1.100000",
                    "10.100000",
                    "101.989998",
                    None,
                    "20000.000000",
                    "-inf",
                ],
                pd.StringDtype(),
            ),
            id="float-vector",
        ),
        pytest.param(
            pd.array([True, False, None, False, True, None, False], pd.BooleanDtype()),
            False,
            pd.array(
                ["true", "false", None, "false", "true", None, "false"],
                pd.StringDtype(),
            ),
            id="bool-vector",
        ),
        pytest.param(
            pd.array(["1", "2", None, "ok"] * 2, pd.StringDtype()),
            False,
            pd.array(["1", "2", None, "ok"] * 2, pd.StringDtype()),
            id="string-vector",
        ),
        pytest.param(
            _dates,
            False,
            None,
            id="date-vector",
        ),
        pytest.param(
            _timestamps,
            False,
            None,
            id="timestamp-vector",
        ),
        pytest.param(
            _dates_nans,
            False,
            None,
            id="date_with_nulls-vector",
        ),
        pytest.param(
            _timestamps_nans,
            False,
            None,
            id="timestamp_with_nulls-vector",
        ),
        pytest.param(
            pd.Series([bytes(32), b"abcde", b"ihohi04324", None] * 2).values,
            False,
            pd.array(
                [
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    "6162636465",
                    "69686f68693034333234",
                    None,
                ]
                * 2
            ),
            id="binary-vector",
        ),
        pytest.param(
            pd.array(
                [-2147483648, -999, -1, 0, 1, 99, 2147483647, None], pd.Int32Dtype()
            ),
            True,
            "[-2147483648,-999,-1,0,1,99,2147483647,undefined]",
            id="int_array-scalar",
        ),
        pytest.param(
            pd.array(
                [
                    [-2147483648, -999, -1, 0, 1, 99, 2147483647, None],
                    [None, 200, -1, 0, None, -9999, None],
                    [None],
                    [],
                    None,
                ],
                pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            False,
            pd.array(
                [
                    "[-2147483648,-999,-1,0,1,99,2147483647,undefined]",
                    "[undefined,200,-1,0,undefined,-9999,undefined]",
                    "[undefined]",
                    "[]",
                    None,
                ]
            ),
            id="int_array-vector",
        ),
        pytest.param(
            pd.array(["1", "2", None, "ok"], pd.StringDtype()),
            True,
            '["1","2",undefined,"ok"]',
            id="string_array-scalar",
        ),
        pytest.param(
            pd.array(
                [
                    [{"A": 0, "B": "foo"}],
                    [],
                    [{"A": None, "B": "foo"}, None],
                    [None, {"A": 7, "B": "buzz"}],
                    [
                        {"A": 3, "B": None},
                        None,
                        {"A": 0, "B": "bar"},
                        {"A": 9, "B": "fizz"},
                    ],
                    [{"A": 0, "B": "foo"}, {"A": 1, "B": "bar"}],
                    [{"A": None, "B": "fizz"}, {"A": 0, "B": "buzz"}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ),
            False,
            pd.array(
                [
                    '[{"A":0,"B":"foo"}]',
                    "[]",
                    '[{"B":"foo"},undefined]',
                    '[undefined,{"A":7,"B":"buzz"}]',
                    '[{"A":3},undefined,{"A":0,"B":"bar"},{"A":9,"B":"fizz"}]',
                    '[{"A":0,"B":"foo"},{"A":1,"B":"bar"}]',
                    '[{"B":"fizz"},{"A":0,"B":"buzz"}]',
                ]
            ),
            id="struct_array-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [{"A": 0, "B": 1}, {"D": 2, "C": 4}],
                    [{"A": 0, "B": None}, {"D": None, "C": 4}, None],
                    [],
                    None,
                    [{}],
                ],
                dtype=pd.ArrowDtype(pa.list_(pa.map_(pa.string(), pa.int32()))),
            ),
            False,
            pd.array(
                [
                    '[{"A":0,"B":1},{"D":2,"C":4}]',
                    '[{"A":0},{"C":4},undefined]',
                    "[]",
                    None,
                    "[{}]",
                ]
            ),
            id="map_array-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [
                        {
                            "map": {"a": [0, 1], "b": [None, 3]},
                            "struct": {"S": "abc", "D": 1.2},
                        }
                    ],
                    [
                        {
                            "map": {"a": [None, 1], "b": None},
                            "struct": {"S": None, "D": 3.5},
                        }
                    ],
                    [
                        {"map": None, "struct": {"S": "efg", "D": None}},
                        {"map": {}, "struct": {"S": None, "D": None}},
                        {"map": {"a": [0, 1], "b": [], "ok": None}, "struct": None},
                    ],
                    None,
                    [],
                    [
                        None,
                        {"map": None, "struct": {"S": "ok", "D": None}},
                        {"map": {}, "struct": None},
                    ],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field(
                                    "map",
                                    pa.map_(pa.string(), pa.large_list(pa.int32())),
                                ),
                                pa.field(
                                    "struct",
                                    pa.struct(
                                        [
                                            pa.field("S", pa.string()),
                                            pa.field("D", pa.float32()),
                                        ]
                                    ),
                                ),
                            ]
                        )
                    )
                ),
            ),
            True,
            '[[{"map":{"a":[0,1],"b":[undefined,3]},"struct":{"S":"abc","D":1.200000}}],[{"map":{"a":[undefined,1]},"struct":{"D":3.500000}}],[{"struct":{"S":"efg"}},{"map":{},"struct":{}},{"map":{"a":[0,1],"b":[]}}],undefined,[],[undefined,{"struct":{"S":"ok"}},{"map":{}}]]',
            id="nested_array-scalar",
        ),
        pytest.param(
            np.array(
                [
                    bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
                    bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
                    bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
                    bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
                    bodo.types.TimestampTZ.fromUTC("2025-05-06 07:08:09", 500),
                ]
            ),
            False,
            pd.array(
                [
                    "2021-01-02 04:44:05 +0140",
                    "2022-02-03 07:25:06 +0320",
                    "2023-03-04 10:06:07 +0500",
                    "2024-04-05 12:47:08 +0640",
                    "2025-05-06 15:28:09 +0820",
                ]
            ),
            id="timestamptz-vector",
        ),
    ],
)
def test_to_char(input, is_scalar, expected):
    if expected is None:
        expected = vectorized_sol(
            (input,), lambda x: np.nan if pd.isna(x) else str(x), "string"
        ).array

    def impl(arr):
        return bodosql.kernels.to_char(arr, None, is_scalar)

    check_func(
        impl,
        (input,),
        py_output=expected,
        check_dtype=False,
        reset_index=False,
        distributed=not is_scalar,
        is_out_distributed=not is_scalar,
        dist_test=not is_scalar,
    )


@pytest.mark.slow
def test_nullarray_to_char():
    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)

        return pd.Series(bodosql.kernels.to_char(null_arr))

    n = 10
    py_output = pd.Series([None] * n)
    check_func(impl, (n,), py_output=py_output)


@pytest.mark.parametrize(
    "input, expected",
    [
        pytest.param(
            pd.array([1, None, 4, None, 16, None, 64], dtype=pd.Int32Dtype()),
            pd.array([1.0, None, 4.0, None, 16.0, None, 64.0], dtype=pd.Float64Dtype()),
            id="nullable_int",
        ),
        pytest.param(
            np.array([1, -2, 4, -8, 16, -32, 64], dtype=np.int64()),
            np.array([1.0, -2.0, 4.0, -8.0, 16.0, -32.0, 64.0], dtype=np.float64()),
            id="numpy_int",
        ),
        pytest.param(
            pd.array([1.0, -0.3, None, 3.14, 2.718281828], dtype=pd.Float64Dtype()),
            pd.array([1.0, -0.3, None, 3.14, 2.718281828], dtype=pd.Float64Dtype()),
            id="float",
        ),
        pytest.param(
            pd.array([True, False, None] * 2, dtype=pd.BooleanDtype()),
            pd.array([1.0, 0.0, None] * 2, dtype=pd.Float64Dtype()),
            id="boolean",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "0",
                    "0.5",
                    "-144",
                    None,
                    "-2.71",
                    "72.315",
                    "inf",
                    "-inf",
                    "2.024e3",
                    "-1e-5",
                ]
            ),
            pd.array(
                [
                    1.0,
                    0.0,
                    0.5,
                    "-144",
                    None,
                    -2.71,
                    72.315,
                    np.inf,
                    -np.inf,
                    2024.0,
                    -0.00001,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="string",
        ),
    ],
)
def test_to_double(input, expected):
    def impl(arr):
        return bodosql.kernels.to_double(arr, None)

    check_func(
        impl,
        (input,),
        py_output=expected,
        check_dtype=False,
        reset_index=False,
    )


@pytest.mark.parametrize(
    "arr, format_str, is_scalar_arr, expected",
    [
        pytest.param(
            pd.array(
                [
                    pd.Timestamp(2020, 8, 17, 10, 10, 10),
                    pd.Timestamp(2022, 9, 18, 20, 20, 20),
                    pd.Timestamp(2023, 10, 19, 0, 30, 30),
                    pd.Timestamp(2024, 11, 20, 10, 40, 40),
                ]
                * 3,
                dtype="datetime64[ns]",
            ),
            "MM/DD/YYYY HH24:MI:SS",
            False,
            pd.array(
                [
                    "08/17/2020 10:10:10",
                    "09/18/2022 20:20:20",
                    "10/19/2023 00:30:30",
                    "11/20/2024 10:40:40",
                ]
                * 3
            ),
            id="timestamp-vector-scalar",
        ),
        pytest.param(
            pd.array(
                [
                    pd.Timestamp(2020, 8, 17, 10, 10, 10),
                    pd.Timestamp(2022, 9, 18, 20, 20, 20),
                    None,
                    pd.Timestamp(2023, 10, 19, 0, 30, 30),
                    None,
                    pd.Timestamp(2024, 11, 20, 10, 40, 40),
                ]
                * 2,
                dtype="datetime64[ns]",
            ),
            pd.array(
                [
                    "MM/DD/YYYY HH24:MI:SS",
                    None,
                    "MM/DD/YYYY HH24:MI:SS",
                    "MM/DD/YYYY HH24:MI:SS",
                    "MM/DD/YYYY HH24:MI:SS",
                    None,
                ]
                * 2
            ),
            False,
            pd.array(
                [
                    "08/17/2020 10:10:10",
                    "2022-09-18 20:20:20",
                    None,
                    "10/19/2023 00:30:30",
                    None,
                    "2024-11-20 10:40:40",
                ]
                * 2
            ),
            id="timestamp-vector-vector",
        ),
    ],
)
def test_to_char_format_str(arr, format_str, is_scalar_arr, expected):
    def impl(arr, format_str):
        return bodosql.kernels.to_char(arr, format_str, is_scalar_arr)

    check_func(
        impl,
        (arr, format_str),
        py_output=expected,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
def test_option_to_char(flag0, flag1, memory_leak_check):
    def impl(arr, format_str, flag0, flag1):
        arg0 = None if flag0 else arr
        arg1 = None if flag1 else format_str
        return bodosql.kernels.to_char(arg0, arg1, is_scalar=True)

    arr = pd.Timestamp(2023, 12, 4, 16, 50, 10)
    format_str = "MM/DD/YYYY HH24:MI:SS"
    answer = (
        None if flag0 else ("2023-12-04 16:50:10" if flag1 else "12/04/2023 16:50:10")
    )

    check_func(
        impl,
        (arr, format_str, flag0, flag1),
        py_output=answer,
        check_dtype=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize(
    "arr, prec, scale, answer",
    [
        pytest.param(
            pd.Series([1, 2, 3, None, 5, 6, 7, None, 9, 10]),
            3,
            0,
            pd.Series([1, 2, 3, None, 5, 6, 7, None, 9, 10]),
            id="int_with_nulls_all_valid_no_scale",
        ),
        pytest.param(
            pd.Series(["1", "2", "3", None, "5", "6", "7", None, "9", "10"]),
            3,
            0,
            pd.Series([1, 2, 3, None, 5, 6, 7, None, 9, 10]),
            id="string_with_nulls_all_valid_no_scale",
        ),
        pytest.param(
            pd.Series(
                [
                    "1.5",
                    "2.2",
                    "3.11",
                    None,
                    "5.111111",
                    "6.188",
                    "7.9",
                    None,
                    "9.999",
                    "10",
                ]
            ),
            3,
            1,
            pd.Series([1.5, 2.2, 3.1, None, 5.1, 6.2, 7.9, None, 10.0, 10.0]),
            id="string_with_nulls_all_valid_with_scale",
        ),
        pytest.param(
            pd.Series([1.3, 2.5, 3.2, None, 5.7, 6.0, 7.1, None, 9.7, 10.9]),
            3,
            0,
            pd.Series([1, 3, 3, None, 6, 6, 7, None, 10, 11]),
            id="float_with_nulls_all_valid_no_scale",
        ),
        pytest.param(
            pd.Series([1.3, 2.5, 3.2, None, 5.7, 6.0, 7.1, None, 9.7, 10.9]),
            3,
            1,
            pd.Series([1.3, 2.5, 3.2, None, 5.7, 6.0, 7.1, None, 9.7, 10.9]),
            id="float_with_nulls_all_valid_with_scale",
        ),
        pytest.param(
            pd.Series([True, False, None, False, True], dtype="boolean"),
            38,
            0,
            pd.Series([1, 0, None, 0, 1], dtype="Int64"),
            id="boolean_with_nulls",
        ),
    ],
)
def test_to_number(arr, prec, scale, answer):
    # Tests to_number with a number of different valid inputs

    def impl(arr):
        return pd.Series(bodosql.kernels.to_number(arr, prec, scale, True))

    check_func(
        impl,
        (arr,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "arr, prec, scale, answer",
    [
        # Some invalid inputs
        pytest.param(
            pd.Series([1, 2, 113, None, 5, -600, 7, None, 129, 10]),
            2,
            0,
            pd.Series([1, 2, None, None, 5, None, 7, None, None, 10]),
            id="int_with_nulls_some_invalid_no_scale",
        ),
        pytest.param(
            pd.Series(["1", "2", "-113", None, "5", "600", "7", None, "-129", "10"]),
            2,
            0,
            pd.Series([1, 2, None, None, 5, None, 7, None, None, 10]),
            id="string_with_nulls_some_invalid_no_scale",
        ),
        pytest.param(
            pd.Series([1.3, 2.5, -113.2, None, 5.7, 600.0, 7.1, None, 129.7, 10.9]),
            2,
            0,
            pd.Series([1, 3, None, None, 6, None, 7, None, None, 11]),
            id="float_with_nulls_some_invalid_no_scale",
        ),
        pytest.param(
            pd.Series([1, 2, 113, None, 5, 600, 7, None, -129, 10]),
            3,
            1,
            pd.Series([1.0, 2.0, None, None, 5.0, None, 7.0, None, None, 10.0]),
            id="int_with_nulls_some_invalid_with_scale",
        ),
        pytest.param(
            pd.Series(
                [
                    "1.55",
                    "2.1",
                    "113.2",
                    None,
                    "5.8",
                    "600",
                    "7.1",
                    None,
                    "129.2",
                    "10.1",
                ]
            ),
            3,
            1,
            pd.Series([1.6, 2.1, None, None, 5.8, None, 7.1, None, None, 10.1]),
            id="string_with_nulls_some_invalid_with_scale",
        ),
        pytest.param(
            pd.Series([1.3, 2.5, 113.2, None, 5.7, -600.0, 7.1, None, 129.7, 10.9]),
            3,
            1,
            pd.Series([1.3, 2.5, None, None, 5.7, None, 7.1, None, None, 10.9]),
            id="float_with_nulls_some_invalid_with_scale",
        ),
        pytest.param(
            pd.Series(
                [
                    10**9 + 10,
                    10**9 + 0.5,
                    10**11,
                    None,
                    10**9,
                    10**13,
                    10**9,
                    None,
                    10**7,
                    10**9 + 0.4,
                ]
            ),
            10,
            0,
            pd.Series(
                [
                    10**9 + 10,
                    10**9 + 1,
                    None,
                    None,
                    10**9,
                    None,
                    10**9,
                    None,
                    10**7,
                    10**9,
                ]
            ),
            id="test_precision_10",
        ),
        pytest.param(
            pd.Series(
                [
                    10**19,
                    10**16,
                    10**11,
                    None,
                    10**16 + 0.666,
                    10**16 + 0.5,
                    10**9 + 0.9999,
                    None,
                    10**20,
                    10**10 + 0.4,
                ]
            ),
            19,
            2,
            pd.Series(
                [
                    None,
                    10**16,
                    10**11,
                    None,
                    10**16 + 0.67,
                    10**16 + 0.5,
                    10**9 + 1,
                    None,
                    None,
                    10**10 + 0.4,
                ]
            ),
            id="test_precision_18",
            marks=pytest.mark.skip(
                "Int64 overflow issues: Anything larger than 10**18 overflows the bounds of an int64"
            ),
        ),
    ],
)
def test_try_to_number(arr, prec, scale, answer):
    # Tests try_to_number with a number of different valid/invalid inputs

    def impl(arr):
        return pd.Series(bodosql.kernels.try_to_number(arr, prec, scale, True))

    check_func(
        impl,
        (arr,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "arr, prec, scale, answer",
    [
        # scalars
        pytest.param(
            -100,
            3,
            0,
            pd.Series(-100),
            id="scalar_int_no_scale",
        ),
        pytest.param(
            -100,
            4,
            1,
            pd.Series(-100.0),
            id="scalar_int_with_scale",
        ),
        pytest.param(
            -100,
            3,
            1,
            pd.Series([None], dtype=pd.Float64Dtype()),
            id="scalar_int_invalid",
        ),
        pytest.param(
            10.123,
            3,
            0,
            pd.Series(10),
            id="scalar_float_no_scale",
        ),
        pytest.param(
            10.173,
            4,
            1,
            pd.Series(10.2),
            id="scalar_float_with_scale",
        ),
        pytest.param(
            10.123,
            2,
            1,
            pd.Series([None], dtype=pd.Float64Dtype()),
            id="scalar_float_invalid",
        ),
        pytest.param(
            "10.123",
            3,
            0,
            pd.Series(10),
            id="scalar_string_no_scale",
        ),
        pytest.param(
            "10.173",
            4,
            1,
            pd.Series(10.2),
            id="scalar_string_with_scale",
        ),
        pytest.param(
            "10.123",
            2,
            1,
            pd.Series([None], dtype=pd.Float64Dtype()),
            id="scalar_string_invalid",
        ),
    ],
)
def test_try_to_number_scalar(arr, prec, scale, answer):
    # Tests try_to_number with scalar inputs

    def impl(arr):
        return pd.Series(
            bodosql.kernels.try_to_number(arr, prec, scale, True),
            index=pd.RangeIndex(1),
        )

    expected_out = answer

    check_func(
        impl,
        (arr,),
        py_output=expected_out,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "val",
    [
        "10000",
        "-99999.2",
    ],
)
def test_to_number_invalid_precision(val):
    # Tests to_number with too many digits throws an error

    def impl(arr):
        return bodosql.kernels.to_number(arr, 3, 1, True)

    with pytest.raises(
        ValueError, match="Value has too many digits to the left of the decimal"
    ):
        bodo.jit(impl)(val)


@pytest.fixture(
    params=(
        "0.1.2",
        "1-2",
        "--2",
        "hello world",
    ),
)
def invalid_to_number_string_inputs(request):
    """Returns a number of scalar inputs that should result in
    an error for to_number, or None for try_to_number"""
    return request.param


def test_try_to_number_invalid_inputs(invalid_to_number_string_inputs):
    def impl(val):
        return pd.Series(
            bodosql.kernels.try_to_number(val, 38, 0, True),
            index=pd.RangeIndex(1),
        )

    expected_out = pd.Series([None])

    check_func(
        impl,
        (invalid_to_number_string_inputs,),
        py_output=expected_out,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


def test_to_number_invalid_inputs(invalid_to_number_string_inputs):
    def impl(val):
        return bodosql.kernels.to_number(val, 38, 0, True)

    with pytest.raises(ValueError, match="unable to convert string literal"):
        bodo.jit(impl)(invalid_to_number_string_inputs)


def test_to_number_decimal_to_decimal(precision_scale_decimal_array, memory_leak_check):
    def impl(arr):
        return bodosql.kernels.to_number(arr, 28, 3, True)

    py_output = pd.array(
        ["1", "1.55", "1.56", "10.56", "1000.5", None, None, "10004.1", "-11.41"],
        dtype=pd.ArrowDtype(pa.decimal128(28, 3)),
    )
    with temp_config_override("bodo_use_decimal", True):
        check_func(impl, (precision_scale_decimal_array,), py_output=py_output)


def test_try_to_number_decimal_to_decimal(
    precision_scale_decimal_array, memory_leak_check
):
    def impl(arr):
        return bodosql.kernels.try_to_number(arr, 4, 3, True)

    py_output = pd.array(
        ["1", "1.55", "1.56", None, None, None, None, None, None],
        dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
    )
    with temp_config_override("bodo_use_decimal", True):
        check_func(impl, (precision_scale_decimal_array,), py_output=py_output)
