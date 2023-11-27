# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL Snowflake conversion functions"""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_string_dtype

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen

true_vals = {"y", "yes", "t", "true", "on", "1"}
false_vals = {"n", "no", "f", "false", "off", "0"}


def str_to_bool(s):
    s = s.lower() if s else s
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
        return pd.Series(bodo.libs.bodosql_array_kernels.to_boolean(arr))

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
        return pd.Series(bodo.libs.bodosql_array_kernels.try_to_boolean(arr))

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
        return pd.Series(bodo.libs.bodosql_array_kernels.try_to_boolean(arr))

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
        return pd.Series(bodo.libs.bodosql_array_kernels.to_boolean(arr))

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


_dates = pd.Series(pd.date_range("2010-1-1", periods=10, freq="841D")).apply(
    lambda x: x.date()
)
_timestamps = pd.Series(pd.date_range("20130101", periods=10, freq="H"))
_dates_nans = _dates.copy()
_timestamps_nans = _timestamps.copy()
_dates_nans[4] = _dates_nans[7] = np.nan
_timestamps_nans[2] = _timestamps_nans[7] = np.nan


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),),
            id="int_to_str",
        ),
        pytest.param(
            (pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]),),
            id="float_to_str",
        ),
        pytest.param(
            (
                pd.Series(
                    [True, False, True, False, True, False, True, False, True, False]
                ),
            ),
            id="bool_to_str",
        ),
        pytest.param(
            (_dates,),
            id="date_to_str",
        ),
        pytest.param(
            (_timestamps,),
            id="timestamps_to_str",
        ),
        pytest.param(
            (
                pd.Series(
                    np.array([bytes(32), b"abcde", b"ihohi04324", None] * 3, object)
                ),
            ),
            id="binary",
        ),
    ],
)
def test_to_char(args):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_char(arr))

    arr = args[0]
    py_output = None
    if pd.api.types.is_float_dtype(arr):
        py_output = vectorized_sol(
            args, lambda x: np.nan if pd.isna(x) else f"{x:.6f}", "string"
        )
    elif pd.api.types.infer_dtype(arr) == "bytes":
        py_output = vectorized_sol(
            args, lambda x: np.nan if pd.isna(x) else x.hex(), "string"
        )
    elif pd.api.types.is_bool_dtype(arr):
        py_output = vectorized_sol(
            args,
            lambda x: np.nan if pd.isna(x) else ("true" if x else "false"),
            "string",
        )
    else:
        py_output = vectorized_sol(
            args, lambda x: np.nan if pd.isna(x) else str(x), "string"
        )
    if py_output is not None:
        check_func(
            impl,
            args,
            py_output=py_output,
            check_dtype=False,
            reset_index=True,
            check_names=False,
        )


@pytest.mark.parametrize(
    "arr, format_str, answer",
    [
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp(2020, 8, 17, 10, 10, 10),
                    pd.Timestamp(2022, 9, 18, 20, 20, 20),
                    pd.Timestamp(2023, 10, 19, 0, 30, 30),
                    pd.Timestamp(2024, 11, 20, 10, 40, 40),
                ]
                * 3
            ),
            "MM/DD/YYYY HH24:MI:SS",
            pd.Series(
                [
                    "08/17/2020 10:10:10",
                    "09/18/2022 20:20:20",
                    "10/19/2023 00:30:30",
                    "11/20/2024 10:40:40",
                ]
                * 3
            ),
        )
    ],
)
def test_to_char_format_str(arr, format_str, answer):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_char(arr, format_str))

    check_func(
        impl,
        (arr,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (pd.Series([1, 2, 3, None, 5, 6, 7, None, 9, 10]),),
            id="int_with_nulls",
        ),
        pytest.param(
            (pd.Series([1.1, 2.2, 3.3, None, 5.5, 6.6, 7.7, None, 9.9, 10.1]),),
            id="float_with_nulls",
        ),
        pytest.param(
            (
                pd.Series(
                    [True, False, True, None, True, False, True, None, True, False],
                    dtype="boolean",
                ),
            ),
            id="bool_with_nulls",
        ),
        pytest.param(
            (_dates_nans,),
            id="date_with_nulls",
        ),
        pytest.param(
            (_timestamps_nans,),
            id="time_with_nulls",
        ),
    ],
)
def test_to_char_opt(args):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_char(arr))

    arr = args[0]
    if pd.api.types.is_float_dtype(arr):
        py_output = vectorized_sol(
            args, lambda x: np.nan if pd.isna(x) else f"{x:.6f}", "string"
        )
    elif pd.api.types.is_bool_dtype(arr):
        py_output = vectorized_sol(
            args,
            lambda x: np.nan if pd.isna(x) else ("true" if x else "false"),
            "string",
        )
    else:
        py_output = vectorized_sol(
            args, lambda x: np.nan if pd.isna(x) else str(x), "string"
        )
    check_func(
        impl,
        args,
        py_output=py_output,
        check_dtype=False,
        reset_index=True,
        check_names=False,
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
    ],
)
def test_to_number(arr, prec, scale, answer):
    # Tests to_number with a number of different valid inputs

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_number(arr, prec, scale))

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
        return pd.Series(
            bodo.libs.bodosql_array_kernels.try_to_number(arr, prec, scale)
        )

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
            pd.Series([None], dtype=pd.Float64Dtype),
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
            pd.Series([None], dtype=pd.Float64Dtype),
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
            pd.Series([None], dtype=pd.Float64Dtype),
            id="scalar_string_invalid",
        ),
    ],
)
def test_try_to_number_scalar(arr, prec, scale, answer):
    # Tests try_to_number with scalar inputs

    def impl(arr):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.try_to_number(arr, prec, scale),
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
        return bodo.libs.bodosql_array_kernels.to_number(arr, 3, 1)

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
            bodo.libs.bodosql_array_kernels.try_to_number(val, 38, 0),
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
        return bodo.libs.bodosql_array_kernels.to_number(val, 38, 0)

    with pytest.raises(ValueError, match="unable to convert string literal"):
        bodo.jit(impl)(invalid_to_number_string_inputs)
