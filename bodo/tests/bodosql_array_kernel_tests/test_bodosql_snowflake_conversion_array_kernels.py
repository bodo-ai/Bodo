# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL Snowflake conversion functions"""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_string_dtype

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func

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
    if is_string_dtype(arr):
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
    if is_string_dtype(arr):
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
    if is_string_dtype(arr):
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
    if is_string_dtype(arr):
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


_dates = pd.Series(pd.date_range("20130101", periods=10, freq="D"))
_times = pd.Series(pd.date_range("20130101", periods=10, freq="H"))
_dates_nans = _dates.copy()
_times_nans = _times.copy()
_dates_nans[4] = _dates_nans[7] = np.nan
_times_nans[2] = _dates_nans[7] = np.nan


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
            (_times,),
            id="time_to_str",
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
            (_times_nans,),
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
