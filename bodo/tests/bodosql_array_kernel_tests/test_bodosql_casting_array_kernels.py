# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for casting
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            (pd.Series([1, 2, 3, 4, 5]),),
            id="int_array",
        ),
        pytest.param(
            (pd.Series([32.3, None, -2, 4.4, 5]),),
            id="small_float_array",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        32123123.32313423,
                        None,
                        -2234234254,
                        np.nan,
                        50782748792,
                        12379739850550389709,
                        -8454757700450211157,
                    ]
                ),
            ),
            id="large_float_array",
        ),
        pytest.param(
            (pd.Series([1, None, 3, None, 5], dtype="Int32"),),
            id="nullable_int32_array",
        ),
        pytest.param(
            (pd.Series([2**60, None, 3, None, 5], dtype="Int64"),),
            id="nullable_int64_array",
        ),
        pytest.param(
            (
                pd.Series(
                    [2**-150, 2**130, (2 - 2**-24) * (2**-128)], dtype="float64"
                ),
            ),
            id="float64_precision_array",
        ),
        pytest.param((pd.Series(["1", "2", "3", "4", "5"]),), id="str_int_array"),
        pytest.param(
            (
                pd.Series(
                    [
                        "1.1",
                        "3.5",
                        "nan",
                        "4.5",
                        "-inf",
                        "-8454757700450211157",
                        "12379739850550389709",
                    ]
                ),
            ),
            id="str_float_array",
        ),
        pytest.param(
            (pd.Series([True, False, True, False, True]),),
            id="bool_array",
        ),
        pytest.param(
            (pd.Series([True, False, None, False, True], dtype="boolean"),),
            id="nullable_bool_array",
        ),
        pytest.param(
            (11,),
            id="int32_scalar",
        ),
        pytest.param(
            (2**60,),
            id="int64_scalar",
        ),
        pytest.param(
            (14.0,),
            id="float32_scalar",
        ),
        pytest.param(
            (np.float64(2**130),),
            id="float64_scalar",
        ),
        pytest.param(
            ("52.8523",),
            id="str_float_scalar",
        ),
        pytest.param(
            ("-inf",),
            id="str_inf_scalar",
        ),
        pytest.param(
            ("-234",),
            id="str_int_scalar",
        ),
        pytest.param(
            (True,),
            id="bool_scalar",
        ),
    ]
)
def numeric_arrays(request):
    return request.param


def test_cast_float64(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_float64(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_float64(arr)

    # Simulates casting to float64 on a single row
    def float64_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return np.float64(elem)

    answer = vectorized_sol(args, float64_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_float32(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_float32(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_float32(arr)

    # Simulates casting to float32 on a single row
    def float32_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return np.float32(elem)

    answer = vectorized_sol(args, float32_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


def _round_float(f):
    return np.int64(np.fix(f + 0.5 * np.sign(f)))


def test_cast_int64(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_int64(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_int64(arr)

    # Simulates casting to float64 on a single row
    def int64_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elem = np.float64(elem)
        if (
            pd.isna(elem)
            or np.isinf(elem)
            or elem < np.iinfo(np.int64).min
            or elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            # additionally, we must round half away from zero rounding rather than
            # normal banker's rounding in Python / Numpy
            return _round_float(elem)

    answer = vectorized_sol(args, int64_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int64")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int32(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_int32(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_int32(arr)

    # Simulates casting to float64 on a single row
    def int32_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elem = np.float64(elem)
        if (
            pd.isna(elem)
            or np.isinf(elem)
            or elem < np.iinfo(np.int64).min
            or elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            return np.int32(_round_float(elem))

    answer = vectorized_sol(args, int32_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int32")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int16(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_int16(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_int16(arr)

    # Simulates casting to float64 on a single row
    def int16_scalar_fn(elem):
        if pd.isna(elem):
            return None
        f_elem = np.float64(elem)
        if (
            pd.isna(f_elem)
            or np.isinf(f_elem)
            or f_elem < np.iinfo(np.int64).min
            or f_elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            if f_elem.is_integer():
                return np.int16(elem)
            else:
                return np.int16(_round_float(f_elem))

    answer = vectorized_sol(args, int16_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int16")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int8(numeric_arrays):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.cast_int8(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.cast_int8(arr)

    # Simulates casting to float64 on a single row
    def int8_scalar_fn(elem):
        if pd.isna(elem):
            return None
        f_elem = np.float64(elem)
        if (
            pd.isna(f_elem)
            or np.isinf(f_elem)
            or f_elem < np.iinfo(np.int64).min
            or f_elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            if f_elem.is_integer():
                return np.int8(elem)
            else:
                return np.int8(_round_float(f_elem))

    answer = vectorized_sol(args, int8_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int8")
    # breakpoint()
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


@pytest.mark.slow
def test_cast_float_opt():
    def impl(a, b, flag0, flag1):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.cast_float32(arg0),
            bodo.libs.bodosql_array_kernels.cast_float64(arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                np.float32(12) if flag0 else None,
                np.float64(12) if flag1 else None,
            )
            check_func(
                impl,
                (12, 12, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_cast_int_opt():
    def impl(a, b, c, d, flag0, flag1, flag2, flag3):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        arg2 = c if flag2 else None
        arg3 = d if flag3 else None
        return (
            bodo.libs.bodosql_array_kernels.cast_int8(arg0),
            bodo.libs.bodosql_array_kernels.cast_int16(arg1),
            bodo.libs.bodosql_array_kernels.cast_int32(arg2),
            bodo.libs.bodosql_array_kernels.cast_int64(arg3),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    answer = (
                        np.int8(136) if flag0 else None,
                        np.int16(12) if flag1 else None,
                        np.int32(_round_float(140.5)) if flag2 else None,
                        np.int64(12) if flag3 else None,
                    )
                    check_func(
                        impl,
                        (-120, 12, 141, 12, flag0, flag1, flag2, flag3),
                        py_output=answer,
                    )
