# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(pd.Float32Dtype(), id="float32"),
        pytest.param(pd.Float64Dtype(), id="float64"),
    ]
)
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param([], id="empty"),
        pytest.param([float(i) for i in range(30)], id="all_floats"),
        pytest.param([1.0, None, 3.0] * 10, id="floats_and_nulls"),
        pytest.param([None] * 30, id="all_nulls"),
        pytest.param([1.0, None, np.nan] * 10, id="floats_nulls_and_nans"),
        pytest.param([np.nan] * 30, id="all_nans"),
    ]
)
def nullable_float_values(request):
    return request.param


@pytest.mark.slow
def test_float_arr_unbox(nullable_float_values, dtype, memory_leak_check):
    def impl(arr):
        return 42

    arr = pd.array(nullable_float_values, dtype=dtype)

    check_func(impl, (arr,))


@pytest.mark.slow
def test_float_arr_box_unbox(nullable_float_values, dtype, memory_leak_check):
    def impl(arr):
        return arr

    check_func(impl, (pd.array(nullable_float_values, dtype=dtype),))


@pytest.mark.slow
def test_float_arr_isna(nullable_float_values, dtype, memory_leak_check):
    def impl(arr):
        return pd.isna(arr)

    check_func(impl, (pd.array(nullable_float_values, dtype=dtype),))


@pytest.mark.slow
def test_float_arr_getitem(nullable_float_values, dtype, memory_leak_check):
    arr = pd.array(nullable_float_values, dtype=dtype)

    if len(nullable_float_values) == 0:
        pytest.skip(reason="empty array")

    def impl(arr):
        return arr[0]

    check_func(
        impl,
        (arr,),
        py_output=arr[0] if arr[0] is not pd.NA else np.nan,
    )


@pytest.mark.slow
def test_float_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.Float32Dtype(),))
    check_func(impl, (pd.Float64Dtype(),))

    # constructors
    def impl2():
        return pd.Float32Dtype()

    check_func(impl2, ())

    def impl3():
        return pd.Float64Dtype()

    check_func(impl3, ())


@pytest.mark.slow
def test_float_arr_len(memory_leak_check):
    def test_impl(A):
        return len(A)

    A = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_shape(memory_leak_check):
    def test_impl(A):
        return A.shape

    A = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_dtype(nullable_float_values, dtype, memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (pd.array(nullable_float_values, dtype=dtype),))


@pytest.mark.slow
def test_float_arr_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    A = pd.arrays.FloatingArray(
        np.array([1.0, 1.0, 1.0, 2.0, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_copy(nullable_float_values, dtype, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (pd.array(nullable_float_values, dtype=dtype),))


@pytest.mark.slow
def test_float_arr_constant_lowering(nullable_float_values, dtype, memory_leak_check):
    arr = pd.array(nullable_float_values, dtype=dtype)

    def impl():
        return arr

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(arr), check_dtype=False
    )


@pytest.mark.slow
def test_float_arr_nbytes(memory_leak_check):
    def impl(A):
        return A.nbytes

    arr = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    py_out = 40 + bodo.get_size()  # 1 extra byte for null_bit_map per rank
    check_func(impl, (arr,), py_output=py_out, only_1D=True)
    check_func(impl, (arr,), py_output=41, only_seq=True)
