# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.append(
            pd.date_range("2017-01-03", "2017-01-17").date,
            [None, datetime.date(2019, 3, 3)],
        )
    ]
)
def date_arr_value(request):
    return request.param


def test_np_repeat(date_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (date_arr_value,))


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = np.append(
        pd.date_range("2017-01-03", "2017-01-17").date,
        [datetime.date(2019, 3, 3)] * 10,
    )
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_constant_lowering(date_arr_value, memory_leak_check):
    def impl():
        return date_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(date_arr_value), check_dtype=False
    )


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = np.append(
        pd.date_range("2017-01-03", "2017-07-17").date,
        [datetime.date(2016, 3, 3)],
    )

    check_func(impl, (A,))


@pytest.mark.smoke
def test_setitem_int(date_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    # get a non-null value
    val = date_arr_value[0]
    check_func(test_impl, (date_arr_value, val))


@pytest.mark.smoke
def test_setitem_arr(date_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(date_arr_value), 11)
    val = pd.date_range("2021-02-21", periods=len(idx)).date
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single datetime.date as a value, reuses the same idx
    val = datetime.date(2021, 2, 11)
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = np.random.ranf(len(date_arr_value)) < 0.2
    val = pd.date_range("2021-02-21", periods=idx.sum()).date
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single datetime.date as a value, reuses the same idx
    val = datetime.date(2021, 2, 11)
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = slice(1, 4)
    val = pd.date_range("2021-02-21", periods=3).date
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single datetime.date as a value, reuses the same idx
    val = datetime.date(2021, 2, 11)
    check_func(test_impl, (date_arr_value, idx, val), dist_test=False, copy_input=True)


@pytest.mark.slow
def test_lower_constant_getitem(date_arr_value, memory_leak_check):
    """Test that lowering a constant datetime_date array with a
    None value has getitem support."""

    def test_impl(idx):
        # Lower A as a constant
        return A[idx]

    A = date_arr_value
    # Find the first nan value in the array
    idx = pd.Series(date_arr_value).isna().idxmax()

    # Test that we don't get a runtime error. The result is garbage,
    # so we ignore it.
    bodo.jit(test_impl)(
        idx,
    )


@pytest.mark.slow
def test_nbytes(date_arr_value, memory_leak_check):
    """Test DatetimeDateArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    py_out = 139  # 136 for data, 3 for null_bitmap
    check_func(impl, (date_arr_value,), py_output=py_out)
