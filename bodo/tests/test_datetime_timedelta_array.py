# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test File for timedelta array types. Covers basic functionality of get_item
    operations, but it is not comprehensive. It does not cover exception cases
    or test extensively against None.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.append(
            datetime.timedelta(days=5, seconds=4, weeks=4),
            [None, datetime.timedelta(microseconds=100000001213131, hours=5)] * 5,
        )
    ]
)
def timedelta_arr_value(request):
    return request.param


def test_np_repeat(timedelta_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (timedelta_arr_value,))


@pytest.mark.skip("TODO(Nick): Add support for timedelta arrays inside array_to_info")
def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = np.append(
        datetime.timedelta(days=5, seconds=4, weeks=4),
        [datetime.timedelta(microseconds=100000001213131, hours=5)] * 5,
    )
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_constant_lowering(timedelta_arr_value, memory_leak_check):
    def impl():
        return timedelta_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(timedelta_arr_value), check_dtype=False
    )


@pytest.mark.skip("TODO(Nick): Add support for timedelta arrays inside C++ code")
def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = np.append(
        datetime.timedelta(days=5, seconds=4, weeks=4),
        [datetime.timedelta(microseconds=100000001213131, hours=5)] * 20,
    )

    check_func(impl, (A,))


@pytest.mark.smoke
def test_setitem_int(timedelta_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    val = timedelta_arr_value[0]
    check_func(test_impl, (timedelta_arr_value, val))


@pytest.mark.smoke
def test_setitem_arr(timedelta_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(timedelta_arr_value), 11)
    val = pd.timedelta_range(start="7 hours", periods=len(idx)).to_pytimedelta()
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single datetime.timedelta as a value, reuses the same idx
    val = datetime.timedelta(seconds=10)
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = np.random.ranf(len(timedelta_arr_value)) < 0.2
    val = pd.timedelta_range(start="7 hours", periods=idx.sum()).to_pytimedelta()
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single datetime.timedelta as a value, reuses the same idx
    val = datetime.timedelta(seconds=10)
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = slice(1, 4)
    val = pd.timedelta_range(start="7 hours", periods=3).to_pytimedelta()
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single datetime.timedelta as a value, reuses the same idx
    val = datetime.timedelta(seconds=10)
    check_func(
        test_impl, (timedelta_arr_value, idx, val), dist_test=False, copy_input=True
    )


@pytest.mark.slow
def test_nbytes(timedelta_arr_value, memory_leak_check):
    """test DatetimeTimeDeltaArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    py_out = (
        264 + bodo.get_size()
    )  # 88*3 = 264 for data (days, seconds, microseconds), one byte for null_bitmap per rank
    check_func(impl, (timedelta_arr_value,), py_output=266, only_seq=True)
    if bodo.get_size() == 1:  # np=1 has 2 bytes for null_bitmap
        py_out += 1
    check_func(impl, (timedelta_arr_value,), py_output=py_out, only_1DVar=True)
