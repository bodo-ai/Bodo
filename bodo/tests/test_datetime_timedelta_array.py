# Copyright (C) 2020 Bodo Inc. All rights reserved.
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


def test_getitem_int(timedelta_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    bodo_func = bodo.jit(test_impl)
    i = 0
    assert bodo_func(timedelta_arr_value, i) == test_impl(timedelta_arr_value, i)


def test_getitem_bool(timedelta_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    np.random.seed(0)
    ind = np.random.ranf(len(timedelta_arr_value)) < 0.2
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(timedelta_arr_value, ind), test_impl(timedelta_arr_value, ind)
    )


def test_getitem_slice(timedelta_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = slice(1, 4)
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(timedelta_arr_value, ind), test_impl(timedelta_arr_value, ind)
    )


def test_setitem_slice(timedelta_arr_value, memory_leak_check):
    def test_impl(A, ind, vals):
        A[ind] = vals
        return A

    ind = slice(3, 8)
    vals = [datetime.timedelta(days=x, seconds=4, weeks=4) for x in range(5)]
    # TODO: parallel test
    check_func(
        test_impl, (timedelta_arr_value, ind, vals), dist_test=False, copy_input=True
    )


def test_getitem_int_arr(timedelta_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = np.array([1, 2])
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(timedelta_arr_value, ind), test_impl(timedelta_arr_value, ind)
    )


def test_setitem_optional_int(timedelta_arr_value, memory_leak_check):
    def test_impl(A, i, flag):
        if flag:
            x = None
        else:
            x = datetime.timedelta(microseconds=100000001213181, hours=5)
        A[i] = x
        return A

    check_func(
        test_impl,
        (timedelta_arr_value.copy(), 1, False),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (timedelta_arr_value.copy(), 0, True),
        copy_input=True,
        dist_test=False,
    )


def test_setitem_none_int(timedelta_arr_value, memory_leak_check):
    def test_impl(A, i):
        A[i] = None
        return A

    i = 0
    check_func(
        test_impl, (timedelta_arr_value.copy(), i), copy_input=True, dist_test=False
    )


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
    check_func(impl, (arr,), sort_output=True)


# TODO: fix memory leak and add memory_leak_check
def test_constant_lowering(timedelta_arr_value):
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
