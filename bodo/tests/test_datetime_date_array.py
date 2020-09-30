# Copyright (C) 2020 Bodo Inc. All rights reserved.
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


def test_getitem_int(date_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    bodo_func = bodo.jit(test_impl)
    i = 1
    assert bodo_func(date_arr_value, i) == test_impl(date_arr_value, i)


def test_getitem_bool(date_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    np.random.seed(0)
    ind = np.random.ranf(len(date_arr_value)) < 0.2
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(date_arr_value, ind), test_impl(date_arr_value, ind)
    )


def test_getitem_slice(date_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = slice(1, 4)
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(date_arr_value, ind), test_impl(date_arr_value, ind)
    )


def test_getitem_int_arr(date_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = np.array([2, 3])
    # TODO: parallel test
    np.testing.assert_array_equal(
        bodo_func(date_arr_value, ind), test_impl(date_arr_value, ind)
    )


def test_setitem_optional_int(date_arr_value, memory_leak_check):
    def test_impl(A, i, flag):
        if flag:
            x = None
        else:
            x = datetime.date(2020, 9, 8)
        A[i] = x
        return A

    check_func(
        test_impl, (date_arr_value.copy(), 1, False), copy_input=True, dist_test=False
    )
    check_func(
        test_impl, (date_arr_value.copy(), 0, True), copy_input=True, dist_test=False
    )


def test_setitem_none_int(date_arr_value, memory_leak_check):
    def test_impl(A, i):
        A[i] = None
        return A

    i = 1
    check_func(test_impl, (date_arr_value.copy(), i), copy_input=True, dist_test=False)


def test_np_repeat(date_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (date_arr_value,), dist_test=False)


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = np.append(
        pd.date_range("2017-01-03", "2017-01-17").date,
        [datetime.date(2019, 3, 3)] * 10,
    )
    check_func(impl, (arr,), sort_output=True)
