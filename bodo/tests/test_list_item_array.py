# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of list of fixed size items.
"""
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.array([[1, 3], [2], None, [4, 5, 6], [], [1, 1]]),
        np.array([[2.0, -3.2], [2.2, 1.3], None, [4.1, 5.2, 6.3], [], [1.1, 1.2]]),
        np.array([[True, False], [False, False], None, [True, False, True] * 4, [], [True, True]]),
    ]
)
def list_item_arr_value(request):
    return request.param


def test_unbox(list_item_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (list_item_arr_value,))
    check_func(impl2, (list_item_arr_value,))


def test_getitem_int(list_item_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    bodo_out = np.array(bodo.jit(test_impl)(list_item_arr_value, i))
    py_out = np.array(test_impl(list_item_arr_value, i))
    np.testing.assert_almost_equal(bodo_out, py_out)


def test_getitem_bool(list_item_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(0)
    ind = np.random.ranf(len(list_item_arr_value)) < 0.2
    bodo_out = bodo.jit(test_impl)(list_item_arr_value, ind)
    py_out = test_impl(list_item_arr_value, ind)
    pd.testing.assert_series_equal(
        pd.Series(py_out), pd.Series(bodo_out), check_dtype=False
    )


def test_getitem_slice(list_item_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    bodo_out = bodo.jit(test_impl)(list_item_arr_value, ind)
    py_out = test_impl(list_item_arr_value, ind)
    pd.testing.assert_series_equal(
        pd.Series(py_out), pd.Series(bodo_out), check_dtype=False
    )


def test_ndim():
    def test_impl(A):
        return A.ndim

    A = np.array([[1, 2, 3], [2]])
    assert bodo.jit(test_impl)(A) == test_impl(A)


def test_shape():
    def test_impl(A):
        return A.shape

    A = np.array([[1, 2, 3], [2], None, []])
    assert bodo.jit(test_impl)(A) == test_impl(A)


def test_copy(list_item_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (list_item_arr_value,))
