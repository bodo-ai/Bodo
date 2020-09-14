# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Tests for pd.CategoricalDtype/pd.Categorical  functionality
"""
import pandas as pd
import numpy as np
import bodo
import pytest
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "dtype",
    [
        pd.CategoricalDtype(["AA", "B", "CC"]),
        pytest.param(
            pd.CategoricalDtype(["CC", "AA", "B"], True), marks=pytest.mark.slow
        ),
        pytest.param(pd.CategoricalDtype([3, 2, 1, 4]), marks=pytest.mark.slow),
    ],
)
def test_unbox_dtype(dtype, memory_leak_check):
    # just unbox
    def impl(dtype):
        return True

    check_func(impl, (dtype,))

    # unbox and box
    def impl2(dtype):
        return dtype

    check_func(impl2, (dtype,))


@pytest.fixture(
    params=[
        pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"]),
        pytest.param(
            pd.Categorical(["CC", "AA", None, "B", "D", "AA", "B", "CC"], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2]), marks=pytest.mark.slow),
        pytest.param(
            pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2], ordered=True),
            marks=pytest.mark.slow,
        ),
    ]
)
def cat_arr_value(request):
    return request.param


def test_unbox_cat_arr(cat_arr_value, memory_leak_check):
    # just unbox
    def impl(arr):
        return True

    check_func(impl, (cat_arr_value,))

    # unbox and box
    def impl2(arr):
        return arr

    check_func(impl2, (cat_arr_value,))


def test_getitem_int(cat_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    assert bodo.jit(test_impl)(cat_arr_value, i) == test_impl(cat_arr_value, i)


def test_getitem_bool(cat_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(1)
    ind = np.random.ranf(len(cat_arr_value)) < 0.2
    bodo_out = bodo.jit(test_impl)(cat_arr_value, ind)
    py_out = test_impl(cat_arr_value, ind)
    pd.testing.assert_extension_array_equal(py_out, bodo_out)


@pytest.mark.slow
def test_getitem_slice(cat_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    bodo_out = bodo.jit(test_impl)(cat_arr_value, ind)
    py_out = test_impl(cat_arr_value, ind)
    pd.testing.assert_extension_array_equal(py_out, bodo_out)
