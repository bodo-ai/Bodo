# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests for the null array type, which is an array of all nulls
that can be cast to any type. See null_arr_ext.py for the
core implementation.
"""
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


def test_nullable_bool_cast(memory_leak_check):
    """
    Tests casting a nullable array to a boolean array.
    """

    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr.astype(pd.BooleanDtype())

    n = 10
    arr = pd.array([None] * n, dtype=pd.BooleanDtype())
    check_func(impl, [n], py_output=arr)


def test_null_arr_getitem(memory_leak_check):
    """Test getitem with nullable arrays

    Args:
        memory_leak_check: A context manager fixture that makes sure there is no memory leak in the test.
    """

    def impl(n, idx):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return [null_arr[idx]]

    n = 10
    np.random.seed(0)

    # A single integer
    idx = 0
    check_func(
        impl,
        (n, idx),
        py_output=[None],
        check_dtype=False,
        dist_test=False,
    )

    def impl2(n, idx):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr[idx]

    # Array of integers
    idx = np.random.randint(0, n, 11)
    check_func(
        impl2,
        (n, idx),
        py_output=pd.array([None] * len(idx), dtype=pd.BooleanDtype()),
        check_dtype=False,
        dist_test=False,
    )

    # Array of booleans
    idx = [True, False, True, False, False]
    arr = pd.array([None] * len(idx))
    expected_output = arr[idx]
    check_func(
        impl2,
        (n, idx),
        py_output=expected_output,
        check_dtype=False,
        dist_test=False,
    )

    # slice
    idx = slice(5)
    check_func(
        impl2,
        (n, idx),
        py_output=pd.array([None] * 5, dtype=pd.BooleanDtype()),
        check_dtype=False,
    )


def test_isna_check(memory_leak_check):
    """
    Test that isna works properly with NullArrayType
    """

    def test_impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return pd.isna(null_arr)

    n = 10
    py_output = pd.array([True] * n)
    check_func(test_impl, (n,), py_output=py_output)


@pytest.mark.slow
def test_astype_check(memory_leak_check):
    """
    Test that astype works properly with NullArrayType
    """

    def test_impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        S = pd.Series(null_arr)
        return S.astype(np.dtype("datetime64[ns]"))

    n = 10
    py_output = pd.array([None] * n, dtype=pd.BooleanDtype())
    check_func(test_impl, (n,), py_output=py_output)


@pytest.mark.slow
def test_nullarray_to_char():
    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)

        return pd.Series(bodo.libs.bodosql_array_kernels.to_char(null_arr))

    n = 10
    py_output = pd.Series([None] * n)
    check_func(impl, (n,), py_output=py_output)
