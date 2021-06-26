# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Test Bodo's binary array data type
"""
import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.array([b"abc", b"c", np.nan, b"ccdefg" b"abcde", b"poiu"] * 2, object),
    ]
)
def binary_arr_value(request):
    return request.param


def test_unbox(binary_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (binary_arr_value,))
    check_func(impl2, (binary_arr_value,))


@pytest.mark.slow
def test_len(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (binary_arr_value,))


@pytest.mark.slow
def test_shape(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (binary_arr_value,))


@pytest.mark.slow
def test_ndim(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (binary_arr_value,))


def test_copy(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (binary_arr_value,))


def test_constant_lowering(binary_arr_value):
    def test_impl():
        return binary_arr_value

    check_func(test_impl, (), only_seq=True)


def test_hex_method(binary_arr_value):
    def test_impl(A):
        return pd.Series(A).apply(lambda x: None if pd.isna(x) else x.hex())

    check_func(test_impl, (binary_arr_value,))


def test_get_item(binary_arr_value):
    def test_impl(A, idx):
        return A[idx]

    np.random.seed(0)

    # A single integer
    # TODO [BE-927]: Fix distributed for integer
    idx = 0
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)

    # Array of integers
    idx = np.random.randint(0, len(binary_arr_value), 11)
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)

    # Array of booleans
    idx = np.random.ranf(len(binary_arr_value)) < 0.2
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)

    # Slice
    idx = slice(1, 4)
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)
