# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for array of tuple values.
"""

import numpy as np
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        # heterogeneous values
        np.array(
            [
                (1, 3.1),
                (2, 1.1),
                None,
                (-1, 7.8),
                (3, 4.0),
                (-3, -1.2),
                (None, 9.0),
            ],
            dtype=object,
        ),
        # homogeneous values
        pytest.param(
            np.array(
                [
                    (1.1, 3.1),
                    (2.1, 1.1),
                    None,
                    (-1.1, -1.1),
                    (3.1, 4.1),
                    (-3.1, -1.1),
                    (5.1, 9.1),
                ],
                dtype=object,
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def tuple_arr_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(tuple_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (tuple_arr_value,))
    check_func(impl2, (tuple_arr_value,))


@pytest.mark.smoke
def test_getitem_int(tuple_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    check_func(test_impl, (tuple_arr_value, i), dist_test=False)


def test_getitem_bool(tuple_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(0)
    ind = np.random.ranf(len(tuple_arr_value)) < 0.2
    check_func(test_impl, (tuple_arr_value, ind), dist_test=False)


def test_getitem_slice(tuple_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    check_func(test_impl, (tuple_arr_value, ind), dist_test=False)


@pytest.mark.smoke
def test_setitem_int(tuple_arr_value, memory_leak_check):
    def test_impl(A, i, val):
        A[i] = val
        return A

    i = 1
    val = tuple_arr_value[0]
    check_func(test_impl, (tuple_arr_value, i, val), copy_input=True, dist_test=False)


def test_setitem_slice(memory_leak_check):
    def test_impl(A, i, val):
        A[i] = val
        return A

    A = np.array(
        [
            (1, 3.1),
            (2, 1.1),
            None,
            (-1, 7.8),
            (3, 4.0),
            (-3, -1.2),
        ]
    )
    i = slice(1, 3)
    val = A[:2]
    check_func(test_impl, (A, i, val), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_ndim(tuple_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    assert bodo.jit(test_impl)(tuple_arr_value) == test_impl(tuple_arr_value)


@pytest.mark.slow
def test_shape(tuple_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    assert bodo.jit(test_impl)(tuple_arr_value) == test_impl(tuple_arr_value)


@pytest.mark.slow
def test_dtype(tuple_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    assert bodo.jit(test_impl)(tuple_arr_value) == test_impl(tuple_arr_value)


@pytest.mark.slow
def test_copy(tuple_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (tuple_arr_value,))


@pytest.mark.slow
def test_len(tuple_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (tuple_arr_value,))


@pytest.mark.slow
def test_nested_str_tuples(memory_leak_check):
    """make sure nested tuples with variable size data can be converted to tuple array
    properly
    """

    def impl():
        return bodo.utils.conversion.coerce_to_array((("AA", "B"), ("C", "ABC")))

    check_func(impl, (), dist_test=False)


@pytest.mark.slow
def test_nbytes(memory_leak_check):
    """Test TupleArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    tuple_value = np.array(
        [
            (1.1, 3.1),
            (2.1, 1.1),
            None,
            (-1.1, -1.1),
            (3.1, 4.1),
            (-3.1, -1.1),
            (5.1, 9.1),
        ]
    )
    check_func(impl, (tuple_value,), py_output=113, only_seq=True)
    py_out = 112 + bodo.get_size()  # one byte for null_bitmap per rank
    check_func(impl, (tuple_value,), py_output=py_out, only_1DVar=True)
