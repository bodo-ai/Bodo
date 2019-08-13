import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end,
    check_func)


@pytest.fixture(params = [
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int32),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int64),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint8),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint32),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint64),
        np.array([False, True, True, False, False])),
    # large array
    pd.arrays.IntegerArray(np.random.randint(0, 100, 1211),
        np.random.ranf(1211)<.3),
])
def int_arr_value(request):
    return request.param


def test_unbox(int_arr_value):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (int_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (int_arr_value,))


def test_getitem_int(int_arr_value):

    def test_impl(A, i):
        return A[i]

    bodo_func = bodo.jit(test_impl)
    i = 1
    # make sure the element is not NA
    int_arr_value._mask[i] = False
    assert bodo_func(int_arr_value, i) == test_impl(int_arr_value, i)


def test_getitem_bool(int_arr_value):

    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    np.random.seed(0)
    ind = np.random.ranf(len(int_arr_value)) < .2
    # TODO: parallel test
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, ind), test_impl(int_arr_value, ind))


def test_getitem_slice(int_arr_value):

    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = slice(1, 4)
    # TODO: parallel test
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, ind), test_impl(int_arr_value, ind))


def test_setitem_int(int_arr_value):

    def test_impl(A, val):
        A[2] = val
        return A

    # get a non-null value
    int_arr_value._mask[0] = False
    val = int_arr_value[0]
    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, val), test_impl(int_arr_value, val))


def test_setitem_arr(int_arr_value):

    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(int_arr_value), 11)
    val = np.random.randint(0, 50, 11, int_arr_value._data.dtype)
    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx, val), test_impl(int_arr_value, idx, val))
    # IntegerArray as value
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < .2)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx, val), test_impl(int_arr_value, idx, val))

    idx_bool = np.random.ranf(len(int_arr_value)) < .2
    val = np.random.randint(0, 50, idx_bool.sum(), int_arr_value._data.dtype)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx_bool, val),
        test_impl(int_arr_value, idx_bool, val))
    # IntegerArray as value
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < .2)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx_bool, val),
        test_impl(int_arr_value, idx_bool, val))

    idx = slice(1, 4)
    val = np.random.randint(0, 50, 3, int_arr_value._data.dtype)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx, val),
        test_impl(int_arr_value, idx, val))
    # IntegerArray as value
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < .2)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value, idx, val),
        test_impl(int_arr_value, idx, val))


def test_len():
    def test_impl(A):
        return len(A)

    A = pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int8),
                               np.array([False, True, True, False, False]))
    check_func(test_impl, (A,))


def test_shape():
    def test_impl(A):
        return A.shape

    A = pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int8),
                                np.array([False, True, True, False, False]))
    check_func(test_impl, (A,))


@pytest.mark.parametrize('ufunc',
[f for f in numba.targets.ufunc_db.get_ufuncs() if f.nin == 1])
def test_unary_ufunc(ufunc):
    def test_impl(A):
        return ufunc(A)

    A = pd.arrays.IntegerArray(np.array([1, 1, 1, -3, 10], np.int32),
                                np.array([False, True, True, False, False]))
    check_func(test_impl, (A,), is_out_distributed=False)


def test_unary_ufunc_explicit_np():
    def test_impl(A):
        return np.negative(A)

    A = pd.arrays.IntegerArray(np.array([1, 1, 1, -3, 10], np.int32),
                                np.array([False, True, True, False, False]))
    check_func(test_impl, (A,), is_out_distributed=False)
