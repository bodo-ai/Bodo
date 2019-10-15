# Copyright (C) 2019 Bodo Inc.
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end,
    check_func)
np.random.seed(0)


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


def test_int_dtype():
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.Int32Dtype(),))
    check_func(impl, (pd.Int8Dtype(),))
    check_func(impl, (pd.UInt32Dtype(),))

    # constructors
    def impl2():
        return pd.Int8Dtype()

    check_func(impl2, ())

    def impl3():
        return pd.UInt32Dtype()

    check_func(impl3, ())


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
    check_func(test_impl, (A,))


def test_unary_ufunc_explicit_np():
    def test_impl(A):
        return np.negative(A)

    A = pd.arrays.IntegerArray(np.array([1, 1, 1, -3, 10], np.int32),
                                np.array([False, True, True, False, False]))
    check_func(test_impl, (A,))


@pytest.mark.parametrize('ufunc',
[f for f in numba.targets.ufunc_db.get_ufuncs() if f.nin == 2])
def test_binary_ufunc(ufunc):
    def test_impl(A1, A2):
        return ufunc(A1, A2)

    A1 = pd.arrays.IntegerArray(np.array([1, 1, 1, 2, 10], np.int32),
        np.array([False, True, True, False, False]))
    A2 = pd.arrays.IntegerArray(np.array([4, 2, 1, 1, 12], np.int32),
        np.array([False, False, True, True, False]))
    arr = np.array([1, 3, 7, 11, 2])
    check_func(test_impl, (A1, A2))
    check_func(test_impl, (A1, arr))
    check_func(test_impl, (arr, A2))


@pytest.mark.parametrize('op',
    numba.typing.npydecl.NumpyRulesArrayOperator._op_map.keys())
def test_binary_op(op):
    # Pandas doesn't support these operators yet, but Bodo does to be able to
    # replace all numpy arrays
    if op in (operator.lshift, operator.rshift, operator.and_, operator.or_,
              operator.xor):
        return
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += "  return A {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars['test_impl']

    # TODO: use int32 when Numba #4449 is resolved
    A1 = pd.arrays.IntegerArray(np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]))
    A2 = pd.arrays.IntegerArray(np.array([4, 2, 1, 1, 12], np.int64),
        np.array([False, False, True, True, False]))
    check_func(test_impl, (A1, A2))
    check_func(test_impl, (A1, 2))
    check_func(test_impl, (2, A2))


@pytest.mark.parametrize('op',
    numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys())
def test_inplace_binary_op(op):
    # Numba can't handle itruediv
    # pandas doesn't support the others
    if op in (operator.ilshift, operator.irshift, operator.iand, operator.ior,
              operator.ixor, operator.itruediv):
        return
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += "  A {} other\n".format(op_str)
    func_text += "  return A\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars['test_impl']

    A1 = pd.arrays.IntegerArray(np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]))
    A2 = pd.arrays.IntegerArray(np.array([4, 2, 1, 1, 12], np.int64),
        np.array([False, False, True, True, False]))
    # TODO: test inplace change properly
    check_func(test_impl, (A1, A2), copy_input=True)
    check_func(test_impl, (A1, 2), copy_input=True)


@pytest.mark.skip(reason="pd.arrays.IntegerArray doesn't support unary op")
@pytest.mark.parametrize('op', (operator.neg, operator.invert, operator.pos))
def test_unary_op(op):
    # TODO: fix operator.pos
    if op == operator.pos:
        return

    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A):\n"
    func_text += "  return {} A\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars['test_impl']

    A = pd.arrays.IntegerArray(np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]))
    check_func(test_impl, (A,))


def test_dtype(int_arr_value):

    def test_impl(A):
        return A.dtype

    check_func(test_impl, (int_arr_value,))


def test_ndim():

    def test_impl(A):
        return A.ndim

    A = pd.arrays.IntegerArray(np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]))
    check_func(test_impl, (A,))


def test_copy(int_arr_value):

    def test_impl(A):
        return A.copy()

    check_func(test_impl, (int_arr_value,))


@pytest.mark.parametrize('dtype',
    [pd.Int8Dtype(), np.float64])
def test_astype(int_arr_value, dtype):

    def test_impl(A, dtype):
        return A.astype(dtype)

    check_func(test_impl, (int_arr_value, dtype))


def test_astype_str(int_arr_value):

    def test_impl(A):
        return A.astype('float64')

    check_func(test_impl, (int_arr_value,))


def test_unique(int_arr_value):

    def test_impl(A):
        return A.unique()

    # only sequential check since not directly parallelized
    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        bodo_func(int_arr_value), test_impl(int_arr_value))
