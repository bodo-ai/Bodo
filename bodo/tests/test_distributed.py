import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end)


@pytest.mark.parametrize('A', [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape1(A):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape[0]

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')


def test_array_shape2():
    # get first dimention size using array.shape for distributed arrays
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape[1]

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')
    # TODO: test Array.ctypes.shape[0] cases


@pytest.mark.parametrize('A', [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape3(A):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')


def test_array_shape4():
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')


def test_array_len1():
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return len(A)

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')
    # TODO: tests with array created inside the function


@pytest.mark.parametrize('A', [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_size1(A):
    def impl1(A):
        return A.size

    bodo_func = bodo.jit(distributed={'A'})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains('dist_reduce')
    # TODO: tests with array created inside the function


@pytest.mark.skip(reason="TODO: fix 1D Var array and parfors")
@pytest.mark.parametrize('A', [np.arange(11), np.arange(33).reshape(11, 3)])
def test_1D_Var_parfor(A):
    # 1D_Var parfor where index is used in computation
    def impl1(A, B):
        C = A[B != 0]
        s = 0
        for i in bodo.prange(len(C)):
            s += i + C[i]
        return s

    bodo_func = bodo.jit(distributed={'A', 'B'})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_print1():
    # no vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, 2)
    bodo_func(np.ones(3), 3)
    bodo_func((3, 4), 2)


def test_print2():
    # vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a):
        print(*a)

    bodo_func = bodo.jit()(impl1)
    bodo_func((3, 4))
    bodo_func((3, np.ones(3)))


def test_print3():
    # arg and vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, *b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, (3, 4))
    bodo_func(np.ones(3), (3, np.ones(3)))
