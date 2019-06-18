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
