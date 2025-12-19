import numpy as np

import bodo
from bodo.tests.utils import pytest_spawn_mode

pytestmark = pytest_spawn_mode


def test_setitem(memory_leak_check):
    @bodo.jit(spawn=True)
    def setitem_jit(A):
        A[0] = 1

    @bodo.jit(spawn=True)
    def do_test():
        arr = np.zeros(1)
        setitem_jit(arr)
        if arr[0] == 0:
            raise Exception("test failed")

    do_test()


def test_getitem(memory_leak_check):
    @bodo.jit(spawn=True)
    def test_impl():
        N = 128
        A = np.ones(N)
        B = np.ones(N) > 0.5
        C = A[B]
        if C.sum() != 128:
            raise Exception("test failed")

    test_impl()
