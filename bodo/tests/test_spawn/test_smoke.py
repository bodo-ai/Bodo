import numpy as np

from bodo.submit.spawner import submit_jit
from bodo.tests.utils import pytest_spawn_mode

pytestmark = pytest_spawn_mode


@submit_jit
def setitem_jit(A):
    A[0] = 1


def test_setitem(memory_leak_check):
    @submit_jit
    def do_test():
        arr = np.zeros(1)
        setitem_jit(arr)
        if arr[0] == 0:
            raise Exception("test failed")

    do_test()


def test_getitem(memory_leak_check):
    @submit_jit
    def test_impl():
        N = 128
        A = np.ones(N)
        B = np.ones(N) > 0.5
        C = A[B]
        if C.sum() != 128:
            raise Exception("test failed")

    test_impl()
