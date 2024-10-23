"""Test that inputs and outputs are supported by spawn mode"""

import numpy as np
import pytest

import bodo
from bodo.submit.spawner import get_num_workers
from bodo.tests.utils import check_func


@pytest.mark.skip("submit_jit does not support inputs")
def test_distributed_input_scalar():
    def test(i):
        return i

    check_func(test, 42, use_spawn_mode=True)


@pytest.mark.skip("submit_jit does not support inputs")
def test_distributed_input_array():
    def test(A):
        s = A.sum()
        return s

    A = np.ones(1000, dtype=np.int64)
    check_func(test, A, use_spawn_mode=True)


@pytest.mark.skip("submit_jit does not support output")
def test_distributed_scalar_output():
    def test():
        return 1

    check_func(test, (), use_spawn_mode=True)


@pytest.mark.skip("submit_jit does not support inputs/output")
def test_distributed_input_output_df(df_value):
    def test(df):
        return df

    check_func(test, df_value, use_spawn_mode=True)


@pytest.mark.skip("submit_jit does not support inputs/output")
def test_distributed_output():
    def test(A):
        return A + 1

    A = np.random.randn(1000)
    check_func(test, A, use_spawn_mode=True)


@pytest.mark.skip("submit_jit does not support inputs/output")
def test_spawn_distributed():
    @bodo.jit(distributed={"A"}, spawn_mode=True)
    def test(A):
        s = A.sum()
        return s

    A = np.ones(1000, dtype=np.int64)
    assert test(A) == (get_num_workers() * 1000)
