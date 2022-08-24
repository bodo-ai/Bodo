# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on numeric
column elements.

Test names refer to the names of the spark function they map to.
"""

import math

import numpy as np
import pandas as pd
import pytest
import scipy.special

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 3, 10, 15, 20, 11, 19] * 10,
                    "B": [201, 30, -10, 15, -30, -1000, 100] * 10,
                }
            )
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.slow
def test_cbrt(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.cbrt(x))

    # Numpy uses different floating point libaries in
    # different platforms so precision may vary. This
    # should be fixed when numba adds support for it.
    check_func(test_impl, (dataframe_val,), atol=2e-06)


@pytest.mark.slow
def test_factorial(dataframe_val, memory_leak_check):
    """Factorial limit for a 64 bit integer is 20"""

    def test_impl1(df):
        return df.A.map(lambda x: math.factorial(x))

    def test_impl2(df):
        return df.A.map(lambda x: np.math.factorial(x))

    def test_impl3(df):
        return df.A.map(lambda x: scipy.special.factorial(x, True))

    check_func(test_impl1, (dataframe_val,))
    check_func(test_impl2, (dataframe_val,))
    check_func(test_impl3, (dataframe_val,))


@pytest.mark.slow
def test_shiftLeft(dataframe_val, memory_leak_check):
    def impl(df, num_bits):
        return np.left_shift(df.A, num_bits)

    check_func(impl, (dataframe_val, 0))
    check_func(impl, (dataframe_val, 1))
    check_func(impl, (dataframe_val, 15))


@pytest.mark.slow
def test_shiftLeft_uint64(memory_leak_check):
    """Known numba bug prevents shifts on uint64 values. As a
    result we need to manually convert the dtype.
    """

    def impl(df, num_bits):
        return np.left_shift(df.A.astype(np.int64), num_bits).astype(np.uint64)

    df = pd.DataFrame(
        {
            "A": [2, 3, 0xFFFFFFFFFFFFFFF1, 15, 20, 0xFFFFFFFFFFFFFFFF, 19] * 10,
        },
        dtype=np.uint64,
    )
    check_func(impl, (df, 0))
    check_func(impl, (df, 1))
    check_func(impl, (df, 15))


@pytest.mark.slow
def test_shiftRight(dataframe_val, memory_leak_check):
    def impl(df, num_bits):
        return np.left_shift(df.B, num_bits)

    check_func(impl, (dataframe_val, 0))
    check_func(impl, (dataframe_val, 1))
    check_func(impl, (dataframe_val, 15))


@pytest.mark.slow
def test_shiftRight_uint64(memory_leak_check):
    """Known numba bug prevents shifts on uint64 values. As a
    result we need to manually convert the dtype.
    """

    def impl(df, num_bits):
        bits_minus_1 = max((num_bits - 1), 0)
        mask_bits = (np.int64(1) << bits_minus_1) - 1
        mask = ~(mask_bits << (63 - bits_minus_1))
        return (np.right_shift(df.A.astype(np.int64), num_bits) & mask).astype(
            np.uint64
        )

    df = pd.DataFrame(
        {
            "A": [2, 3, 0xFFFFFFFFFFFFFFF1, 15, 20, 0xFFFFFFFFFFFFFFFF, 19] * 10,
        },
        dtype=np.uint64,
    )
    check_func(impl, (df, 0))
    check_func(impl, (df, 1))
    check_func(impl, (df, 15))


@pytest.mark.slow
def test_shiftRightUnsigned(dataframe_val, memory_leak_check):
    """Known numba bug prevents shifts on uint64 values. As a
    result we need to manually convert the dtype.
    """

    def impl(df, num_bits):
        bits_minus_1 = max((num_bits - 1), 0)
        mask_bits = (np.int64(1) << bits_minus_1) - 1
        mask = ~(mask_bits << (63 - bits_minus_1))
        return (np.right_shift(df.A.astype(np.int64), num_bits) & mask).astype(
            np.uint64
        )

    check_func(impl, (dataframe_val, 0))
    check_func(impl, (dataframe_val, 1))
    check_func(impl, (dataframe_val, 15))
