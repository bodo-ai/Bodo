# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


def test_floor_ceil_float(memory_leak_check):
    """
    Testing the FLOOR and CEIL functions with precisions, with hardcoded answers
    obtained from Snowflake, on float data.
    """

    def impl(S, p):
        return (
            pd.Series(bodo.libs.bodosql_array_kernels.ceil(S, p)),
            pd.Series(bodo.libs.bodosql_array_kernels.floor(S, p)),
        )

    S = pd.Series([3.1415926, 1.0, -72.5]).repeat(5)
    p = pd.Series([0, 1, -1, 3, -3] * 3)
    ceil_answer = pd.Series(
        [
            4.0,
            3.2,
            10.0,
            3.142,
            1000.0,
            1.0,
            1.0,
            10.0,
            1.0,
            1000.0,
            -72.0,
            -72.5,
            -70.0,
            -72.5,
            0.0,
        ]
    )
    floor_answer = pd.Series(
        [
            3.0,
            3.1,
            0.0,
            3.141,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            -73.0,
            -72.5,
            -80.0,
            -72.5,
            -1000.0,
        ]
    )
    check_func(
        impl,
        (S, p),
        py_output=(ceil_answer, floor_answer),
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_floor_ceil_int(memory_leak_check):
    """
    Testing the FLOOR and CEIL functions with precisions, with hardcoded answers
    obtained from Snowflake, on integer data.
    """

    def impl(S, p):
        return (
            pd.Series(bodo.libs.bodosql_array_kernels.ceil(S, p)),
            pd.Series(bodo.libs.bodosql_array_kernels.floor(S, p)),
        )

    S = pd.Series([10, 1024, -10625], dtype=pd.Int32Dtype()).repeat(2)
    p = pd.Series([1, -2] * 3)
    ceil_answer = pd.Series(
        [
            10,
            100,
            1024,
            1100,
            -10625,
            -10600,
        ],
        dtype=pd.Int32Dtype(),
    )
    floor_answer = pd.Series(
        [
            10,
            0,
            1024,
            1000,
            -10625,
            -10700,
        ],
        dtype=pd.Int32Dtype(),
    )
    check_func(
        impl,
        (S, p),
        py_output=(ceil_answer, floor_answer),
        check_dtype=False,
        reset_index=True,
    )
