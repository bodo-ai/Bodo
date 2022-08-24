# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests that various optimizations inside of parfors work properly
with Bodo data types. These tests should be used to check that specific
compiler optimizations (i.e. dce) are working properly.
"""

import numba
import numpy as np
import pandas as pd
from numba import prange

import bodo
from bodo.tests.utils import ParforTestPipeline, check_func


def test_setna_parfor_dce(memory_leak_check):
    """
    Check that when setna is used inside a parfor
    that should be unused that the parfor can be properly
    removed.
    """

    def test_impl(df):
        # This filter should be removed because the result is
        # unused.
        df["B"] > 4
        return df["C"]

    df = pd.DataFrame(
        {
            "A": [1, 2, 4, None] * 3,
            "B": [1, 2, 4, None] * 3,
            "C": [1, 2, 4, None] * 3,
        },
        # Nullable array will make the condition output a nullable
        # boolean array, which contains setna.
        dtype="Int64",
    )

    check_func(test_impl, (df,))
    # Check that there is no parfor in the code.

    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl)
    bodo_func(df)
    _check_num_parfors(bodo_func, 0)


def test_parfor_and_or_dce(memory_leak_check):
    """
    Check that when and/or with a nullable boolean array
    creates a parfor that should be unused, then the parfor
    can be properly removed.
    """

    def test_impl_and(df):
        # This filter should be removed because the result is
        # unused.
        (df["B"] > 1) & (df["B"] < 4)
        return df["C"]

    def test_impl_or(df):
        # This filter should be removed because the result is
        # unused.
        (df["B"] < 2) | (df["B"] > 3)
        return df["C"]

    df = pd.DataFrame(
        {
            "A": [1, 2, 4, None] * 3,
            "B": [1, 2, 4, None] * 3,
            "C": [1, 2, 4, None] * 3,
        },
        # Nullable array will make the condition output a nullable
        # boolean array, which contains setna.
        dtype="Int64",
    )

    check_func(test_impl_and, (df,))
    # Check that there is no parfor in the code.
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl_and)
    bodo_func(df)
    _check_num_parfors(bodo_func, 0)

    check_func(test_impl_or, (df,))
    # Check that there is no parfor in the code.
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl_or)
    bodo_func(df)
    _check_num_parfors(bodo_func, 0)


def test_parfor_str_eq_dce(memory_leak_check):
    """
    Check that when a string equality operator is used
    in a dead parfor that it can be properly eliminated.
    """

    def test_impl(df):
        # This filter should be removed because the result is
        # unused.
        df["B"] == "af3"
        return df["C"]

    df = pd.DataFrame(
        {
            "A": [1, 2, 4, None] * 3,
            "B": ["232", "af3", "r32", None] * 3,
            "C": [1, 2, 4, None] * 3,
        },
    )

    check_func(test_impl, (df,))
    # Check that there is no parfor in the code.
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl)
    bodo_func(df)
    _check_num_parfors(bodo_func, 0)


def test_tuple_parfor_fusion():
    """
    Check to make sure numba can fuse a loop when `ShapeEquivSet._getnames()
    attempts to analyze an unsupported type (float tuple).
    """

    def test_impl(arr):
        a = (1.2, 1.3)
        n1 = len(arr)
        arr2 = np.empty(n1, np.float64)
        for i in prange(n1):
            arr2[i] = arr[i] * a[0]
        n2 = len(arr2)
        arr3 = np.empty(n2, np.float64)
        for j in prange(n2):
            arr3[j] = arr2[j] - a[1]
        total = 0.0
        n3 = len(arr3)
        for k in prange(n3):
            total += arr3[k]
        return total + a[0]

    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl)
    bodo_func(np.arange(100))
    _check_num_parfors(bodo_func, 1)


def _check_num_parfors(bodo_func, expected):
    """
    Ensure that the bodo function contains a specific number of parfors.
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    num_parfors = 0
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                num_parfors += 1
    assert num_parfors == expected, f"Expected {expected}, found {num_parfors} parfors"
