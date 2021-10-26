# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Tests that various optimizations inside of parfors work properly
with Bodo data types. These tests should be used to check that specific
compiler optimizations (i.e. dce) are working properly.
"""

import numba
import pandas as pd

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
    _check_no_parfors(bodo_func)


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
    _check_no_parfors(bodo_func)

    check_func(test_impl_or, (df,))
    # Check that there is no parfor in the code.
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(test_impl_or)
    bodo_func(df)
    _check_no_parfors(bodo_func)


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
    _check_no_parfors(bodo_func)


def _check_no_parfors(bodo_func):
    """
    Ensure that the bodo function does not contain a parfor.
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    for block in fir.blocks.values():
        for stmt in block.body:
            assert not isinstance(
                stmt, numba.parfors.parfor.Parfor
            ), "Encountered an unexpected parfor"
