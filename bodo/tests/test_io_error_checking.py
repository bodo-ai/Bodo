# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tests I/O error checking for CSV, Parquet, HDF5, etc.
"""
# TODO: Move error checking tests from test_io to here.

import os

import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_csv_nrows_type(memory_leak_check):
    """
    Test read_csv(): 'nrows' wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, nrows=-2)

    def impl2():
        return pd.read_csv(fname, nrows="wrong")

    with pytest.raises(ValueError, match="integer >= 0"):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match="must be integer"):
        bodo.jit(impl2)()


@pytest.mark.slow
def test_csv_skiprows_type(memory_leak_check):
    """
    Test read_csv(): 'skiprows' wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, skiprows=-2)

    def impl2():
        return pd.read_csv(fname, skiprows="wrong")

    def impl3():
        return pd.read_csv(
            fname, skiprows=lambda x: x > 1, names=["X", "Y", "Z", "MM", "AA"]
        )

    with pytest.raises(ValueError, match="integer >= 0"):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match="must be integer"):
        bodo.jit(impl2)()

    with pytest.raises(BodoError, match="callable not supported yet"):
        bodo.jit(impl3)()
