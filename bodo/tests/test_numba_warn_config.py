# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test reloading numba config from environment variables
"""

import warnings

import numba
import pandas as pd

import bodo


def test_numba_warn_config():
    """
    Tests that the Numba config persists even if Numba reloads.

    numba.core.config._env_reloader.reset() reloads the config
    by reading the environment variables, otherwise setting them
    to Numba's default values

    """

    @bodo.jit
    def impl():
        a = pd.read_parquet("bodo/tests/data/example.parquet")
        return len(a)

    numba.config._env_reloader.reset()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        impl()
        for x in w:
            if x.category == numba.errors.NumbaPerformanceWarning:
                AssertionError
