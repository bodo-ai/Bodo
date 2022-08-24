# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
PYTEST_DONT_REWRITE
"""

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


# using separate file for assert tests to avoid pytest rewrite errors
@pytest.mark.slow
def test_assert_inline():
    # make sure assert in an inlined function works
    def g(a):
        assert a == 0

    bodo_g = bodo.jit(g)

    def f():
        bodo_g(0)

    # TODO: check for static raise presence in IR
    bodo_f = bodo.jit(f)
    bodo_f()


@pytest.mark.slow
def test_df_apply_assertion(memory_leak_check):
    """test assertion in UDF passed to df.apply().
    Bodo shouldn't inline the UDF since Numba's prange doesn't support assertions
    (multiple loop exit points in general).
    TODO: support assertions in parallel loops in general.
    """

    def test_impl(df):
        def udf(r):
            if r.A == 2:
                return 3
            assert r.A == 1
            return 5

        return df.apply(udf, axis=1)

    n = 11
    df = pd.DataFrame({"A": [1, 1, 2] * n})
    check_func(test_impl, (df,))
