"""
Test that Named Parameters are cached properly in both
sequential and parallel code.
"""
# Copyright (C) 2021 Bodo Inc. All rights reserved.


import numpy as np
import pytest
from bodosql.tests.caching_utils import check_caching
from bodosql.tests.utils import InputDist

import bodosql


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_dist",
    [
        InputDist.REP,
        # Only checking REP until we fix the Rep/Dist caching issue
        # InputDist.OneD,
        # InputDist.OneDVar,
    ],
)
def test_cache_int_add(basic_df, input_dist, is_cached):
    """
    Test that caching works with named parameters
    for integer addition.
    """

    def test_impl(df, a):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select A + @a from table1", {"a": a})

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defualts to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    if check_cache:
        args = (basic_df["table1"], np.int64(46))
    else:
        args = (basic_df["table1"], np.int64(3))

    check_caching(
        test_impl,
        args,
        check_cache,
        input_dist=input_dist,
    )
