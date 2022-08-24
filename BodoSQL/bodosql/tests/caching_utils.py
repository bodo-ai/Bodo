"""
Infrastructure used to test caching.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import pandas as pd
from bodosql.tests.utils import InputDist, _check_query_equal, _get_dist_df

import bodo


def check_caching(
    impl,
    args,
    is_cached,
    input_dist=InputDist.REP,
    check_names=False,
    check_dtype=False,
    sort_output=False,
):
    """Test caching by compiling a BodoSQL function with
    cache=True, then running it again loading from cache.

    This function does not test correctness, so its assumed these
    functions are tested elsewhere.

    impl: the function to compile
    args: arguments to pass to the function
    is_cached: true if we expect the function to already be cached, false if we do not.
    input_dist: The InputDist for the dataframe argumennts. This is used
        in the flags for compiling the function.
    """
    # compile impl in the correct dist
    if input_dist == InputDist.OneD:
        args = tuple(
            [_get_dist_df(x) if isinstance(x, pd.DataFrame) else x for x in args]
        )

    elif input_dist == InputDist.OneDVar:
        args = tuple(
            [
                _get_dist_df(x, var_length=True) if isinstance(x, pd.DataFrame) else x
                for x in args
            ]
        )

    all_args_distributed_block = input_dist == InputDist.OneD
    all_args_distributed_varlength = input_dist == InputDist.OneDVar
    all_returns_distributed = input_dist != InputDist.REP
    returns_maybe_distributed = input_dist != InputDist.REP
    args_maybe_distributed = input_dist != InputDist.REP
    bodo_func = bodo.jit(
        cache=True,
        all_args_distributed_block=all_args_distributed_block,
        all_args_distributed_varlength=all_args_distributed_varlength,
        all_returns_distributed=all_returns_distributed,
        returns_maybe_distributed=returns_maybe_distributed,
        args_maybe_distributed=args_maybe_distributed,
    )(impl)

    bodo_out = bodo_func(*args)
    py_out = impl(*args)
    output_dist = returns_maybe_distributed

    _check_query_equal(
        bodo_out,
        py_out,
        check_names,
        check_dtype,
        sort_output,
        output_dist,
        "Correctness check failed",
        None,
    )

    bodo.barrier()

    # get signature of compiled function
    sig = bodo_func.signatures[0]

    if is_cached:
        # assert that it was loaded from cache
        assert bodo_func._cache_hits[sig] == 1
        assert bodo_func._cache_misses[sig] == 0
    else:
        # assert that it wasn't loaded from cache
        assert bodo_func._cache_hits[sig] == 0
        assert bodo_func._cache_misses[sig] == 1

    return bodo_out
