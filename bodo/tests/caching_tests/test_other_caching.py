# Copyright (C) 2022 Bodo Inc. All rights reserved.
import logging

import pandas as pd
import pytest
from caching_tests_common import fn_distribution  # noqa

from bodo.tests.test_metadata import (  # noqa
    bytes_gen_dist_df,
    int_gen_dist_df,
    str_gen_dist_df,
    struct_gen_dist_df,
)
from bodo.tests.utils import InputDist, check_caching, reduce_sum


def test_groupby_agg_caching(fn_distribution, is_cached, memory_leak_check):
    """Test compiling function that uses groupby.agg(udf) with cache=True
    and loading from cache"""

    def impl(df):
        A = df.groupby("A").agg(lambda x: x.max() - x.min())
        return A

    def impl2(df):
        def g(X):
            z = X.iloc[0] + X.iloc[2]
            return X.iloc[0] + z

        A = df.groupby("A").agg(g)
        return A

    df = pd.DataFrame({"A": [0, 0, 1, 1, 1, 0], "B": range(6)})

    # test impl (regular UDF)
    check_caching(impl, (df,), is_cached, fn_distribution)

    # test impl2 (general UDF)
    check_caching(impl2, (df,), is_cached, fn_distribution)


def test_merge_general_cond_caching(fn_distribution, is_cached, memory_leak_check):
    """
    test merge(): with general condition expressions like "left.A == right.A"
    """

    def impl(df1, df2):
        return df1.merge(df2, on="right.D <= left.B + 1 & left.A == right.A")

    df1 = pd.DataFrame({"A": [1, 2, 1, 1, 3, 2, 3], "B": [1, 2, 3, 1, 2, 3, 1]})
    df2 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": [3, 2, 1, 3, 2, 1, 2],
            "D": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D <= B + 1")

    check_caching(
        impl,
        (df1, df2),
        is_cached,
        fn_distribution,
        py_output=py_out,
        reset_index=True,
        sort_output=True,
    )


def test_format_cache(fn_distribution, is_cached, memory_leak_check):
    """
    test caching for string formatting
    """

    def impl():
        return "{}".format(3)

    check_caching(impl, (), is_cached, fn_distribution, is_out_dist=False)


def test_lowered_rootlogger_cache(fn_distribution, is_cached, memory_leak_check):
    """
    test caching for rootlogger
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    def impl():
        logger.info("Compilation Finished")
        return 0

    check_caching(impl, (), is_cached, fn_distribution, is_out_dist=False)


@pytest.mark.parametrize(
    "gen_type_annotated_df_func",
    [
        pytest.param(int_gen_dist_df, id="int"),
        pytest.param(str_gen_dist_df, id="str"),
        pytest.param(bytes_gen_dist_df, id="bytes"),
        pytest.param(struct_gen_dist_df, id="struct"),
    ],
)
def test_metadata_cache(gen_type_annotated_df_func, is_cached, memory_leak_check):
    """Checks that, in a situation where we need type inference to determine the type of the input
    dataframe (i.e. empty array on certain ranks), we still get caching when running on other
    dataframes of the same type.

    gen_type_annotated_df_func: function that returns a dataframe that requires type annotation to infer the dtype
    """

    def impl(df):
        return df

    df_with_type_annotation = gen_type_annotated_df_func()

    check_caching(
        impl,
        (df_with_type_annotation,),
        is_cached,
        InputDist.OneD,
        args_already_distributed=True,
    )


def test_jit_func_cache(is_cached, memory_leak_check):
    """
    test caching for jitted functions
    """

    import numpy as np

    import bodo

    @bodo.jit
    def f(A):
        C = g(A)
        return C

    @bodo.jit(cache=True)
    def g(A):
        B = A + 1
        return B

    # because both f and g are jitted, and f needs to be jitted first,
    # we cannot use `check_caching` here.

    # Add a barrier to reduce the odds of possible race condition
    # between ranks getting a cached implementation.
    bodo.barrier()

    assert np.array_equal(f(np.ones(4)), np.ones(4) + 1)

    sig = f.signatures[0]

    if is_cached:
        # assert that it was loaded from cache
        assert g._cache_hits[sig] == 1
        assert g._cache_misses[sig] == 0
    else:
        # assert that it wasn't loaded from cache
        assert g._cache_hits[sig] == 0
        assert g._cache_misses[sig] == 1


def test_index_info_caching(is_cached, memory_leak_check):
    """
    test caching for index info when reading from a parquet file
    """

    import os

    import bodo

    def impl():
        df = pd.read_parquet("test.pq")
        return df

    # create different sized dataframes with the same schema
    if is_cached:
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "B": range(0, 22, 2)}
        )
    else:
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": range(0, 20, 2)})

    # write and read the dataframe to/from a parquet file
    try:
        if bodo.get_rank() == 0:
            df.to_parquet("test.pq")
        bodo.barrier()
        check_caching(impl, (), is_cached, InputDist.OneD)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("test.pq")


def test_caching_version_check(is_cached, memory_leak_check):
    """
    test caching version check:
      - assert that the bodo version string is in the cache filename and change
        the version of the cached function in first pass
      - assert that the cache is not used in the second pass because the version doesn't match
    """

    import os

    import bodo

    @bodo.jit(cache=True)
    def impl(x):
        return x + 1

    bodo.barrier()

    # first pass
    if not is_cached:
        impl(1)

        # check that bodo version is in the cache filename
        passed = 1
        try:
            # Since these are filesystem operations, we only perform them on rank 0
            # to avoid race conditions (e.g. during the rename).
            if bodo.get_rank() == 0:
                cache_file = impl._cache._cache_file._index_path
                assert bodo.__version__ in cache_file

                # change the version of the cached filename
                fake_cache_file = cache_file.replace(bodo.__version__, "0.0.0")
                os.rename(
                    os.path.join(impl._cache.cache_path, cache_file),
                    os.path.join(impl._cache.cache_path, fake_cache_file),
                )
        except Exception as e:
            passed = 0
            print(e)

        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()

        bodo.barrier()

    # second pass
    else:
        impl(1)

        # assert that it wasn't loaded from cache
        sig = impl.signatures[0]
        assert impl._cache_hits[sig] == 0
        assert impl._cache_misses[sig] == 1
