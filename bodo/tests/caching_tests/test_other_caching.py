# Copyright (C) 2019 Bodo Inc. All rights reserved.
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
from bodo.tests.utils import InputDist, check_caching


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
