# Copyright (C) 2019 Bodo Inc. All rights reserved.
import unittest
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import numba
import bodo
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    check_func,
)
import pytest


# TODO: when groupby null keys are properly ignored
#       adds np.nan in column "A"
@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": [-8, 2, 3, 1, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": pd.Series(np.full(7, np.nan), dtype="Int64"),
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2.1, -1.5, 0.0, -1.5, 2.1, 2.1, 1.5],
                "B": [-8.3, np.nan, 3.8, 1.3, 5.4, np.nan, -7.0],
                "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3, -3.7],
            }
        ),
    ]
)
def test_df(request):
    return request.param


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": [-8, 2, 3, 1, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": [-8, np.nan, 3, np.nan, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2.1, -1.5, 0.0, -1.5, 2.1, 2.1, 1.5],
                "B": [-8.3, np.nan, 3.8, 1.3, 5.4, np.nan, -7.0],
                "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3, -3.7],
            }
        ),
    ]
)
def test_df_int_no_null(request):
    """
    Testing data for functions that does not support nullable integer columns
    with nulls only 

    Ideally, all testing function using test_df_int_no_null as inputs
    should support passing tests with test_df
    """
    return request.param


def test_agg():
    """
    Test Groupby.agg(): one user defined func and all cols
    """

    def impl(df):
        A = df.groupby("A").agg(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl, (df,), sort_output=True)


def test_agg_as_index():
    """
    Test Groupby.agg() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        A = df.groupby("A", as_index=False).agg(lambda x: x.max() - x.min())
        return A

    def impl2(df):
        A = df.groupby("A", as_index=False)["B"].agg(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl1, (df,), sort_output=True)
    check_func(impl2, (df,), sort_output=True, check_dtype=False)


def test_agg_select_col():
    """
    Test Groupby.agg() with explicitely select one column
    """

    def impl_num(df):
        A = df.groupby("A")["B"].agg(lambda x: x.max() - x.min())
        return A

    def impl_str(df):
        A = df.groupby("A")["B"].agg(lambda x: (x == "a").sum())
        return A

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].agg(lambda x: x.max() - x.min())
        return A

    df_int = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    df_float = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2]}
    )
    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )
    check_func(impl_num, (df_int,), sort_output=True)
    check_func(impl_num, (df_float,), sort_output=True)
    check_func(impl_str, (df_str,), sort_output=True)
    check_func(test_impl, (11,), sort_output=True)


def test_agg_multi_udf():
    """
    Test Groupby.agg() multiple user defined functions
    """

    def impl(df):
        def id1(x):
            return (x <= 2).sum()

        def id2(x):
            return (x > 2).sum()

        return df.groupby("A")["B"].agg((id1, id2))

    df = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    check_func(impl, (df,), sort_output=True)


def test_aggregate():
    """
    Test Groupby.aggregate(): one user defined func and all cols
    """

    def impl(df):
        A = df.groupby("A").aggregate(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl, (df,), sort_output=True)


def test_aggregate_as_index():
    """
    Test Groupby.aggregate() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        A = df.groupby("A", as_index=False).aggregate(lambda x: x.max() - x.min())
        return A

    def impl2(df):
        A = df.groupby("A", as_index=False)["C"].aggregate(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl1, (df,), sort_output=True)
    check_func(impl2, (df,), sort_output=True, check_dtype=False)


def test_aggregate_select_col():
    """
    Test Groupby.aggregate() with explicitely select one column
    """

    def impl_num(df):
        A = df.groupby("A")["B"].aggregate(lambda x: x.max() - x.min())
        return A

    def impl_str(df):
        A = df.groupby("A")["B"].aggregate(lambda x: (x == "a").sum())
        return A

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].aggregate(lambda x: x.max() - x.min())
        return A

    df_int = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    df_float = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2]}
    )
    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )
    check_func(impl_num, (df_int,), sort_output=True)
    check_func(impl_num, (df_float,), sort_output=True)
    check_func(impl_str, (df_str,), sort_output=True)
    check_func(test_impl, (11,), sort_output=True)


def test_count():
    """
    Test Groupby.count()
    TODO: after groupby.count() properly ignores nulls, adds np.nan to df_str
    """

    def impl1(df):
        A = df.groupby("A").count()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").count()
        return A

    df_int = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7],
            "C": [1.1, 2.4, 3.1, -1.9, 2.3, 3.0, -2.4],
        }
    )
    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", "a", "bb", "aa", "dd", "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df_int,), sort_output=True)
    check_func(impl1, (df_str,), sort_output=True)
    check_func(impl1, (df_dt,), sort_output=True)
    check_func(impl2, (11,), sort_output=True)


def test_count_select_col():
    """
    Test Groupby.count() with explicitely select one column
    TODO: after groupby.count() properly ignores nulls, adds np.nan to df_str
    """

    def impl1(df):
        A = df.groupby("A")["B"].count()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].count()
        return A

    df_int = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7],
            "C": [1.1, 2.4, 3.1, -1.9, 2.3, 3.0, -2.4],
        }
    )
    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", "a", "bb", "aa", "dd", "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df_int,), sort_output=True)
    check_func(impl1, (df_str,), sort_output=True)
    check_func(impl1, (df_dt,), sort_output=True)
    check_func(impl2, (11,), sort_output=True)


def test_filtered_count():
    """
    Test Groupby.count() with filtered dataframe
    """

    def test_impl(df, cond):
        df2 = df[cond]
        c = df2.groupby("A").count()
        return df2, c

    bodo_func = bodo.jit(test_impl)
    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, np.nan, 5, 6, 7],
            "C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    cond = df.A > 1
    res = test_impl(df, cond)
    h_res = bodo_func(df, cond)
    pd.testing.assert_frame_equal(res[0], h_res[0])
    pd.testing.assert_frame_equal(res[1], h_res[1])


def test_as_index_count():
    """
    Test Groupby.count() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).count()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["C"].count()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, np.nan, 5, 6, 7],
            "C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    check_func(impl1, (df,), sort_output=True)
    check_func(impl2, (df,), sort_output=True)


def test_groupby_cumsum():
    """
    Test Groupby.cumsum()
    """

    def impl(df):
        df2 = df.groupby("A").cumsum()
        return df2

    df1 = pd.DataFrame(
        {
            "A": [0, 1, 3, 2, 1, 0, 4, 0, 2, 0],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7, 3, 1, 2],
            "C": [-8, 2, 3, 1, 5, 6, 7, 3, 1, 2],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "B": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    df3 = pd.DataFrame(
        {
            "A": [0.3, np.nan, 3.5, 0.2, np.nan, 3.3, 0.2, 0.3, 0.2, 0.2],
            "B": [-1.1, 1.1, 3.2, 1.1, 5.2, 6.8, 7.3, 3.4, 1.2, 2.4],
            "C": [-8.1, 2.3, 5.3, 1.1, 0.5, 4.6, 1.7, 4.3, -8.1, 5.3],
        }
    )
    check_func(impl, (df1,), sort_output=True)
    check_func(impl, (df2,), sort_output=True)
    check_func(impl, (df3,), sort_output=True)


def test_groupby_multi_intlabels_cumsum_int():
    """
    Test Groupby.cumsum() on int columns
    multiple labels for 'by'
    """

    def impl(df):
        df2 = df.groupby(["A", "B"])["C"].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 1, -8, 1, 5, 1, 7],
            "C": [3, np.nan, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,))


def test_groupby_multi_labels_cumsum_multi_cols():
    """
    Test Groupby.cumsum() 
    multiple labels for 'by', multiple cols to cumsum
    """

    def impl(df):
        df2 = df.groupby(["A", "B"])["C", "D"].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, 5, 6, 5, 4, 4, 3],
            "D": [3.1, 1.1, 6.0, np.nan, 4.0, np.nan, 3],
        }
    )
    check_func(impl, (df,))


def test_groupby_as_index_cumsum():
    """
    Test Groupby.cumsum() on groupby() as_index=False
    for both dataframe and series returns
    TODO: adds np.nan to "A" after groupby null keys are properly ignored
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).cumsum()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["C"].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [3.0, 1.0, 4.1, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, np.nan, 6, 5, 4, 4, 3],
            "D": [3.1, 1.1, 6.0, np.nan, 4.0, np.nan, 3],
        }
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))


def test_cumsum_all_nulls_col():
    """
    Test Groupby.cumsum() on column with all null entrieS
    TODO: change by to "A" after groupby null keys are properly ignored
    """

    def impl(df):
        df2 = df.groupby("B").cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, 5, 6, 5, 4, 4, 3],
            "D": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    check_func(impl, (df,))


def test_max(test_df_int_no_null):
    """
    Test Groupby.max()
    """

    def impl1(df):
        A = df.groupby("A").max()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").max()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True)
    check_func(impl2, (11,))


def test_max_one_col(test_df_int_no_null):
    """
    Test Groupby.max() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].max()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].max()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True)
    check_func(impl2, (11,))


def test_groupby_as_index_max():
    """
    Test max on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).max()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].max()
        return df2

    check_func(impl1, (11,))
    check_func(impl2, (11,))


def test_mean(test_df_int_no_null):
    """
    Test Groupby.mean()
    """

    def impl1(df):
        A = df.groupby("A").mean()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").mean()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_mean_one_col(test_df_int_no_null):
    """
    Test Groupby.mean() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].mean()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].mean()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_mean():
    """
    Test mean on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).mean()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].mean()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_min(test_df_int_no_null):
    """
    Test Groupby.min()
    """

    def impl1(df):
        A = df.groupby("A").min()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").min()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_min_one_col(test_df_int_no_null):
    """
    Test Groupby.min() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].min()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].min()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_min():
    """
    Test min on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).min()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].min()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_min_datetime():
    """
    Test Groupby.min() on datetime column
    for both dataframe and series returns
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).min()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["B"].min()
        return df2

    df = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df,), sort_output=True)
    check_func(impl2, (df,), sort_output=True)


def test_prod(test_df):
    """
    Test Groupby.prod()
    """

    def impl1(df):
        A = df.groupby("A").prod()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").prod()
        return A

    check_func(impl1, (test_df,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_prod_one_col(test_df):
    """
    Test Groupby.prod() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].prod()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].prod()
        return A

    check_func(impl1, (test_df,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_prod():
    """
    Test prod on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).prod()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].prod()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_std(test_df_int_no_null):
    """
    Test Groupby.std()
    """

    def impl1(df):
        A = df.groupby("A").std()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").std()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_std_one_col(test_df_int_no_null):
    """
    Test Groupby.std() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].std()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].std()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_std():
    """
    Test std on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).std()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].std()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_sum(test_df_int_no_null):
    """
    Test Groupby.sum()
    """

    def impl1(df):
        A = df.groupby("A").sum()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").sum()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_sum_one_col(test_df_int_no_null):
    """
    Test Groupby.sum() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].sum()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].sum()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_sum():
    """
    Test sum on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).sum()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].sum()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_multi_intlabels_sum():
    """
    Test df.groupby() multiple labels of string columns
    and Groupy.sum() on integer column
    """

    def impl(df):
        A = df.groupby(["A", "C"])["B"].sum()
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True)


def test_groupby_multi_strlabels():
    """
    Test df.groupby() multiple labels of string columns
    with as_index=False, and Groupy.sum() on integer column
    """

    def impl(df):
        df2 = df.groupby(["A", "B"], as_index=False)["C"].sum()
        return df2

    df = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", "a", "a", "aa", "ccc", "ggg", "a"],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True)


def test_groupby_multiselect_sum():
    """
    Test groupy.sum() on explicitely selected columns
    """

    def impl(df):
        df2 = df.groupby("A")["B", "C"].sum()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True)


def test_groupby_multiselect_list():
    """
    Test groupy on explicitely selected columns using a constant list (#198)
    """

    def impl(df):
        df2 = df.groupby("A")[["B", "C"]].sum()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True)


def test_agg_multikey_parallel():
    """
    Test groupby multikey with distributed df
    """

    def test_impl(df):
        A = df.groupby(["A", "C"])["B"].sum()
        return A.sum()

    bodo_func = bodo.jit(distributed=["df"])(test_impl)
    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    start, end = get_start_end(len(df))
    h_res = bodo_func(df.iloc[start:end])
    p_res = test_impl(df)
    assert h_res == p_res


def test_var(test_df_int_no_null):
    """
    Test Groupby.var()
    """

    def impl1(df):
        A = df.groupby("A").var()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").var()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_var_one_col(test_df_int_no_null):
    """
    Test Groupby.var() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].var()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].var()
        return A

    check_func(impl1, (test_df_int_no_null,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


def test_groupby_as_index_var():
    """
    Test var on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).var()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].var()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, check_dtype=False)


# ------------------------------ pivot, crosstab ------------------------------ #


_pivot_df1 = pd.DataFrame(
    {
        "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
        "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
        "C": [
            "small",
            "large",
            "large",
            "small",
            "small",
            "large",
            "small",
            "small",
            "large",
        ],
        "D": [1, 2, 2, 6, 3, 4, 5, 6, 9],
    }
)


def test_pivot():
    def test_impl(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        return (pt.small.values, pt.large.values)

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(test_impl)
    assert set(bodo_func(_pivot_df1)[0]) == set(test_impl(_pivot_df1)[0])
    assert set(bodo_func(_pivot_df1)[1]) == set(test_impl(_pivot_df1)[1])


def test_pivot_parallel(datapath):
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        res = pt.small.values.sum()
        return res

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(impl)
    assert bodo_func() == impl()


def test_crosstab():
    def test_impl(df):
        pt = pd.crosstab(df.A, df.C)
        return (pt.small.values, pt.large.values)

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(test_impl)
    assert set(bodo_func(_pivot_df1)[0]) == set(test_impl(_pivot_df1)[0])
    assert set(bodo_func(_pivot_df1)[1]) == set(test_impl(_pivot_df1)[1])


def test_crosstab_parallel(datapath):
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = pd.crosstab(df.A, df.C)
        res = pt.small.values.sum()
        return res

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(impl)
    assert bodo_func() == impl()
