# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Unittests for DataFrames
"""
import datetime
import os
import random
import sys
from decimal import Decimal

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core.utils import PYVERSION

import bodo
import bodo.tests.dataframe_common
from bodo.tests.dataframe_common import *  # noqa
from bodo.tests.utils import (
    AnalysisTestPipeline,
    DeadcodeTestPipeline,
    _get_dist_arg,
    check_func,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_array_struct_list_int,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    get_start_end,
    has_udf_call,
)
from bodo.utils.typing import BodoError, BodoWarning, ColNamesMetaType


def test_pd_isna_getitem(memory_leak_check):
    """test support for NA check for array values, e.g. pd.isna(A[i]) pattern matching
    in SeriesPass
    """

    def impl1(df):
        s = 0
        for i in bodo.prange(len(df)):
            l = 0
            if pd.isna(df.iloc[i, 0]):
                l = 10
            else:
                l = len(df.iloc[i, 0])
            s += l
        return s

    def impl2(S, i):
        return pd.notna(S.iloc[i])

    def impl3(A, i):
        return pd.isnull(A[i])

    df = pd.DataFrame(
        {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
        [1.1, -2.1, 7.1, 0.1, 3.1],
    )
    check_func(impl1, (df,))
    S = pd.Series([2.1, 5.3, np.nan, -1.0, -3.7], [3, 5, 6, -2, 4], name="C")
    assert bodo.jit(impl2)(S, 0) == impl2(S, 0)
    assert bodo.jit(impl2)(S, 2) == impl2(S, 2)
    A = np.array([1.3, 2.2, np.nan, 3.1, np.nan, -1.1])
    assert bodo.jit(impl3)(A, 0) == impl3(A, 0)
    assert bodo.jit(impl3)(A, 2) == impl3(A, 2)


def test_setitem_na(memory_leak_check):
    """test support for setting NA value to array location, e.g. A[i] = None"""

    def impl(S, i):
        S.iloc[i] = None
        return S

    S = pd.Series(["AA", np.nan, "", "D", "GG"], name="C")
    # TODO: support distributed setitem with scalar
    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(S.copy(), 0), impl(S.copy(), 0), check_dtype=False
    )
    pd.testing.assert_series_equal(
        bodo_func(S.copy(), 1), impl(S.copy(), 1), check_dtype=False
    )
    pd.testing.assert_series_equal(
        bodo_func(S.copy(), 2), impl(S.copy(), 2), check_dtype=False
    )


@pytest.mark.slow
def test_set_column_index(memory_leak_check):
    """set df column with an Index value"""

    def test_impl(df):
        df["C"] = df.index
        return df

    df = pd.DataFrame({"A": [1, 3, 4]}, index=["a", "ab", "cd"])
    check_func(test_impl, (df,), copy_input=True, only_seq=True)


@pytest.mark.slow
def test_set_column_categorical(memory_leak_check):
    """set df column with a categorical array"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = pd.Categorical(np.arange(n))
        return df

    n = 11
    check_func(test_impl, (n,))


def test_set_column_scalar_str(memory_leak_check):
    """set df column with a string scalar"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = "AA"
        return df

    # test unicode characters
    def test_impl2(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = "😀🐍,⚡ 오늘 ، هذاú,úũ"
        return df

    n = 11
    check_func(test_impl, (n,))
    check_func(test_impl2, (n,))


def test_set_column_scalar_num(memory_leak_check):
    """set df column with a numeric scalar"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = 3
        return df

    n = 11
    check_func(test_impl, (n,))


def test_set_column_scalar_timestamp(memory_leak_check):
    """set df column with a timestamp scalar"""

    def test_impl(n, t):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = t
        return df

    n = 11
    t = pd.Timestamp("1994-11-23T10:11:35")
    check_func(test_impl, (n, t))


def test_set_column_cond1(memory_leak_check):
    # df created inside function case
    def test_impl(n, cond):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df.A

    bodo_func = bodo.jit(distributed=False)(test_impl)
    n = 11
    pd.testing.assert_series_equal(bodo_func(n, True), test_impl(n, True))
    pd.testing.assert_series_equal(bodo_func(n, False), test_impl(n, False))


def test_set_column_cond2(memory_leak_check):
    # df is assigned to other variable case (mutability)
    def test_impl(n, cond):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df2 = df
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df2  # df2.A, TODO: pending set_dataframe_data() analysis fix
        # to avoid incorrect optimization

    bodo_func = bodo.jit(distributed=False)(test_impl)
    n = 11
    pd.testing.assert_frame_equal(
        bodo_func(n, True), test_impl(n, True), check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bodo_func(n, False), test_impl(n, False), check_column_type=False
    )


def test_set_column_cond3(memory_leak_check):
    # df is assigned to other variable case (mutability) and has parent
    def test_impl(df, cond):
        df2 = df
        # df2['A'] = np.arange(n) + 1.0, TODO: make set column inplace
        # when there is another reference
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df2  # df2.A, TODO: pending set_dataframe_data() analysis fix
        # to avoid incorrect optimization

    bodo_func = bodo.jit(test_impl)
    n = 11
    df1 = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
    df2 = df1.copy()
    pd.testing.assert_frame_equal(
        bodo_func(df1, True), test_impl(df2, True), check_column_type=False
    )
    pd.testing.assert_frame_equal(df1, df2, check_column_type=False)
    df1 = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
    df2 = df1.copy()
    pd.testing.assert_frame_equal(
        bodo_func(df1, False), test_impl(df2, False), check_column_type=False
    )
    pd.testing.assert_frame_equal(df1, df2, check_column_type=False)


def test_set_column_setattr(memory_leak_check):
    """set df column using setattr instead of setitem"""

    # same type as existing column
    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df.B = 2
        return df

    # change column type
    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df.B = "AA"
        return df

    n = 11
    check_func(impl1, (n,))
    check_func(impl2, (n,))


def test_set_column_reflect_error(memory_leak_check):
    """set column of dataframe argument that is passed from another JIT function, so it
    doesn't have a parent dataframe object (even though it is an argument).
    See set_df_column_with_reflect()
    """

    @bodo.jit
    def f(data):
        data["B"] = data["A"].str.len()
        return data

    def impl():
        df = pd.DataFrame({"A": ["BB", "CCB", "DDD", "A"]})
        df = f(df)
        return df

    pd.testing.assert_frame_equal(bodo.jit(impl)(), impl(), check_column_type=False)


def test_set_column_native_reflect(memory_leak_check):
    """set column of dataframe argument that is passed from another JIT function, so it
    doesn't have a parent dataframe object (even though it is an argument).
    It still needs to be updated inplace for the caller to see changes.
    See set_df_column_with_reflect()
    """

    @bodo.jit
    def f(df):
        df["A"] = 0.0

    def impl():
        df = pd.DataFrame({"A": np.ones(4)})
        f(df)
        return df

    check_func(impl, (), only_seq=True)


def test_set_multi_column_reflect(memory_leak_check):
    """make sure setting multiple columns to df arg reflects back to Python properly"""

    @bodo.jit(distributed=False)
    def f(df):
        df[["A", "B"]] = 0.0

    df = pd.DataFrame({"A": [1, 2, 3]})
    f(df)
    assert "B" in df


def test_set_column_detect_update1(memory_leak_check):
    """Make sure get_dataframe_data optimization can detect that a dataframe may be
    updated in another JIT function
    """

    @bodo.jit
    def f(df):
        df["A"] = 0.0

    def impl():
        df = pd.DataFrame({"A": np.ones(4)})
        f(df)
        return df["A"].sum()

    check_func(impl, (), only_seq=True)


def test_set_column_detect_update2(memory_leak_check):
    """Make sure get_dataframe_data optimization can detect that a dataframe may be
    updated in a UDF
    """

    @bodo.jit
    def f(r, data):
        data["A"] = 0.0
        return 1

    def impl():
        df1 = pd.DataFrame({"A": np.zeros(4)})
        df = pd.DataFrame({"A": np.ones(4)})
        df1.apply(f, axis=1, data=df)
        return df["A"].sum()

    check_func(impl, (), only_seq=True)


def test_set_column_detect_update3(memory_leak_check):
    """Make sure get_dataframe_data optimization can detect that a dataframe may be
    updated in nested UDFs
    """

    def impl(df):
        def f(r1):
            copy_df = pd.DataFrame({"B": np.ones(4)})

            def h(r2, copy_df):
                copy_df["B"] = 5.0
                return 4.0

            temp_df = pd.DataFrame({"A": np.ones(4)})
            temp_df.apply(h, axis=1, args=(copy_df,))

            return copy_df["B"].sum()

        output = df.apply(
            f,
            axis=1,
        )
        return output

    df = pd.DataFrame({"C": [2.1, 1.2, 4.2, 232.1]})
    check_func(impl, (df,), only_seq=True)


def test_set_column_detect_update_err1(memory_leak_check):
    """Make sure invalid dataframe column set is detected properly when new column
    is being set in UDF.
    """

    @bodo.jit
    def f(r, data, _bodo_inline=True):
        data["B"] = 1
        return 1

    def impl():
        df1 = pd.DataFrame({"A": np.zeros(4)})
        df = pd.DataFrame({"A": np.ones(4)})
        df1.apply(f, axis=1, data=df, _bodo_inline=True)
        return df["A"].sum()

    with pytest.raises(
        BodoError, match="Setting new dataframe columns inplace is not supported"
    ):
        bodo.jit(impl)()


def test_set_column_detect_update_err2(memory_leak_check):
    """Make sure invalid dataframe column set is detected properly when column data type
    changes in UDF.
    """

    @bodo.jit
    def f(r, data, _bodo_inline=True):
        data["A"] = 1
        return 1

    def impl():
        df1 = pd.DataFrame({"A": np.zeros(4)})
        df = pd.DataFrame({"A": np.ones(4)})
        df1.apply(f, axis=1, data=df, _bodo_inline=True)
        return df["A"].sum()

    with pytest.raises(
        BodoError,
        match="Changing dataframe column data type inplace is not supported in conditionals/loops",
    ):
        bodo.jit(impl)()


def test_set_column_table_format(memory_leak_check):
    """test setting a column of a dataframe with table format, including corner cases.
    See set_df_column_with_reflect()
    """

    def impl(df, c, val):
        df[c] = val
        return df

    df = pd.DataFrame(
        {"A": [1, 2, 3], "B": [3, 4, 5], "C": [5, 6, 7], "D": [1.1, 2.2, 3.3]}
    )
    # add a bunch of columns to trigger table format
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df[f"F{i}"] = 11

    # existing column, same type
    check_func(impl, (df, "B", 12), only_seq=True)
    # existing column, new type
    check_func(impl, (df, "B", "abc"), only_seq=True)
    # existing column, existing type, type change
    check_func(impl, (df, "B", 1.3), only_seq=True)
    # existing column, new type, previous type eliminated
    check_func(impl, (df, "D", "abc"), only_seq=True)
    # new column, existing type, same type as before
    check_func(impl, (df, "E", 3), only_seq=True)
    # new column, new type
    check_func(impl, (df, "E", "abc"), only_seq=True)


def test_set_table_data_replace(memory_leak_check):
    """Test for adding an array to a table block after a previous array was removed
    from it (BE-1635).
    """

    def impl():
        df = pd.read_parquet("demand.pq")
        df["tier"] = 3
        df["grouping"] = "abc"
        return df

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame(
                {
                    "consumer": ["ABC"] * 10,
                    "locked_price": [3] * 10,
                    "tier": ["ACD"] * 10,
                }
            )
            df.to_parquet("demand.pq")
        bodo.barrier()

        check_func(impl, (), only_seq=True)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("demand.pq")
        bodo.barrier()


@pytest.mark.skip(
    reason="[BE-240] detect changing input dataframe column data type in nested JIT calls"
)
def test_set_column_detect_update_err3(memory_leak_check):
    """Make sure invalid dataframe column set is detected properly when column data type
    changes in nested Bodo JIT calls.
    """

    @bodo.jit
    def f(data):
        data["A"] = 3

    def impl():
        df = pd.DataFrame({"A": np.ones(4)})
        f(df)
        return df["A"].sum()

    with pytest.raises(
        BodoError,
        match="Changing dataframe column data type inplace is not supported",
    ):
        bodo.jit(impl)()


def test_df_filter(memory_leak_check):
    def test_impl(df, cond):
        df2 = df[cond]
        return df2

    def test_impl2(df, cond):
        # using .values to test nullable boolean array
        df2 = df[cond.values]
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1] * 2,
            "B": ["A", "B", np.nan, "ACDE", "C", np.nan, "AA"] * 2,
            "C": [2, 3, -1, 1, np.nan, 3.1, -1] * 2,
        }
    )
    cond = df.A > 1
    check_func(test_impl, (df, cond))
    check_func(test_impl2, (df, cond))


def test_df_filter_table_format(memory_leak_check):
    """test filtering a dataframe with table format"""

    def impl(df, cond):
        df2 = df[cond]
        return df2

    # test column uses with control flow
    def impl2(df, cond, flag, flag2):
        if flag:
            df2 = df[cond]
            df2 = df2[["A", "C", "D"]]
            df2["C"] = 11
        else:
            df2 = df[cond][["A", "C", "D"]]

        return df2

    # test column uses with control flow
    def impl3(df, cond, flag):
        df2 = df[cond]
        if flag:
            s = df2["A"].sum()
        else:
            s = df2["C"].sum()
        return s

    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 4,
            "B": ["abc", "bc", "cd"] * 4,
            "C": [5, 6, 7] * 4,
            "D": [1.1, 2.2, 3.3] * 4,
        }
    )
    # add a bunch of columns to trigger table format
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df[f"F{i}"] = 11

    cond = df.A > 1
    check_func(impl, (df, cond))
    # TODO(ehsan): make sure dead columns are eliminated in filter
    check_func(impl2, (df, cond, True, True))
    check_func(impl3, (df, cond, False))


def test_create_series_input1(memory_leak_check):
    def test_impl(S):
        df = pd.DataFrame({"A": S})
        return df

    bodo_func = bodo.jit(test_impl)
    S = pd.Series([2, 4], [3, -1])
    pd.testing.assert_frame_equal(bodo_func(S), test_impl(S), check_column_type=False)


def test_df_apply_getitem(memory_leak_check):
    """test getitem access of row value passed in df.apply()"""

    def test_impl(df):
        return df.apply(lambda r: r["B"] if r["A"] == "AA" else 0, axis=1)

    df = pd.DataFrame(
        {"A": ["AA", "B", "CC", "C", "AA"], "B": [3, 1, 2, 5, 9]}, index=[3, 1, 4, 6, 0]
    )
    check_func(test_impl, (df,))


def test_df_apply_name_heterogeneous(memory_leak_check):
    """
    Check that you can get name information from DataFrame.apply with
    a heterogeneous series.
    """

    def test_impl(df):
        return df.apply(lambda x: x.name, axis=1)

    df = pd.DataFrame({"C": ["go", "to", "bed", "a", "b"], "A": [1, 2, 3, 4, 1]})
    check_func(test_impl, (df,))


def test_df_apply_name_homogeneous(memory_leak_check):
    """
    Check that you can get name information from DataFrame.apply with
    a homogeneous series.
    """

    def test_impl(df):
        return df.apply(lambda x: x.name, axis=1)

    df = pd.DataFrame({"A": [1, 2, 3, 4, 1]})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_name_datetime_index(memory_leak_check):
    """
    Check that you can get name information from DataFrame.apply with
    a homogeneous series. It checks for the correct type by using an
    attribute that only a pd.Timestamp value has.
    """

    def test_impl(df):
        return df.apply(lambda x: x.name.value, axis=1)

    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 1]}, index=pd.date_range("2018-01-01", periods=5, freq="H")
    )
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_name_timedelta_index(memory_leak_check):
    """
    Check that you can get name information from DataFrame.apply with
    a homogeneous series. It checks for the correct type by using an
    attribute that only a pd.Timedelta value has.
    """

    def test_impl(df):
        return df.apply(lambda x: x.name.value, axis=1)

    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 1]}, index=pd.timedelta_range(start="1 day", periods=5)
    )
    check_func(test_impl, (df,))


def test_df_apply_int_getitem_unsorted_columns(memory_leak_check):
    """
    test int getitem access of row passed in df.apply() where column names are not in
    sorted order (issue #2019)
    """

    def impl(df):
        return df.apply(lambda x: (x[0], x[2], x[1]), axis=1)

    df = pd.DataFrame(
        {"A": np.arange(10), "C": np.arange(10, 20), "B": np.arange(20, 30)}
    )
    check_func(impl, (df,))


def test_df_apply_bool(memory_leak_check):
    # check bool output of UDF for BooleanArray use
    def test_impl(df):
        return df.apply(lambda r: r.A == 2, axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df,))


def test_df_apply_str(memory_leak_check):
    """make sure string output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: r.A if r.A == "AA" else "BB", axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_list_str(memory_leak_check):
    """make sure list(str) output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: [r.A] if r.A == "AA" else ["BB", r.A], axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_array_item(memory_leak_check):
    """make sure array(item) output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: [len(r.A)] if r.A == "AA" else [3, len(r.A)], axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_date(memory_leak_check):
    """make sure datetime.date output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: r.A.date(), axis=1)

    df = pd.DataFrame(
        {"A": pd.date_range(start="2018-04-24", end="2019-04-29", periods=5)}
    )
    check_func(test_impl, (df,))


def test_df_apply_timestamp(memory_leak_check):
    """make sure Timestamp (converted to datetime64) output can be handled in apply()
    properly
    """

    def test_impl(df):
        return df.apply(lambda r: r.A + datetime.timedelta(days=1), axis=1)

    df = pd.DataFrame(
        {"A": pd.date_range(start="2018-04-24", end="2019-04-29", periods=5)}
    )
    check_func(test_impl, (df,))


def test_df_apply_general_colnames(memory_leak_check):
    """make sure all column names (e.g. not string, not identifier-compatible string) can be handled in apply() properly"""

    def impl1(df):
        return df.apply(lambda r: r["C C"], axis=1)

    def impl2(df):
        return df.apply(lambda r: r[2], axis=1)

    def impl3(df):
        return df.apply(lambda r: r.A, axis=1)

    df = pd.DataFrame(
        {
            "A": ["AA", "B", "CC", "C", "AA"],
            2: [3, 1, 4, 2, 6],
            "C C": [1.1, 2.2, 3.3, 4.4, 5.5],
        },
        index=[3, 1, 4, 6, 0],
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.slow
def test_df_apply_decimal(memory_leak_check):
    """make sure Decimal output can be handled in apply() properly"""
    # just returning input value since we don't support any Decimal creation yet
    # TODO: support Decimal(str) constructor
    # TODO: fix using freevar constants in UDFs
    def test_impl(df):
        return df.apply(lambda r: r.A, axis=1)

    df = pd.DataFrame(
        {
            "A": [
                Decimal("1.6"),
                Decimal("-0.222"),
                Decimal("1111.316"),
                Decimal("1234.00046"),
                Decimal("5.1"),
                Decimal("-11131.0056"),
                Decimal("0.0"),
            ]
        }
    )
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_args(memory_leak_check):
    """test passing extra args to apply UDF"""

    def test_impl(df, b):
        return df.apply(lambda r, a: r.A == a, axis=1, args=(b,))

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df, 3))


def test_df_apply_kws(memory_leak_check):
    """test passing extra keyword args to apply UDF"""

    # only kw args
    def impl1(df, b):
        return df.apply(lambda r, c=1, a=2: r.A == a + c, a=b, axis=1)

    # both positional and kw args
    def impl2(df, b, d):
        return df.apply(lambda r, c=1, a=2: r.A == a + c, a=b, axis=1, args=(d,))

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(impl1, (df, 3))
    check_func(impl2, (df, 3, 2))


def g(r):
    return 2 * r.A


def test_df_apply_func_case1(memory_leak_check):
    """make sure a global function can be used in df.apply"""

    def test_impl(df):
        return df.apply(g, axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df,))


@bodo.jit
def g2(r):
    # functions called from UDFs should not be distributed (would cause hanging)
    # using an array operation to make sure distributed code is not generated
    A = np.arange(r[0])
    return 2 * A.sum()


@pytest.mark.slow
def test_df_apply_func_case2(memory_leak_check):
    """make sure a UDF calling another function doesn't fail (#964)"""

    def test_impl(df):
        return df.apply(lambda x, _bodo_inline=False: g2(x), axis=1, _bodo_inline=False)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    # NOTE: not using check_func since regular Pandas calling g2 can cause hangs due to
    # barriers generated by Bodo
    res = bodo.jit(
        test_impl, all_args_distributed_block=True, all_returns_distributed=True
    )(_get_dist_arg(df, False))
    res = bodo.allgatherv(res)
    py_res = df.apply(lambda r: 2 * np.arange(r[0]).sum(), axis=1)
    pd.testing.assert_series_equal(res, py_res)


def test_df_apply_freevar(memory_leak_check):
    """Test transforming freevars into apply() arguments"""

    def impl1(df, b):
        return df.apply(lambda r: r.A == b, axis=1)

    # complex case with mix of constant and non-constant freevars, existing arguments,
    # local variables in UDF
    def impl2(df, a):
        d = 1
        e = 4
        m = 3

        def f(r, a):
            cmp = d + e + a
            m = 5  # name conflict with caller
            if a == 0:
                return False
            return r.A == cmp

        return df.apply(f, axis=1, args=(a,))

    # storing to freevar cannot be handled
    def impl3(df, a):
        b = 3

        def f(r):
            nonlocal b
            b = 4
            return r.A == a

        return df.apply(f, axis=1)

    n = 4
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(impl1, (df, 3))
    check_func(impl2, (df, 7))
    with pytest.raises(
        BodoError, match="Inner function is using non-constant variable"
    ):
        bodo.jit(impl3)(df, 3)


def test_df_apply_error_check():
    """make sure a proper error is raised when UDF is not supported (not compilable)"""

    def test_impl(df):
        # some UDF that cannot be supported, lambda calling a non-jit function
        return df.apply(lambda r: g(r), axis=1)

    df = pd.DataFrame({"A": np.arange(11)})
    with pytest.raises(
        BodoError, match="DataFrame.apply.*: user-defined function not supported"
    ):
        bodo.jit(test_impl)(df)


def test_df_apply_heterogeneous_series(memory_leak_check):
    """
    Test calling another JIT function in apply() UDF when row series is heterogeneous
    Tests [BE-1135]
    """

    @bodo.jit
    def apply_f(x):
        return x["A"] + 1

    @bodo.jit
    def test_impl(df):
        return df.apply(lambda x: apply_f(x), axis=1)

    df = pd.DataFrame({0: ["go", "to", "bed", "a", "b"], "A": [1, 2, 3, 4, 1]})
    # not using check_func since calling apply_f from non-jit fails
    pd.testing.assert_series_equal(test_impl(df), df.A + 1, check_names=False)


@pytest.mark.slow
def test_df_apply_df_output(memory_leak_check):
    """test DataFrame.apply() with dataframe output 1 column"""

    def impl1(df):
        return df.apply(lambda a: pd.Series([a[0], "AA"]), axis=1)

    def impl2(df):
        def g(a):
            # TODO: support assert in UDFs properly
            # assert a > 0.0
            if a[0] > 3:
                return pd.Series([a[0], 2 * a[0]], ["A", "B"])
            return pd.Series([a[0], 3 * a[0]], ["A", "B"])

        return df.apply(g, axis=1)

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    check_func(impl1, (df,))
    check_func(impl2, (df,))


def test_df_apply_df_output_multicolumn(memory_leak_check):
    """test DataFrame.apply() with dataframe output with multiple columns"""

    def test_impl(df):
        return df.apply(lambda a: pd.Series([a[0], a[1]]), axis=1)

    df = pd.DataFrame({"A": np.arange(20), "B": ["hi", "there"] * 10})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_df_output_multistring(memory_leak_check):
    def test_impl(df):
        def f(row):
            s1 = ""
            s2 = ""
            s3 = ""
            s4 = ""
            if row[1] == 0:
                s3 = str(row[0]) + ","
                s4 = str(row[2]) + ","
            elif row[1] == 1:
                s1 = str(row[0]) + ","
                s2 = str(row[2]) + ","
            return pd.Series([s1, s2, s3, s4], index=["s1", "s2", "s3", "s4"])

        return df.apply(f, axis=1)

    df = pd.DataFrame({"A": np.arange(40), "B": [0, 1] * 20, "C": np.arange(40)})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_udf_inline(memory_leak_check):
    """make sure UDFs with dataframe input are not inlined, but _bodo_inline=True can
    be used to force inlining.
    """

    def g(r, df, _bodo_inline=False):
        return r.A + df.A.sum()

    def impl1(df, df2):
        return df.apply(g, axis=1, df=df2)

    def impl2(df, df2):
        return df.apply(g, axis=1, df=df2, _bodo_inline=True)

    j_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl1)
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]})
    df2 = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]})
    # check correctness first
    pd.testing.assert_series_equal(j_func(df, df2), impl1(df, df2))
    fir = j_func.overloads[j_func.signatures[0]].metadata["preserved_ir"]
    assert has_udf_call(fir)

    j_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl2)
    # check correctness first
    pd.testing.assert_series_equal(j_func(df, df2), impl2(df, df2))
    fir = j_func.overloads[j_func.signatures[0]].metadata["preserved_ir"]
    assert not has_udf_call(fir)


@pytest.mark.slow
def test_df_apply_supported_types(df_value, memory_leak_check):
    """Test DataFrame.apply with all Bodo supported Types"""

    def test_impl(df):
        return df.apply(lambda r: None if pd.isna(r.A) else r.A, axis=1)

    # Increase data size to pass testing with 3 ranks.
    df = df_value.apply(lambda row: row.repeat(2), axis=0)

    # Rename 1st column to A (if not already A)
    if not "A" in df.columns:
        df.rename(columns={df.columns[0]: "A"}, inplace=True)

    check_func(test_impl, (df,), check_dtype=False)


@pytest.mark.slow
def test_df_apply_datetime(memory_leak_check):
    def test_impl(df):
        return df.apply(lambda r: r.A, axis=1)

    # timedelta
    df = pd.DataFrame({"A": pd.Series(pd.timedelta_range(start="1 day", periods=6))})
    check_func(test_impl, (df,))

    # datetime
    df = pd.DataFrame(
        {"A": pd.Series(pd.date_range(start="1/1/2018", end="1/4/2018", periods=6))}
    )
    check_func(test_impl, (df,))

    # boolean array
    df = pd.DataFrame({"A": [True, False, False, True, True, False]})
    check_func(test_impl, (df,))


def test_udf_other_module(memory_leak_check):
    """Test Bodo compiler pipeline replacement for a UDF function dependency in another
    module [BE-1315]
    """

    def impl():
        df = pd.DataFrame({"A": np.ones(4)})
        return df.apply(
            lambda r: bodo.tests.dataframe_common.udf_dep(r["A"] + 3), axis=1
        )

    check_func(impl, ())


def test_dataframe_pipe():
    """
    Test DataFrame.pipe()
    """

    def impl1(df):
        return df.pipe(lambda df: df.sum())

    # test *args, **kwargs
    def impl2(df, a, b):
        return df.pipe(lambda df, a, b: df.sum() + a + b, a, b=b)

    df = pd.DataFrame(
        {
            "A": [1, 4, 4, 11, 4, 1],
            "B": [1, 2, 3, 4, 5, 6],
        }
    )
    check_func(impl1, (df,), is_out_distributed=False)
    check_func(impl2, (df, 1, 2), is_out_distributed=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [
        # int
        pd.DataFrame(
            {"A": ["aa", "bb", "aa", "cc", "aa", "bb"], "B": [1, 2, 3, 5, 3, 1]}
        ),
        # float64
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": [1.5, 2.2, 2.2, 5.3, 3.4, 1.5],
            }
        ),
        # Categorical
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": pd.Series(pd.Categorical([1, 2, 5, None, 2, 5], ordered=True)),
            }
        ),
        # uint8
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": np.array([1, 8, 0, 0, 1, 1], dtype=np.uint8),
            }
        ),
        # float32
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": np.array([1.1, np.nan, 0.5, 1.1, -1.1, 0.5], dtype=np.float32),
            }
        ),
        # String
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": ["abc", "xyz", "cc", "o", "cc", "abc"],
            }
        ),
        # timedelta
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": pd.Series(pd.timedelta_range(start="1 day", periods=6)),
            }
        ),
        # datetime
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": pd.Series(
                    pd.date_range(start="1/1/2018", end="1/4/2018", periods=6)
                ),
            }
        ),
        # boolean array
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": [True, False, False, True, True, False],
            }
        ),
        # nullable float
        pd.DataFrame(
            {
                "A": ["aa", "bb", "aa", "cc", "aa", "bb"],
                "B": pd.Series([1.5, 2.2, 2.2, 5.3, None, 1.5], dtype="Float64"),
            }
        ),
    ],
)
# memory_leak_check doesn't work with Categorical
def test_df_groupby_supported_types(data):
    """Test DataFrame.groupby with all Bodo supported Types"""

    # Testing different type of columns for max operation
    def test_impl(df):
        return df.groupby("A").max()

    check_func(test_impl, (data,), sort_output=True, reset_index=True)

    # Testing different type of column used for grouping
    def test_impl2(df):
        return df.groupby("B").max()

    check_func(
        test_impl2, (data,), sort_output=True, reset_index=True, check_dtype=False
    )


def test_df_drop_inplace_branch(memory_leak_check):
    def test_impl(cond):
        if cond:
            df = pd.DataFrame({"A": [2, 3, 4], "B": [1, 2, 6]})
        else:
            df = pd.DataFrame({"A": [5, 6, 7], "B": [1, 0, -6]})
        df.drop("B", axis=1, inplace=True)
        return df

    check_func(test_impl, (True,), False)


# TODO: add memory_leak_check when join memory leaks are fixed
@pytest.mark.slow
def test_df_filter_rm_index():
    """
    Make sure dataframe index is removed correctly and parallelism warning is thrown
    when a dataframe is filtered after a join.
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3[df3.A > 3]

    df1 = pd.DataFrame({"A": [2, 3, 4], "B": [1, 2, 6]})
    df2 = pd.DataFrame({"A": [3, 4, 1]})
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(df1, df2)
    else:
        bodo.jit(impl)(df1, df2)


def test_concat_df_columns(memory_leak_check):
    """Test dataframe concatenation with axis=1 (add new columns)"""

    def test_impl(df, df2):
        return pd.concat([df, df2], axis=1)

    df = pd.DataFrame({"A": [1, 2, 3, 9, 11]})
    df2 = pd.DataFrame({"B": [4.0, 5.0, 4.1, 6.2, 2.1], "C": [7, 1, 3, -4, -1]})
    check_func(test_impl, (df, df2))


def test_concat_typing_transform(memory_leak_check):
    """Test list to tuple trasnform in typing pass, when other typing related changes
    are also required.
    """

    def test_impl(df, df2):
        df3 = pd.concat([df, df2], axis=1)
        df3["D"] = 3
        return df3

    df = pd.DataFrame({"A": [1, 2, 3, 9, 11]})
    df2 = pd.DataFrame({"B": [4.0, 5.0, 4.1, 6.2, 2.1], "C": [7, 1, 3, -4, -1]})
    check_func(test_impl, (df, df2))


def test_concat_int_float(memory_leak_check):
    """Test dataframe concatenation when integer and float are put together"""

    def test_impl(df, df2):
        return df.append(df2, ignore_index=True)

    def test_impl_concat(df, df2):
        return pd.concat((df, df2), ignore_index=True)

    df = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [4.0, 5.0]})
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)
    check_func(test_impl_concat, (df, df2), sort_output=True, reset_index=True)


def test_concat_nulls(memory_leak_check):
    """Test dataframe concatenation when full NA arrays need to be appended"""

    def test_impl(df, df2):
        return df.append(df2, ignore_index=True)

    def test_impl_concat(df, df2):
        return pd.concat((df, df2), ignore_index=True)

    n = 5
    df = pd.DataFrame(
        {
            "A": ["ABC", None, "AA", "B", None, "AA"],
            "D": pd.date_range(start="2017-01-12", periods=6),
        }
    )
    df2 = pd.DataFrame(
        {
            "B": np.arange(n),
            "C": np.ones(n),
            "E": pd.timedelta_range(start=3, periods=n),
        }
    )
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)
    check_func(test_impl_concat, (df, df2), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df",
    [
        # RangeIndex and numeric types
        pd.DataFrame(
            {
                "B": np.arange(11),
                "C": np.ones(11),
                "E": pd.timedelta_range(start=3, periods=11),
            },
        ),
        # variable item size data and index
        pd.DataFrame(
            {
                "A": ["ABC", None, "AA", "B", None, "AA", "CC", "G"],
            },
            index=["AA", "C", "BB", "A", "D", "L", "K", "P"],
        ),
    ],
)
def test_append_empty_df(df):
    """Test appending to an empty dataframe in a loop (common pattern)"""
    # TODO: fix casting refcount in Numba since Numba increfs value after cast

    def test_impl(df2):
        df = pd.DataFrame()
        for _ in range(3):
            df = df.append(df2)
        return df

    check_func(test_impl, (df,), sort_output=True, reset_index=True, check_dtype=False)


def test_init_dataframe_array_analysis():
    """make sure shape equivalence for init_dataframe() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        return df

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(10)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("n")


def test_get_dataframe_data_array_analysis():
    """make sure shape equivalence for get_dataframe_data() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(df):
        B = df.A.values
        return B

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.DataFrame({"A": np.ones(10), "B": np.arange(10)}))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("B#0")


def test_get_dataframe_index_array_analysis():
    """make sure shape equivalence for get_dataframe_index() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(df):
        B = df.index
        return B

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.DataFrame({"A": np.ones(10), "B": np.arange(10)}))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("B#0")


def test_df_const_set_rm_index(memory_leak_check):
    """Make sure dataframe related variables like the index are removed correctly and
    parallelism warning is thrown when a column is being set using a constant.
    Test for a bug that was keeping RangeIndex around as a 1D so warning wasn't thrown.
    """

    def impl(A):
        df = pd.DataFrame({"A": A})
        df["B"] = 1
        return df.A.values

    A = np.arange(10)
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(A)
    else:
        bodo.jit(impl)(A)


def test_df_dropna_df_value(df_value):
    def impl(df):
        return df.dropna()

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_fillna_df_value(df_value):
    def impl(df, value):
        return df.fillna(value)

    col = df_value.columns[0]
    df = df_value[[df_value.columns[0]]]
    value = df.dropna().iat[0, 0]
    check_func(impl, (df, value))


@pytest.mark.slow
def test_df_fillna_type_mismatch_failure():
    df = pd.DataFrame({"A": [1.2, np.nan, 242.1] * 5})
    value = "A"
    message = "Cannot use value type"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda df, value: df.fillna(value))(df, value)


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": pd.Series(
                    [None, None, "a", "b", None, None, "c", "d", None] * 3,
                ),
                "B": pd.Series([2, 3, -4, 5, 1] * 6, dtype=np.int16),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([2, 3, 4, 5, 1, 4] * 3, dtype=np.uint32),
                "B": pd.Series(
                    [None, True, None, False, True, False, None] * 3, dtype="boolean"
                ),
                "C": pd.Series(
                    [None, None, 2, -1, None, -3, 4, 5] * 3, dtype=pd.Int64Dtype()
                ),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([None, 1, 2, None, 3, None] * 3, dtype=pd.Int8Dtype()),
                "B": pd.Series(
                    [np.nan, 1.0, 2, np.nan, np.nan, 3.0, 4] * 3, dtype=np.float32
                ),
                "C": pd.Series(
                    [np.nan, 1.0, -2, np.nan, np.nan, 3.0, 4] * 3, dtype=np.float64
                ),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series(
                    [np.nan, np.nan, 1e16, 3e17, np.nan, 1e18, 500, np.nan] * 3,
                    dtype="datetime64[ns]",
                ),
                "B": pd.Series(
                    [np.nan, 3e17, 5e17, np.nan, np.nan, 1e18, 500, np.nan] * 3,
                    dtype="timedelta64[ns]",
                ),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([None] * 10 + ["a"], dtype="string"),
                "B": pd.Series([1] + [None] * 10, dtype="Int16"),
                "C": pd.Series([np.nan] * 5 + [1.1] + [np.nan] * 5, dtype="float32"),
            }
        ),
        pd.DataFrame({"A": pd.Series([np.nan] * 35, dtype="boolean")}),
    ],
)
def fillna_dataframe(request):
    return request.param


@pytest.mark.parametrize(
    "method",
    [
        "bfill",
        pytest.param("backfill", marks=pytest.mark.slow),
        "ffill",
        pytest.param("pad", marks=pytest.mark.slow),
    ],
)
def test_dataframe_fillna_method(fillna_dataframe, method, memory_leak_check):
    def test_impl(df, method):
        return df.fillna(method=method)

    # Set check_dtype=False because Bodo's unboxing type does not match
    # dtype="string"
    check_func(test_impl, (fillna_dataframe, method), check_dtype=False)


@pytest.mark.slow
def test_df_replace_df_value(df_value):
    def impl(df, to_replace, value):
        return df.replace(to_replace, value)

    df = df_value.dropna()
    df = df[[df.columns[0]]]
    to_replace = df.iat[0, 0]
    value = df.iat[1, 0]

    if any(isinstance(x, pd.Timestamp) for x in [to_replace, value]):
        message = "Not supported for types PandasTimestampType"
        with pytest.raises(BodoError, match=message):
            bodo.jit(impl)(df, to_replace, value)
    else:
        check_func(impl, (df, to_replace, value))


@pytest.mark.slow
def test_df_dropna_cat_unknown():
    """Test df.dropna() with a categorical column not known at compile time."""

    def impl(df):
        df["B"] = df["B"].astype("category")
        return df.dropna()

    df = pd.DataFrame(
        {
            "A": [1, 1, 1, 4, 5],
            "B": ["LB1", "LB2", "LB1", None, "LB2"],
            "C": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    check_func(impl, (df,), copy_input=True)


def test_df_dropna(memory_leak_check):
    """Test df.dropna() with various data types and arguments"""

    def impl1(df):
        return df.dropna(subset=["A", "B"])

    def impl2(df):
        return df.dropna(thresh=2)

    def impl3(df):
        return df.dropna(how="all")

    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 1.0] * 3,
            "B": [4, 5, 6, np.nan] * 3,
            "C": [np.nan, "AA", np.nan, "ABC"] * 3,
            "D": [[1, 2], None, [1], []] * 3,
        }
    )
    # TODO: fix 1D_Var RangeIndex
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


def test_df_dropna_inplace_check():
    """make sure inplace=True is not used in df.dropna()"""

    def test_impl(df):
        df.dropna(inplace=True)

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
    with pytest.raises(BodoError, match="inplace=True is not supported"):
        bodo.jit(test_impl)(df)


def test_df_drop_inplace_instability_check():
    """make sure df.drop(inplace=True) doesn't cause type instability"""

    def test_impl(a):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        if len(a) > 3:
            df.drop("B", 1, inplace=True)
        return df

    with pytest.raises(BodoError, match="inplace change of dataframe schema"):
        bodo.jit(test_impl)([2, 3])


# TODO: fix casting refcount in Numba since Numba increfs value after cast
def test_df_range_index_unify():
    """Test dataframe type unification when RangeIndex should be converted to
    IntegerIndex
    """

    def test_impl(df, df2):
        if len(df2) > 0:
            df = df2
        return df

    df = pd.DataFrame({"A": [1, -2, 3, 5, 11, 4, -1]})
    df2 = pd.DataFrame(
        {"A": [0, 4, 11, 2, -2, -3, 9]}, index=[11, -20, 31, 52, 1, 41, -11]
    )
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)


################################## indexing  #################################


@pytest.mark.smoke
def test_column_list_getitem1(memory_leak_check):
    """Test df[["A", "B"]] getitem case"""

    def test_impl(df):
        return df[["A", "C", "B"]]

    df = pd.DataFrame(
        {
            "A": [1.1, 2.3, np.nan, 1.7, 3.6] * 2,
            "A2": [3, 1, 2, 3, 5] * 2,
            "B": [True, False, None, False, True] * 2,
            "C": ["AA", "C", None, "ABC", ""] * 2,
        },
        index=[3, 1, 2, 4, 0] * 2,
    )
    check_func(test_impl, (df,))


def test_column_list_getitem_infer(memory_leak_check):
    """Test df[["A", "B"]] getitem case when column names list has to be inferred in
    partial typing.
    """

    def test_impl(df):
        return df[["A"] + ["C", "B"]]

    df = pd.DataFrame(
        {
            "A": [1.1, 2.3, np.nan, 1.7, 3.6] * 2,
            "A2": [3, 1, 2, 3, 5] * 2,
            "B": [True, False, None, False, True] * 2,
            "C": ["AA", "C", None, "ABC", ""] * 2,
        },
        index=[3, 1, 2, 4, 0] * 2,
    )
    check_func(test_impl, (df,))


def test_df_slice(memory_leak_check):
    """test slice getitem directly on dataframe object"""

    def impl(df, n):
        return df[1:n]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    bodo_func = bodo.jit(impl)
    # TODO: proper distributed support for slicing
    pd.testing.assert_frame_equal(
        bodo_func(df, n), impl(df, n), check_column_type=False
    )


def test_df_setitem_multi(memory_leak_check):
    """test df[col_names] setitem where col_names is a list of column names"""

    def impl1(df):
        df[["A", "B"]] = 1.3
        return df

    def impl2(df):
        df[["A", "B"]] = df[["D", "B"]]
        return df

    # test definition update error in BE-139
    def impl3():
        df = pd.DataFrame({"A": [1], "B": [2]})
        if len(df) > 1:
            df[["E", "F"]] = df.apply(lambda x: x)

    n = 11
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.arange(n) ** 2,
            "C": np.ones(n),
            "D": np.arange(n) + 1.0,
        }
    )
    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)
    with pytest.raises(
        BodoError,
        match=r"Dataframe.apply\(\): only axis=1 supported for user-defined functions",
    ):
        bodo.jit(impl3)()


def test_iloc_bool_arr(memory_leak_check):
    """test df.iloc[bool_arr]"""

    def test_impl(df):
        return df.iloc[(df.A > 3).values]

    def test_impl2(df):
        return df.iloc[(df.A > 3).values, [1, 2]]

    def test_impl3(df):
        return df.iloc[(df.A > 3).values, 1]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(test_impl, (df,))
    check_func(test_impl2, (df,))
    check_func(test_impl3, (df,))


def test_iloc_slice(memory_leak_check):
    def test_impl(df, n):
        return df.iloc[1:n]

    def test_impl2(df, n):
        return df.iloc[1:n, [1, 2]]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    bodo_func = bodo.jit(test_impl)
    # TODO: proper distributed support for slicing
    pd.testing.assert_frame_equal(
        bodo_func(df, n), test_impl(df, n), check_column_type=False
    )
    bodo_func = bodo.jit(test_impl2)
    pd.testing.assert_frame_equal(
        bodo_func(df, n), test_impl2(df, n), check_column_type=False
    )


def test_iloc_getitem_row(memory_leak_check):
    """test getitem of a single row with iloc"""

    def test_impl1(df):
        return df.iloc[1]

    def test_impl2(df):
        return df.iloc[1, [1, 2]]

    df1 = pd.DataFrame({"A": [1, 4, 6, 11, 4], "B": ["AB", "DD", "E", "A", "GG"]})
    check_func(test_impl1, (df1,), check_names=False, is_out_distributed=False)
    df2 = pd.DataFrame(
        {
            "A": [1, 4, 6, 11, 4],
            "B": ["AB", "DD", "E", "A", "GG"],
            "C": [1.2, 3.1, 4.4, -1.2, 3.2],
        }
    )
    check_func(test_impl2, (df2,), check_names=False, is_out_distributed=False)


@pytest.mark.slow
def test_iloc_getitem_row_alltypes(df_value, memory_leak_check):
    """test getitem of a single row with iloc"""

    def test_impl1(df):
        return df.iloc[1]

    def test_impl2(df):
        return df.iloc[1, [0, 1]]

    def test_impl3(df):
        # For dataframes with only 1 column
        return df.iloc[1, [0]]

    # getitem cannot match pandas when returning a NA value (need to check with pd.isna)
    # Remove any values to avoid issues.
    df = df_value.dropna()
    # Bodo converts float32 -> float64 as a common dtype, so avoid checking dtype.

    check_func(
        test_impl1,
        (df,),
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    if len(df.columns == 1):
        check_func(
            test_impl3,
            (df,),
            check_names=False,
            check_dtype=False,
            is_out_distributed=False,
        )
    else:
        check_func(
            test_impl2,
            (df,),
            check_names=False,
            check_dtype=False,
            is_out_distributed=False,
        )


@pytest.mark.slow
def test_iloc_getitem_value_alltypes(df_value, memory_leak_check):
    """test getitem of a single value with iloc. The value will be returned
    as a Series."""

    def test_impl(df):
        return df.iloc[1, 0]

    # getitem cannot match pandas when returning a NA value (need to check with pd.isna)
    # Remove any values to avoid issues.
    df = df_value.dropna()
    check_func(
        test_impl,
        (df,),
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_iloc_getitem_rows_list_alltypes(df_value, memory_leak_check):
    """test getitem of a list or rows with iloc."""

    def test_impl1(df, rows):
        return df.iloc[rows, [0, 1]]

    def test_impl2(df, rows):
        # For dataframes with only 1 column
        return df.iloc[rows, [0]]

    def test_impl3(df, rows):
        # For dataframes with only 1 column
        return df.iloc[rows, 0]

    np.random.seed(0)
    bool_idx = np.random.ranf(len(df_value)) < 0.5
    row_options = [
        # List of ints
        [0, 1, 2, 4],
        # Array of ints
        np.array([0, 1, 2, 4]),
        # List of booleans
        list(bool_idx),
        # Array of booleans
        bool_idx,
        # slice
        slice(1, 6),
    ]
    for rows in row_options:
        if len(df_value.columns) > 1:
            check_func(
                test_impl1,
                (df_value, rows),
                check_names=False,
                dist_test=False,
            )
        else:
            check_func(
                test_impl2,
                (df_value, rows),
                check_names=False,
                dist_test=False,
            )
        check_func(
            test_impl3,
            (df_value, rows),
            check_names=False,
            dist_test=False,
        )


@pytest.mark.slow
def test_iloc_getitem_slice_col_alltypes(df_value, memory_leak_check):
    """test getitem of a list or rows with iloc."""

    def test_impl1(df, rows):
        return df.iloc[rows, 0:2]

    # TODO [BE-472]: Handle selecting 0 columns
    # def test_impl2(df, rows):
    #     return df.iloc[rows, 1:3]

    np.random.seed(0)
    bool_idx = np.random.ranf(len(df_value)) < 0.5
    row_options = [
        2,
        # List of ints
        [0, 1, 2, 4],
        # Array of ints
        np.array([0, 1, 2, 4]),
        # List of booleans
        list(bool_idx),
        # Array of booleans
        bool_idx,
        # slice
        slice(1, 6),
    ]
    for rows in row_options:
        # Bodo converts float32 -> float64 as a common dtype, so avoid checking dtype.
        check_func(
            test_impl1,
            (df_value, rows),
            check_names=False,
            check_dtype=False,
            dist_test=False,
        )
        # TODO [BE-472]: Handle selecting 0 columns
        # check_func(
        #     test_impl2,
        #     (df_value, rows),
        #     check_names=False,
        #     check_dtype=False,
        #     dist_test=False,
        # )


def test_iloc_slice_col_ind(memory_leak_check):
    """test df.iloc[slice, col_ind]"""

    def test_impl(df):
        return df.iloc[:, 1].values

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_iloc_slice_col_slice(memory_leak_check):
    """test df.iloc[slice, slice] which selects a set of columns"""

    def test_impl1(df):
        return df.iloc[:, 1:]

    def test_impl2(df):
        return df.iloc[:, 1:3]

    def test_impl3(df):
        return df.iloc[:, :-1]

    def test_impl4(df):
        return df.iloc[1:, 1:]

    def test_impl5(df):
        return df.iloc[:, :]

    def test_impl6(df, n):
        if n > 3:
            n -= 1
        return df.iloc[:, 1:n]

    def test_impl7(df):
        return df.iloc[:, 0:3:2]

    def test_impl8(df, c):
        col_ind = df.columns.get_loc(c)
        return df.iloc[:, :col_ind]

    n = 11
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.arange(n) ** 2 + 1.0,
            "C": np.arange(n) + 2.0,
            "D": np.arange(n) + 3,
        }
    )
    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))
    check_func(test_impl3, (df,))
    check_func(test_impl4, (df,))
    check_func(test_impl5, (df,))
    # error checking for when slice is not constant
    with pytest.raises(BodoError, match=r"df.iloc\[slice1,slice2\] should be constant"):
        bodo.jit(test_impl6)(df, 3)
    check_func(test_impl7, (df,))
    check_func(test_impl8, (df, "C"))


def test_iloc_int_col_ind(memory_leak_check):
    """test df.iloc[int, col_ind]"""

    def test_impl(df):
        return df.iloc[3, 1]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_iloc_setitem(memory_leak_check):
    """test df.iloc[idx, col_ind] setitem where col_ind is a list of column indices"""

    # set column with full slice
    def impl1(df):
        df.iloc[:, 1] = 1.3
        return df

    # set values with bool index
    def impl2(df):
        df.iloc[df.A > 4, [1, 2]] = 11
        return df

    def impl3(df):
        df.iloc[df.A > 4, (1, 2)] = 11
        return df

    def impl4(df):
        df.iloc[0] = pd.Series([4, -1, 3])
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)
    check_func(impl3, (df,), copy_input=True)
    # TODO: Support impl4
    # check_func(impl4, (df,), copy_input=True)


def test_loc_bool_arr(memory_leak_check):
    """test df.loc[bool_arr]"""

    def test_impl(df):
        return df.loc[(df.A > 3).values]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_loc_col_name(memory_leak_check):
    """test df.loc[slice, col_ind]"""

    def test_impl(df):
        return df.loc[(df.A > 3).values, "B"].values

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_loc_range_index(memory_leak_check):
    """test df.loc[int, col_ind] for RangeIndex"""

    def test_impl(df, i):
        return df.loc[i, "B"]

    n = 11
    i = 4
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df, i))


def test_loc_range_index_prange(memory_leak_check):
    """test df.loc[int, col_ind] for RangeIndex in a parallel loop"""

    def impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        s = 0
        for i in bodo.prange(len(df)):
            s += df.loc[i, "B"]
        return s

    n = 11
    check_func(impl, (n,))


def test_loc_col_select(memory_leak_check):
    """test df.iloc[slice, col_ind] where col_ind is a list of column names or bools"""

    def impl1(df):
        return df.loc[:, ["A", "C"]]

    def impl2(df):
        return df.loc[:, [True, False, True]]

    def impl3(df):
        return df.loc[:, df.columns != "B"]

    def impl4(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        df.columns = ["AB", "CD", "EF"]
        return df.loc[:, ["AB", "EF"]]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))
    check_func(impl4, (n,))


@pytest.mark.slow
def test_getitem_loc_integer_cols():

    # TODO: BE-1325, support non string literals for scalar case
    # def impl1(df):
    #     return df.loc[0, 1]

    def impl2(df):
        return df.loc[:, [True, False, True, True, False, True]]

    def impl3(df):
        return df.loc[:, [1, 2, 3]]

    df = pd.DataFrame(
        data=np.arange(36).reshape(6, 6),
        columns=np.arange(6),
    )

    # check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.skip("Coverage gaps for df.loc, see BE-1324")
def test_loc_getitem_gaps():
    df = pd.DataFrame(
        data=np.arange(36).reshape(6, 6),
        columns=np.arange(6),
    )

    def impl(df):
        return df.loc[0]

    def impl2(df):
        return df.loc[:0]

    check_func(impl, (df,))
    check_func(impl2, (df,))


@pytest.mark.skip("TODO: add support for multi level getitem, see BE-1324")
def test_getitem_loc_multi_level_supported():
    """
    tests loc getitem on a dataframe with MultiIndexed columns/rows
    """

    def impl1(df):
        return df.loc[[True, True, False, False, True, False]]

    def impl2(df):
        return df.loc[df.A.CC > 2]

    def impl3(df):
        return df.loc[:0, "A"]

    def impl4(df):
        return df.loc[:0, ["A", "B"]]

    def impl5(df):
        return df.loc[:0, [True, True, False, False, True, False]]

    def impl6(df):
        return df.loc[:0, df.columns.get_level_values(0) > "A"]

    def impl7(df):
        return df.loc[[True, True, False, False, True, False], "A"]

    def impl8(df):
        return df.loc[df.A.CC > 2, "A"]

    # rep
    def impl9(df):
        return df.loc[[True, True, False, False, True, False], ["A", "B"]]

    def impl10(df):
        return df.loc[df.A.CC > 2, ["A", "B"]]

    # rep
    def impl11(df):
        return df.loc[
            [True, True, False, False, True, False],
            [True, True, False, False, True, False],
        ]

    def impl12(df):
        return df.loc[df.A.CC > 2, df.columns.get_level_values(0) > "A"]

    df = pd.DataFrame(
        data=np.arange(36).reshape(6, 6),
        columns=pd.MultiIndex.from_product((["A", "B"], ["CC", "DD", "EE"])),
        index=pd.MultiIndex.from_product(([0, 1], [1, 2, 3])),
    )

    check_func(impl1, (df,), dist_test=False)
    check_func(impl2, (df,))
    check_func(impl3, (df,))
    check_func(impl4, (df,))
    check_func(impl5, (df,), dist_test=False)
    check_func(impl6, (df,))
    check_func(impl7, (df,), dist_test=False)
    check_func(impl8, (df,))
    check_func(impl9, (df,), dist_test=False)
    check_func(impl10, (df,))
    check_func(impl11, (df,), dist_test=False)
    check_func(impl12, (df,))


def test_loc_setitem(memory_leak_check):
    """test df.loc[idx, col_ind] setitem where col_ind is a list of column names or bools"""

    # set existing column with full slice
    def impl1(df):
        df.loc[:, "B"] = 11
        return df

    # set new columns with full slice
    def impl2(df):
        df.loc[:, ["D", "E"]] = 11
        return df

    # set values with bool index
    def impl3(df):
        df.loc[df.A > 4, "B"] = 11
        return df

    # boolean column selection
    def impl4(df):
        df.loc[:, [True, False, True]] = 11
        return df

    # dynamic column selection
    def impl5(df):
        df.loc[:, df.columns != "B"] = 11
        return df

    # schema change
    def impl6(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        df.columns = ["AB", "CD", "EF"]
        df.loc[:, ["AB", "EF"]] = 11
        return df

    # boolean column selection
    def impl7(df):
        df.loc[df.A > 4, [True, False, True]] = 11
        return df

    # dynamic column selection
    def impl8(df):
        df.loc[df.A > 4, df.columns != "B"] = 11
        return df

    # schema change
    def impl9(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        cond = df.A > 4
        df.columns = ["AB", "CD", "EF"]
        df.loc[cond, ["AB", "EF"]] = 11
        return df

    # set columns using a 2D array
    def impl10(df, A):
        df.loc[:, ["D", "E"]] = A
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)
    check_func(impl3, (df,), copy_input=True)
    check_func(impl4, (df,), copy_input=True)
    check_func(impl5, (df,), copy_input=True)
    check_func(impl6, (n,))
    check_func(impl7, (df,), copy_input=True)
    check_func(impl8, (df,), copy_input=True)
    check_func(impl9, (n,))
    A = np.arange(2 * n).reshape(n, 2)
    check_func(impl10, (df, A))


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_loc_setitem_str(memory_leak_check):
    """test df.iloc[idx, col_ind] setitem for string array"""

    def impl(df):
        df.loc[df.A > 3, "B"] = "CCD"
        return df

    df = pd.DataFrame(
        {
            "A": [11, 4, 2, 5, 6, 1, 1, 4, 9],
            "B": ["AA", None, "ABC", "DD", None, "AAA", "", "EFG", None],
        }
    )
    check_func(
        impl, (df,), copy_input=True, only_1D=True, use_dict_encoded_strings=False
    )


def test_iat_getitem(df_value, memory_leak_check):
    """test df.iat[] getitem (single value)"""

    def impl(df, n):
        return df.iat[n - 1, 0]

    df = df_value.copy(deep=True)
    n = 1

    # Returning NaN won't match because of Bodo's strict typing
    # Replace all NaN with a different value.
    if pd.isna(df.iat[n - 1, 0]):
        df[df.columns[0]] = df[df.columns[0]].fillna(1)

    check_func(impl, (df, n))


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_iat_setitem_all_types(df_value, memory_leak_check):
    """test df.iat[] setitem (single value)"""

    def impl(df, n, val):
        df.iat[n - 1, 0] = val
        return df

    non_null_vals = df_value.dropna()
    if len(non_null_vals) == 0:
        # Only Nullable Int df has all nulls
        val = 1
    else:
        val = non_null_vals.iat[0, 0]

    check_func(
        impl,
        (df_value, len(df_value), val),
        copy_input=True,
        use_dict_encoded_strings=False,
    )


@pytest.mark.smoke
def test_iat_setitem():
    """test df.iat[] setitem (single value)"""

    def impl(df, n):
        df.iat[n - 1, 1] = n**2
        return df

    n = 11
    df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
    check_func(impl, (df, n), copy_input=True)


def test_df_schema_change(memory_leak_check):
    """
    Dataframe operations like setting new columns change the schema, so other
    operations need to handle type change during typing pass.

    df.drop() checks for drop columns to be in the schema, so it has to let typing pass
    change the type. This example is from the forecast code.
    """

    def test_impl(df):
        df["C"] = 3
        return df.drop(["C"], 1)

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(df), test_impl(df), check_column_type=False)


def test_set_df_column_names(memory_leak_check):
    """test setting dataframe column names using df.columns"""

    def impl1(df):
        df.columns = ["a", "b"]
        return df

    # invalid length
    def impl2(df):
        df.columns = ["a", "b", "c"]
        return df

    # type instability due to control flow
    def impl3(df, flag):
        if flag:
            df.columns = ["a", "b"]
        return df

    # non-constant column names
    def impl4(df, a):
        df.columns = a[1:]
        return df

    # test setattr on df with nested names (#2126)
    def impl5(df):
        df1 = df.groupby(["A"], as_index=False)
        df2 = df1.agg({"B": ["sum", "count"], "C": ["sum", "count"]})
        df2.columns = ["A", "testCol1", "count(B)", "testCol2", "count(C)"]
        return df2

    # test loop unrolling when necessary
    def impl6(df):
        df.columns = [x.lower() for x in df.columns]
        return df

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    check_func(impl1, (df,), copy_input=True)
    with pytest.raises(
        BodoError,
        match="DataFrame.columns: number of new column names does not match number of existing columns",
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError,
        match="DataFrame.columns: setting dataframe column names inside conditionals and loops not supported yet",
    ):
        bodo.jit(impl3)(df, False)
    with pytest.raises(
        BodoError, match="DataFrame.columns: new column names should be a constant list"
    ):
        bodo.jit(impl4)(df, ["a", "b", "c"])
    df = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1], "C": [2, 3, 1, 5]}
    )
    check_func(impl5, (df,), reset_index=True, sort_output=True)
    check_func(impl6, (df,), copy_input=True)


def test_set_df_index(memory_leak_check):
    """test setting dataframe index using df.index"""

    def impl1(df):
        df.index = ["AA", "BB", "CC", "DD"]
        return df

    def impl2(df):
        df.index = pd.Int64Index([3, 1, 2, 0])
        return df

    # type instability due to control flow
    def impl3(df, flag):
        if flag:
            df.index = ["AA", "BB", "CC", "DD"]
        return df

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    check_func(impl1, (df,), copy_input=True, dist_test=False)
    check_func(impl2, (df,), copy_input=True, dist_test=False)
    with pytest.raises(
        BodoError,
        match="DataFrame.index: setting dataframe index inside conditionals and loops not supported yet",
    ):
        bodo.jit(impl3)(df, True)


def test_df_multi_schema_change(memory_leak_check):
    """Test multiple df schema changes while also calling other Bodo functions.
    Makes sure global state variables in typing pass are saved properly and are not
    disrupted by calling another Bodo function (which calls the compiler recursively)
    """

    @bodo.jit
    def g(df):
        return df.assign(D=np.ones(len(df)))

    # inspired by user code (PO reconciliation project)
    def test_impl(df):
        df["C"] = 3
        df_cols = list(df.columns)
        df = g(df)
        df_cols = df_cols + ["D"]
        return df[df_cols]

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(df), test_impl(df), check_column_type=False)


def test_df_drop_column_check(memory_leak_check):
    def test_impl(df):
        return df.drop(columns=["C"])

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
    with pytest.raises(BodoError, match="not in DataFrame columns"):
        bodo.jit(test_impl)(df)


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type,
    reason="not supported for dict string type",
)
def test_df_fillna_str_inplace(memory_leak_check):
    """Make sure inplace fillna for string columns is reflected in output"""

    def test_impl(df):
        df.B.fillna("ABC", inplace=True)
        return df

    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["ab", "b", np.nan, "c", "bdd", "c", "a"]}
    )
    check_func(test_impl, (df_str,), copy_input=True, use_dict_encoded_strings=False)


def test_df_fillna_binary_inplace(memory_leak_check):
    """Make sure inplace fillna for string columns is reflected in output"""

    def test_impl(df):
        df.B.fillna(b"kjlkas", inplace=True)
        return df

    df_str = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [b"ab", b"", np.nan, b"hkjl", b"bddsad", b"asdfc", b"sdfa"],
        }
    )
    check_func(test_impl, (df_str,), copy_input=True)


def test_df_alias(memory_leak_check):
    """Test alias analysis for df data arrays. Without proper alias info, the fillna
    changes in data array will be optimized away incorrectly.
    This example is from the forecast code.
    """

    def test_impl():
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
        df.B.fillna(1, inplace=True)
        return df

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_column_type=False)


def test_df_type_unify_error():
    """makes sure type unification error is thrown when two dataframe schemas for the
    same variable are not compatible.
    """

    def test_impl(a):
        if len(a) > 3:  # some computation that cannot be inferred statically
            df = pd.DataFrame({"A": [1, 2, 3]})
        else:
            df = pd.DataFrame({"A": ["a", "b", "c"]})
        return df

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE

    # Test as a developer
    numba.core.config.DEVELOPER_MODE = 1

    if PYVERSION == (3, 10):
        # In Python 3.10 this function has two returns in the bytecode
        # as opposed to a phi node
        error_type = BodoError
        match_msg = "Unable to unify the following function return types"
    else:
        error_type = numba.TypingError
        match_msg = "Cannot unify dataframe"

    with pytest.raises(error_type, match=match_msg):
        bodo.jit(test_impl)([3, 4])

    # Reset back to original setting
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest.mark.slow
def test_dataframe_constant_lowering(memory_leak_check):
    df = pd.DataFrame({"A": [2, 1], "B": [1.2, 3.3]})

    def impl():
        return df

    pd.testing.assert_frame_equal(bodo.jit(impl)(), df, check_column_type=False)

    df2 = pd.DataFrame({"A": [2, 1], "B": [1.2, 3.3]})
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df2[f"F{i}"] = "ABC"

    def impl():
        return df2

    pd.testing.assert_frame_equal(
        bodo.jit(impl)(), df2, check_column_type=False, check_dtype=False
    )


def test_dataframe_columns_const_passing(memory_leak_check):
    """Test passing df.columns as a constant to another call"""

    def impl(df):
        return df.dropna(subset=df.columns, how="any")

    df = pd.DataFrame({"A": [2, 1], "B": [1.2, 3.3]})
    check_func(impl, (df,))


def test_dataframe_sample_number(memory_leak_check):
    """Checking the random routine is especially difficult to do.
    We can mostly only check incidental information about the code"""

    def f(df):
        return df.sample(n=4, replace=False).size

    bodo_f = bodo.jit(all_args_distributed_block=True, all_returns_distributed=False)(f)
    n = 10
    df = pd.DataFrame({"A": [x for x in range(n)]})
    py_output = f(df)
    df_loc = _get_dist_arg(df)
    assert bodo_f(df_loc) == py_output


@pytest.mark.slow
def test_dataframe_sample_uniform(memory_leak_check):
    """Checking the random routine, this time with uniform input"""

    def f1(df):
        return df.sample(n=4, replace=False)

    def f2(df):
        return df.sample(frac=0.5, replace=False)

    n = 10
    df = pd.DataFrame({"A": [1 for _ in range(n)]})
    check_func(f1, (df,), reset_index=True, is_out_distributed=False)
    check_func(f2, (df,), reset_index=True, is_out_distributed=False)


@pytest.mark.slow
def test_dataframe_sample_sorted(memory_leak_check):
    """Checking the random routine. Since we use sorted and the number of entries is equal to
    the number of sampled rows, after sorting the output becomes deterministic, that is independent
    of the random number generated"""

    def f(df, n):
        return df.sample(n=n, replace=False)

    n = 10
    df = pd.DataFrame({"A": [x for x in range(n)]})
    check_func(f, (df, n), reset_index=True, sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_dataframe_sample_index(memory_leak_check):
    """Checking that the index passed coherently to the A entry."""

    def f(df):
        return df.sample(5)

    df = pd.DataFrame({"A": list(range(20))})
    bodo_f = bodo.jit(all_args_distributed_block=False, all_returns_distributed=False)(
        f
    )
    df_ret = bodo_f(df)
    S = df_ret.index == df_ret["A"]
    assert S.all()


# TODO: fix leak and add memory_leak_check
@pytest.mark.slow
def test_dataframe_sample_nested_datastructures():
    """The sample function relies on allgather operations that deserve to be tested"""

    def check_gather_operation(df):
        siz = df.size

        def f(df, m):
            return df.sample(n=m, replace=False).size

        py_output = f(df, siz)
        start, end = get_start_end(len(df))
        df_loc = df.iloc[start:end]
        bodo_f = bodo.jit(
            all_args_distributed_block=True, all_returns_distributed=False
        )(f)
        df_ret = bodo_f(df_loc, siz)
        assert df_ret == py_output

    n = 10
    random.seed(1)
    df1 = pd.DataFrame({"B": gen_random_arrow_array_struct_int(10, n)})
    df2 = pd.DataFrame({"B": gen_random_arrow_array_struct_list_int(10, n)})
    df3 = pd.DataFrame({"B": gen_random_arrow_list_list_int(1, 0.1, n)})
    df4 = pd.DataFrame({"B": gen_random_arrow_struct_struct(10, n)})
    check_gather_operation(df1)
    check_gather_operation(df2)
    check_gather_operation(df3)
    check_gather_operation(df4)


@pytest.mark.slow
def test_dataframe_sample_frac_one_replace_false():
    def test_impl1(df):
        return df.sample(frac=1, replace=False)

    def test_impl2(df):
        return df.sample(frac=1.0)

    def test_impl3(df, num):
        return df.sample(n=num)

    df = pd.DataFrame(
        {"A": np.arange(10), "B": 1.5 + np.arange(10)},
        index=[f"i{i}" for i in range(10)],
    )
    n = len(df)

    check_func(test_impl1, (df,), sort_output=True, is_out_distributed=False)
    check_func(test_impl2, (df,), sort_output=True, is_out_distributed=False)
    check_func(
        test_impl3,
        (
            df,
            n,
        ),
        sort_output=True,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_dataframe_columns_name():
    """A little known feature of pandas dataframe is that one can attribute
    a name to the columns. As far as I know this shows up only in pivot_table
    and crosstab functionalities.
    -
    This feature is only partially supported in BODO. It is supported in
    boxing/unboxing, but it is not in gatherv which makes this test fail
    in distributed mode. When columns name are supported, remove the
    dist_test=False
    -
    A complete support of this feature in Bodo looks like a lot of work
    for only esthetic purposes."""

    def f(df):
        return df

    data = {"Name": ["Tom", "Jack", "nick", "juli"], "marks": [99, 98, 95, 90]}
    df = pd.DataFrame(data, index=["rank1", "rank2", "rank3", "rank4"])
    df.columns.name = "D"
    df.index.name = "A"
    check_func(f, (df,), dist_test=False)


@pytest.mark.slow
def test_dataframe_empty_with_index():
    """Make sure dataframe boxing works when dataframe has no columns but has a
    non-empty Index.
    """
    col_names = ColNamesMetaType(())

    def impl(A):
        return bodo.hiframes.pd_dataframe_ext.init_dataframe((), A, col_names)

    A = pd.Index([1, 3, 4, 11, 16, 19], dtype="Int64")
    check_func(impl, (A,), py_output=pd.DataFrame(index=A), only_seq=True)


def test_dataframe_columns_list():
    """Make sure loop unrolling can handle container updates inside list comprehension
    properly and handles binop of containers (#2473)
    """

    def impl1(nrows, nvars):
        X = np.zeros((nrows, nvars + 1))
        cols = ["y"] + ["x{}".format(i) for i in range(nvars)]
        df = pd.DataFrame(X, columns=cols)
        return df

    check_func(impl1, (10, 3), only_1D=True)


def test_unroll_loop(memory_leak_check, is_slow_run):
    """Test unrolling constant loops when necessary for typing in getitem/setitem of
    dataframe columns.
    """

    def impl1(df):
        s = 0
        for c in df.columns:
            s += df[c].sum()
        return s

    def impl2(df):
        s = 0
        for c in ["A", "B"]:
            if c != "A":
                s += df[c].sum()
        return s

    def impl3(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
        for c in df.columns:
            df[c + "2"] = (df[c] - df[c].mean()) / df[c].std()
        return df

    # loop with multiple exits shouldn't be transformed
    def impl4(df):
        s = 0
        i = 0
        c_list = ["A", "B"]
        while True:
            c = c_list[i]
            if c not in ["B"]:
                break
            s += df[c].sum()
            i += 1
            if i == len(c_list):
                break
        return s

    def impl5(n):
        df = pd.DataFrame({"A1": np.arange(n), "A2": np.arange(n) ** 2})
        s = 0
        for i in range(2):
            s += df["A" + str(i + 1)].sum()
        return df

    # unroll loop to enable typing (without the need for constant inference)
    def impl6(n):
        df = pd.DataFrame({"A1": np.arange(n), "A2": np.arange(n) ** 2})
        for _ in [3, 5, 9]:
            df = pd.concat((df, df + 1), axis=1)
        return df

    def impl7(n):
        df = pd.DataFrame({"A1": np.arange(n), "A2": np.arange(n) ** 2})
        for i in [3, 5, 9]:
            new_df = (df[["A1", "A2"]] + i).rename(
                columns={"A1": "A1_{}".format(i), "A2": "A2_{}".format(i)}
            )
            df = pd.concat((df, new_df), axis=1)
        return df

    # [BE-1332] two dependent loops need to unroll, and there is conditional list update
    def impl8(df):
        for c_name in [x for x in df.columns if "E" in x]:
            df[f"{c_name}_copy"] = df[c_name]
        return df

    # [BE-1354] extra condition in comprehension to be removed
    def impl9(df):
        for c_name in [x for x in df.columns if "E" in x if "A" not in x]:
            df[f"{c_name}_copy"] = df[c_name]
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    if is_slow_run:
        check_func(impl3, (n,))
    with pytest.raises(
        BodoError,
        match=r"df\[\] getitem selecting a subset of columns requires providing constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants.",
    ):
        bodo.jit(impl4)(df)
    check_func(impl5, (n,))
    check_func(impl6, (n,))
    check_func(impl7, (n,))
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "AE2": np.arange(n) ** 2,
            "C": np.ones(n),
            "E": np.ones(n) + 1,
        }
    )
    check_func(impl8, (df,), copy_input=True)
    check_func(impl9, (df,), copy_input=True)


def test_df_copy_update(memory_leak_check):
    """
    Test if df.copy() works as expected with
    deep=True.
    """

    def impl1(df):
        df1 = df.copy()
        df1["B"] = np.arange(len(df1))
        df1["A"][2] = 7
        return df, df1

    def impl2(df):
        df1 = df.copy(deep=False)
        df1["B"] = np.arange(len(df1))
        df1["A"][2] = 7
        return df, df1

    df = pd.DataFrame({"A": [1, 2, 3] * 10, "C": [2, 3, 4] * 10})
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df[f"A{i}"] = df["A"].copy()
    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)


@pytest.mark.slow
def test_unsupported_df_method():
    """Raise Bodo error for unsupported df methods"""

    def test_impl():
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
        return df.agg(["sum", "min"])

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_impl)()


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # Simple test (RangeIndex)
        pd.DataFrame({"A": [1, 2, 3], "B": [1.1, 2.2, np.nan]}),
        # string index
        pd.DataFrame(
            {"A": [1, 2, 3], "B": [1.1, 2.2, np.nan]}, index=["x1", "x2", "x3"]
        ),
        # NumericIndex
        pd.DataFrame(
            {"A": [1, 2, 3], "B": [1.1, 2.2, np.nan], "C": [1, 2, 3]},
            index=[1.2, 3.4, 5.6],
        ),
        # datetime
        pd.DataFrame(
            {"A": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)}
        ),
        # timedelta
        pd.DataFrame({"D": pd.Series(pd.timedelta_range(start="1 day", periods=4))}),
        # string
        pd.DataFrame({"A": [1, 2, 3], "B": ["xx", "yy", np.nan]}),
        # nullable int
        pd.DataFrame({"A": pd.Series([1, 2] * 20, dtype="Int32"), "B": [2, 3] * 20}),
        # categorical and boolean
        pd.DataFrame(
            {
                "BCDEF": [1, 2, 3, 4, 5],
                "LongNameTest": pd.Categorical([1, 2, 5, 3, 3], ordered=True),
                "D": [1.1, 2.2, 3.3, 4.4, np.nan],
                "E": [True, False, True, False, False],
            }
        ),
        # empty dataframe
        pd.DataFrame(),
    ],
)
def test_df_info(df):
    def impl(df):
        df.info()

    # Redirect print to a variable
    import io

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    bodo.jit(impl)(df)
    bodo_output = new_stdout.getvalue()
    bodo_output = bodo_output.splitlines()

    impl(df)
    py_output = new_stdout.getvalue()
    py_output = py_output.splitlines()

    # Return to normal std out
    sys.stdout = old_stdout
    if bodo.get_rank() == 0:
        # Assert first column information is the same
        # Or empty dataframe
        if len(df.columns) == 0:
            assert bodo_output[2] == py_output[2]
        else:
            assert bodo_output[5] == py_output[5]


def test_df_mem_usage(memory_leak_check):
    """Test DataFrame.memory_usage() with and w/o index"""

    def impl1(df):
        return df.memory_usage()

    def impl2(df):
        return df.memory_usage(index=False)

    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6], "B": [2.1, 3.2, 4.4, 5.2, 10.9, 6.8]},
        index=pd.date_range(start="2018-04-24", end="2018-04-27", periods=6, name="A"),
    )
    py_out = pd.Series([48, 48, 48], index=["Index", "A", "B"])
    check_func(impl1, (df,), py_output=py_out, is_out_distributed=False)
    py_out = pd.Series([48, 48], index=["A", "B"])
    check_func(impl2, (df,), py_output=py_out, is_out_distributed=False)
    # Empty DataFrame
    df = pd.DataFrame()
    # StringIndex only. Bodo has different underlying arrays than Pandas
    # Test sequential case
    py_out = pd.Series([8], index=pd.Index(["Index"]))
    check_func(impl1, (df,), only_seq=True, py_output=py_out, is_out_distributed=False)
    # Test parallel case. Index is replicated across ranks
    py_out = pd.Series([8 * bodo.get_size()], index=pd.Index(["Index"]))
    check_func(impl1, (df,), only_1D=True, py_output=py_out, is_out_distributed=False)

    # Empty and no index.
    check_func(impl2, (df,), is_out_distributed=False)


from .test_matplotlib import bodo_check_figures_equal


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple(fig_test, fig_ref):
    """
    Tests a basic example for df.plot replicated.
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016),
            yticks=(7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_scatter(fig_test, fig_ref):
    """
    Tests a basic example for df.plot(scatter)
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(kind="scatter", x="year", y="sales", ax=ax, figsize=(5, 5))

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_scatter(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_scatter(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016),
            yticks=(7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_dist(fig_test, fig_ref):
    """
    Tests a basic example for df.plot distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016, 2018),
            yticks=(6600000, 7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_scatter_dist(fig_test, fig_ref):
    """
    Tests a basic example for df.plot(scatter)
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(kind="scatter", x="year", y="sales", ax=ax, figsize=(5, 5))

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_scatter_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_scatter_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016, 2018),
            yticks=(6600000, 7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_x_y_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x and y are None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        # df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))
        df.plot(ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_x_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x is None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(y="sales", ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_y_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x is None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "month": ["Jan", "Feb", "March", "April", "May"],
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(x="year", ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)
