# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test join operations like df.merge(), df.join(), pd.merge_asof() ...
"""
import unittest
import os
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
from numba.untyped_passes import PreserveIR
from numba.typed_passes import NopythonRewrites
import bodo
from bodo.libs.str_arr_ext import StringArray
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    check_func,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
)
from bodo.utils.typing import BodoError
import pytest


# ------------------------------ merge() ------------------------------ #


class DeadcodeTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeleine used in test_join_deadcode_cleanup()
    additional PreserveIR pass then bodo_pipeline
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(True)
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, NopythonRewrites)
        pipeline.finalize()
        return [pipeline]


def test_merge_key_change():
    """
    Test merge(): make sure const list typing doesn't replace const key values
    """

    def test_impl(df1, df2, df3, df4):
        o1 = df1.merge(df2, on=["A"]).sort_values("A").reset_index(drop=True)
        o2 = df3.merge(df4, on=["B"]).sort_values("B").reset_index(drop=True)
        return o1, o2

    bodo_func = bodo.jit(test_impl)
    n = 11
    df1 = pd.DataFrame({"A": np.arange(n) + 3, "AA": np.arange(n) + 1.0})
    df2 = pd.DataFrame({"A": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
    df3 = pd.DataFrame({"B": 2 * np.arange(n) + 1, "BB": n + np.arange(n) + 1.0})
    df4 = pd.DataFrame({"B": 2 * np.arange(n) + 1, "BBB": n + np.arange(n) + 1.0})
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2, df3, df4)[0], test_impl(df1, df2, df3, df4)[0]
    )
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2, df3, df4)[1], test_impl(df1, df2, df3, df4)[1]
    )


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3]}),
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}),
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"A": [-1, 1, 3]}),
        pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1]}),
        pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1], "C": [-11, 0, 4]}),
    ],
)
def test_merge_common_cols(df1, df2):
    """
    test merge() default behavior: 
    merge on common columns when key columns not provided
    """

    def impl(df1, df2):
        df3 = df1.merge(df2)
        return df3

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A").reset_index(drop=True),
        impl(df1, df2).sort_values("A").reset_index(drop=True),
    )


def test_merge_left1():
    """
    Test merge(): 'how' = left on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="left", on="key")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], np.float)}
    )
    df2 = pd.DataFrame(
        {"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], np.float)}
    )

    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("key").reset_index(drop=True),
        test_impl(df1, df2).sort_values("key").reset_index(drop=True),
    )


def test_merge_left2():
    """
    Test merge(): 'how' = left on specified integer column
    where a key is repeated on left but not right side
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="left", on="key")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 3, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], np.float)}
    )
    df2 = pd.DataFrame(
        {"key": [1, 2, 9, 3, 10], "B": np.array([1, 7, 2, 6, 5], np.float)}
    )

    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("key").reset_index(drop=True),
        test_impl(df1, df2).sort_values("key").reset_index(drop=True),
    )


def test_merge_right():
    """
    Test merge(): 'how' = right on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="right", on="key")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], np.float)}
    )
    df2 = pd.DataFrame(
        {"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], np.float)}
    )

    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("key").sort_values("A").reset_index(drop=True),
        test_impl(df1, df2).sort_values("key").sort_values("A").reset_index(drop=True),
    )


def test_merge_outer():
    """
    Test merge(): 'how' = outer on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="outer", on="key")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], np.float)}
    )
    df2 = pd.DataFrame(
        {"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], np.float)}
    )

    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("key").reset_index(drop=True),
        test_impl(df1, df2).sort_values("key").reset_index(drop=True),
    )


@pytest.mark.parametrize("n", [11, 11111])
def test_merge_int_key(n):
    """
    Test merge(): key column is of type int
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, left_on="key1", right_on="key2")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
    df2 = pd.DataFrame({"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0})
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("key1").reset_index(drop=True),
        test_impl(df1, df2).sort_values("key1").reset_index(drop=True),
    )


def test_merge_multi_int_key():
    """
    Test merge(): sequentially merge on more than one integer key columns
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on=["A", "B"])
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
    )

    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A").reset_index(drop=True),
        test_impl(df1, df2).sort_values("A").reset_index(drop=True),
    )


def test_merge_str_key():
    """
    Test merge(): sequentially merge on key column of type string
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, left_on="key1", right_on="key2")
        return df3.B

    df1 = pd.DataFrame({"key1": ["foo", "bar", "baz"]})
    df2 = pd.DataFrame({"key2": ["baz", "bar", "baz"], "B": ["b", "zzz", "ss"]})

    bodo_func = bodo.jit(test_impl)
    assert set(bodo_func(df1, df2)) == set(test_impl(df1, df2))


def test_merge_str_nan1():
    """
    test merging dataframes containing string columns with nan values
    """

    def test_impl(df1, df2):
        return pd.merge(df1, df2, left_on="key1", right_on="key2")

    df1 = pd.DataFrame(
        {"key1": ["foo", "bar", "baz", "baz"], "A": ["b", "", np.nan, "ss"]}
    )
    df2 = pd.DataFrame(
        {"key2": ["baz", "bar", "baz", "foo"], "B": ["b", np.nan, "", "AA"]}
    )

    check_func(test_impl, (df1, df2), sort_output=True)


def _gen_df_str(n):
    """
    helper function that generate dataframe with int and string columns
    """
    str_vals = []
    for _ in range(n):
        # store NA with 30% chance
        if random.random() < 0.3:
            str_vals.append(np.nan)
            continue

        k = random.randint(1, 10)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
        str_vals.append(val)

    A = np.random.randint(0, 100, n)
    df = pd.DataFrame({"A": A, "B": str_vals})
    return df


def test_merge_str_nan2():
    """
    test merging dataframes containing string columns with nan values
    on larger dataframes
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on="A")

    # seeds should be the same on different processors for consistent input
    random.seed(2)
    np.random.seed(3)
    n = 1211
    df1 = _gen_df_str(n)
    df2 = _gen_df_str(n)
    check_func(test_impl, (df1, df2), sort_output=True)


def test_merge_bool_nan():
    """
    test merging dataframes containing boolean columns with nan values
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on=["A"])

    # XXX the test can get stuck if output of join for boolean arrays is empty
    # or just nan on some processor, since default type is string for object
    # arrays, resulting in inconsistent types
    df1 = pd.DataFrame(
        {
            "A": [3, 1, 1, 3, 4, 2, 4, 11],
            "B": [True, False, True, False, np.nan, True, False, True],
        }
    )

    df2 = pd.DataFrame(
        {
            "A": [2, 1, 4, 4, 3, 2, 4, 11],
            "C": [False, True, np.nan, False, False, True, False, True],
        }
    )
    check_func(test_impl, (df1, df2), sort_output=True, check_dtype=False)


def test_merge_out_str_na():
    """
    Test merge(): setting NA in output string data column
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, left_on="key1", right_on="key2", how="left")
        return df3.B

    df1 = pd.DataFrame({"key1": ["foo", "bar", "baz"]})
    df2 = pd.DataFrame({"key2": ["baz", "bar", "baz"], "B": ["b", "zzz", "ss"]})

    bodo_func = bodo.jit(test_impl)
    assert set(bodo_func(df1, df2)) == set(test_impl(df1, df2))


def test_merge_datetime():
    """
    Test merge(): merge on key column of type DatetimeIndex
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, on="time")
        return df3

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(["2017-01-03", "2017-01-06", "2017-02-21"]),
            "B": [4, 5, 6],
        }
    )
    df2 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(["2017-01-01", "2017-01-06", "2017-01-03"]),
            "A": [7, 8, 9],
        }
    )
    pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))


def test_merge_datetime_parallel():
    """
    Test merge(): merge on key column of type DatetimeIndex
    ensure parallelism
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, on="time")
        return (df3.A.sum(), df3.time.max(), df3.B.sum())

    bodo_func = bodo.jit(distributed=["df1", "df2"])(test_impl)
    df1 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(["2017-01-03", "2017-01-06", "2017-02-21"]),
            "B": [4, 5, 6],
        }
    )
    df2 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(["2017-01-01", "2017-01-06", "2017-01-03"]),
            "A": [7, 8, 9],
        }
    )
    start1, end1 = get_start_end(len(df1))
    start2, end2 = get_start_end(len(df2))
    assert bodo_func(df1.iloc[start1:end1], df2.iloc[start2:end2]) == test_impl(
        df1, df2
    )
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}),
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1]}),
        pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1], "C": [-11, 0, 4]}),
    ],
)
def test_merge_suffix(df1, df2):
    """
    test merge() default behavior: 
    column name overlaps, require adding suffix('_x', '_y') to column names
    """

    def impl1(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A").reset_index(drop=True),
        impl1(df1, df2).sort_values("A").reset_index(drop=True),
    )

    def impl2(df1, df2):
        df3 = df1.merge(df2, on=["B", "A"])
        return df3

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl2(df1, df2))


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}, index=[1, 4, 3]),
        pd.DataFrame(
            {"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}, index=[1, 4, 3]
        ),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1]}, index=[-1, 1, 3]),
        pd.DataFrame(
            {"A": [-1, 1, 3], "B": [-1, 0, 1], "C": [-11, 0, 4]}, index=[-1, 1, 3]
        ),
    ],
)
def test_merge_index(df1, df2):
    """
    merge with left_index and right_index specified, merge using index
    """

    def impl1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A_x").reset_index(drop=True),
        impl1(df1, df2).sort_values("A_x").reset_index(drop=True),
    )

    # pandas duplicates key column if one side is using index
    # TODO: replicate pandas behavior
    # def impl2(df1, df2):
    #     return df1.merge(df2, left_on='A', right_index=True)

    # bodo_func = bodo.jit(impl2)
    # print(bodo_func(df1, df2), impl2(df1, df2))
    # pd.testing.assert_frame_equal(bodo_func(df1, df2), impl2(df1, df2))

    # def impl3(df1, df2):
    #     return df1.merge(df2, left_index=True, right_on='A')

    # bodo_func = bodo.jit(impl3)
    # pd.testing.assert_frame_equal(bodo_func(df1, df2), impl3(df1, df2))


def test_merge_match_key_types():
    """
    test merge() where key types mismatch but values can be equal
    happens especially when Pandas convert ints to float to use np.nan
    """

    def test_impl1(df1, df2):
        return df1.merge(df2, on=["A"])

    def test_impl2(df1, df2):
        return df1.merge(df2, on=["A", "B"])

    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [1, 2, 3, 2, 3]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 3, 2, 3, 2]}
    )
    df2["A"] = df2.A.astype(np.float64)
    check_func(test_impl1, (df1, df2), sort_output=True)
    check_func(test_impl1, (df2, df1), sort_output=True)
    check_func(test_impl2, (df1, df2), sort_output=True)
    check_func(test_impl2, (df2, df1), sort_output=True)


def test_merge_cat1_inner():
    """
    Test merge(): merge dataframes containing categorical values
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.int, "C2": ct_dtype, "C3": str}
        df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        n = len(df1)
        df2 = pd.DataFrame({"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1")
        return df3

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(
        bodo_func().sort_values("C1").reset_index(drop=True),
        test_impl().sort_values("C1").reset_index(drop=True),
    )


def test_merge_cat1_right_2cols1():
    """
    Test merge(): setting NaN in categorical array
    a smaller test case for test_join_cat1_right()
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat3.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.int, "C2": ct_dtype}
        df1 = pd.read_csv(fname, names=["C1", "C2"], dtype=dtypes)
        n = len(df1)
        df2 = pd.DataFrame({"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(
        bodo_func().sort_values("C1").reset_index(drop=True),
        test_impl().sort_values("C1").reset_index(drop=True),
    )


def test_merge_cat1_right_2cols2():
    """
    Test merge(): setting NaN in categorical array
    bug fixed: some get_item_size that did not work for strings
    a smaller test case for test_join_cat1_right()
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat4.csv")

    def test_impl():
        dtypes = {"C1": np.int, "C2": str}
        df1 = pd.read_csv(fname, names=["C1", "C2"], dtype=dtypes)
        n = len(df1)
        df2 = pd.DataFrame({"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(
        bodo_func().sort_values("C1").reset_index(drop=True),
        test_impl().sort_values("C1").reset_index(drop=True),
    )


def test_merge_cat1_right():
    """
    Test merge(): setting NaN in categorical array
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.int, "C2": ct_dtype, "C3": str}
        df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        n = len(df1)
        df2 = pd.DataFrame({"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(
        bodo_func().sort_values("C1").reset_index(drop=True),
        test_impl().sort_values("C1").reset_index(drop=True),
    )


@pytest.mark.parametrize("n", [11, 11111])
def test_merge_parallel_optmize(n):
    """
    Test merge(): ensure parallelism optimization
    """

    def test_impl(n):
        df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
        df2 = pd.DataFrame({"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0})
        df3 = pd.merge(df1, df2, left_on="key1", right_on="key2")
        return df3.B.sum()

    bodo_func = bodo.jit(test_impl)
    check_func(test_impl, (n,))
    assert count_array_REPs() == 0  # assert parallelism
    assert count_parfor_REPs() == 0  # assert parallelism


def test_merge_left_parallel():
    """
    Test merge(): merge with only left dataframe columns distributed
    ensure parallelism
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on=["A", "B"])
        return df3.C.sum() + df3.D.sum()

    bodo_func = bodo.jit(distributed=["df1"])(test_impl)
    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
    )
    start, end = get_start_end(len(df1))
    assert test_impl(df1, df2) == bodo_func(df1.iloc[start:end], df2)


def test_join_rm_dead_data_name_overlap1():
    """
    Test join dead code elimination when there are matching names in data columns of
    input tables but only one of them is actually used.
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on="user_id")
        return len(df3.id_x.values)

    df1 = pd.DataFrame({"id": [3, 4], "user_id": [5, 6]})
    df2 = pd.DataFrame({"id": [3, 4], "user_id": [5, 6]})
    assert bodo.jit(test_impl)(df1, df2) == test_impl(df1, df2)


def test_join_rm_dead_data_name_overlap2():
    """
    Test join dead code elimination when there are matching names in data columns of
    input tables but only one of them is actually used.
    """

    def test_impl(df1, df2):
        return df1.merge(df2, left_on=["id"], right_on=["user_id"])

    df1 = pd.DataFrame({"id": [3, 4, 1]})
    df2 = pd.DataFrame({"id": [3, 4, 2], "user_id": [3, 5, 6]})
    pd.testing.assert_frame_equal(bodo.jit(test_impl)(df1, df2), test_impl(df1, df2))


def test_join_deadcode_cleanup():
    """
    Test join dead code elimination when a merged dataframe is never used, 
    merge() is not executed
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on=["A"])
        return

    def test_impl_with_join(df1, df2):
        df3 = df1.merge(df2, on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "C": [4, 5, 6]})

    j_func = numba.njit(pipeline_class=DeadcodeTestPipeline)(test_impl)
    j_func_with_join = numba.njit(pipeline_class=DeadcodeTestPipeline)(
        test_impl_with_join
    )
    j_func(df1, df2)  # calling the function to get function IR
    j_func_with_join(df1, df2)
    fir = j_func.overloads[j_func.signatures[0]].metadata["preserved_ir"]
    fir_with_join = j_func_with_join.overloads[j_func.signatures[0]].metadata[
        "preserved_ir"
    ]

    for block in fir.blocks.values():
        for statement in block.body:
            assert not isinstance(statement, bodo.ir.join.Join)

    joined = False
    for block in fir_with_join.blocks.values():
        for statement in block.body:
            if isinstance(statement, bodo.ir.join.Join):
                joined = True
                break
        if joined:
            break
    assert joined


# ------------------------------ join() ------------------------------ #


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}, index=[1, 4, 3]),
        pd.DataFrame(
            {"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}, index=[1, 4, 3]
        ),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"D": [-1.0, 1.0, 3.0]}, index=[-1, 1, 3]),
        pd.DataFrame({"D": [-1.0, 1.0, 3.0], "E": [-1.0, 0.0, 1.0]}, index=[-1, 1, 3]),
    ],
)
def test_join_call(df1, df2):
    """
    test join() default behavior: 
    impl1(): join on index when key columns not provided
    impl2(): left on key column, right on index
    """

    def impl1(df1, df2):
        df3 = df1.join(df2)
        return df3

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A").reset_index(drop=True),
        impl1(df1, df2).sort_values("A").reset_index(drop=True),
    )

    def impl2(df1, df2):
        return df1.join(df2, on="A")

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2).sort_values("A").reset_index(drop=True),
        impl2(df1, df2).sort_values("A").reset_index(drop=True),
    )


# ------------------------------ merge_asof() ------------------------------ #


def test_merge_asof_seq():
    """
    Test merge_asof(): merge_asof sequencially on key column of type DatetimeIndex
    """

    def test_impl(df1, df2):
        return pd.merge_asof(df1, df2, on="time")

    bodo_func = bodo.jit(test_impl)
    df1 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(["2017-01-03", "2017-01-06", "2017-02-21"]),
            "B": [4, 5, 6],
        }
    )
    df2 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(
                ["2017-01-01", "2017-01-02", "2017-01-04", "2017-02-23", "2017-02-25"]
            ),
            "A": [2, 3, 7, 8, 9],
        }
    )
    pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))


def test_merge_asof_parallel(datapath):
    """
    Test merge_asof(): merge_asof in parallel on key column of type DatetimeIndex
    """
    fname1 = datapath("asof1.pq")
    fname2 = datapath("asof2.pq")

    def impl():
        df1 = pd.read_parquet(fname1)
        df2 = pd.read_parquet(fname2)
        df3 = pd.merge_asof(df1, df2, on="time")
        return (df3.A.sum(), df3.time.max(), df3.B.sum())

    bodo_func = bodo.jit(impl)
    assert bodo_func() == impl()


class TestJoin(unittest.TestCase):
    @pytest.mark.slow
    def test_join_parallel(self):
        """
        Test merge(): ensure parallelism optimization
        """

        def test_impl(n):
            df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
            df2 = pd.DataFrame(
                {"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0}
            )
            df3 = pd.merge(df1, df2, left_on="key1", right_on="key2")
            return df3.B.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)  # assert parallelism
        self.assertEqual(count_parfor_REPs(), 0)  # assert parallelism
        n = 11111
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_merge_cat_parallel1(self):
        # TODO: cat as keys
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
            dtypes = {"C1": np.int, "C2": ct_dtype, "C3": str}
            df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
            n = len(df1)
            df2 = pd.DataFrame(
                {"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0}
            )
            df3 = df1.merge(df2, on="C1")
            return df3

        bodo_func = bodo.jit(distributed=["df3"])(test_impl)
        # TODO: check results
        self.assertTrue((bodo_func().columns == test_impl().columns).all())


if __name__ == "__main__":
    unittest.main()
