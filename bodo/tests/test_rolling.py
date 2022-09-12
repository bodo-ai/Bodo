# Copyright (C) 2022 Bodo Inc. All rights reserved.
import itertools
import os
import re
import unittest

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.hiframes.rolling import supported_rolling_funcs
from bodo.tests.utils import check_func, count_array_REPs, count_parfor_REPs
from bodo.utils.typing import BodoError

LONG_TEST = (
    int(os.environ["BODO_LONG_ROLLING_TEST"]) != 0
    if "BODO_LONG_ROLLING_TEST" in os.environ
    else False
)

test_funcs = ("mean", "max")
if LONG_TEST:
    # all functions except apply, cov, corr
    test_funcs = supported_rolling_funcs[:-3]


@pytest.fixture(
    params=[
        pd.DataFrame({"B": [0, 1, 2, np.nan, 4]}, [4, 1, 3, 0, -1])
        # pd.DataFrame({'B': [0, 1, 2, -2, 4]}, [4, 1, 3, 0, -1])
    ]
)
def test_df(request, memory_leak_check):
    return request.param


@pytest.mark.smoke
def test_fixed_index(test_df, memory_leak_check):
    def impl(df):
        return df.rolling(2).mean()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(test_df), impl(test_df), check_column_type=False
    )


@pytest.mark.slow
def test_rolling_cov_unsupported_args(memory_leak_check):
    def impl1(df):
        return df.rolling(2).cov(df.A)

    def impl2(df):
        return df.rolling(2).cov(df, pairwise=True)

    df = pd.DataFrame(
        {
            "A": [1.51, 2.421, 233232, 12.21] * 5,
        }
    )

    err_msg = re.escape(
        "DataFrame.rolling.cov(): requires providing a DataFrame for 'other'"
    )
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)
    err_msg = "pairwise parameter only supports default value None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)


@pytest.mark.slow
def test_rolling_corr_unsupported_args(memory_leak_check):
    def impl1(df):
        return df.rolling(2).corr(df.A)

    def impl2(df):
        return df.rolling(2).corr(df, pairwise=True)

    df = pd.DataFrame(
        {
            "A": [1.51, 2.421, 233232, 12.21] * 5,
        }
    )

    err_msg = re.escape(
        "DataFrame.rolling.corr(): requires providing a DataFrame for 'other'"
    )
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)
    err_msg = "pairwise parameter only supports default value None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)


@pytest.mark.slow
def test_rolling_unsupported(test_df, memory_leak_check):
    """
    Test an unsupported argument for df.rolling
    """

    def impl(df):
        return df.rolling(2, axis=1)

    err_msg = "axis parameter only supports default value 0"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(test_df)


# TODO [BE-1797]: Add memory leak check
@pytest.mark.slow
def test_fixed_index_groupby():
    def impl(df):
        return df.groupby("B").rolling(2).mean()

    df = pd.DataFrame(
        {
            "A": [1, 2, 24, None] * 5,
            "B": ["421", "f31"] * 10,
            "C": [1.51, 2.421, 233232, 12.21] * 5,
        }
    )
    # Groupby ordering isn't defined so we must sort.
    check_func(impl, (df,), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_variable_on_index(memory_leak_check):
    def impl(df):
        return df.rolling("2s").mean()

    bodo_func = bodo.jit(impl)
    df = pd.DataFrame(
        {"B": [0, 1, 2, np.nan, 4]},
        [
            pd.Timestamp("20130101 09:00:00"),
            pd.Timestamp("20130101 09:00:02"),
            pd.Timestamp("20130101 09:00:03"),
            pd.Timestamp("20130101 09:00:05"),
            pd.Timestamp("20130101 09:00:06"),
        ],
    )
    pd.testing.assert_frame_equal(bodo_func(df), impl(df), check_column_type=False)


@pytest.mark.slow
def test_column_select(memory_leak_check):
    """test selecting columns explicitly in rolling calls"""

    def impl1(df):
        return df.rolling("2s", on="C")["B"].mean()

    def impl2(df):
        return df.rolling("2s", on="C")["A", "B"].mean()

    def impl3(df):
        return df.rolling(2)["B"].mean()

    def impl4(df):
        return df.rolling(3)["A", "B"].mean()

    def impl5(df):
        return df.rolling(3).B.mean()

    def impl6(df):
        return df.rolling("2s", on="C").B.mean()

    # select a single column but using a list
    def impl7(df):
        return df.rolling("2s", on="C")[["B"]].mean()

    df = pd.DataFrame(
        {
            "A": [5, 12, 21, np.nan, 3],
            "B": [0, 1, 2, np.nan, 4],
            "C": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
            ],
        }
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))
    check_func(impl4, (df,))
    check_func(impl5, (df,))
    check_func(impl6, (df,))
    check_func(impl7, (df,))


@pytest.mark.slow
def test_min_periods(memory_leak_check):
    """test min_periods argument in rolling calls"""
    # fixed window
    def impl1(df, center, minp):
        return df.rolling(3, center=center, min_periods=minp)[["A", "B"]].mean()

    # fixed window with UDF
    def impl2(df, center, minp):
        return df.rolling(3, center=center, min_periods=minp)[["A", "B"]].apply(
            lambda a: a.sum()
        )

    # variable window
    def impl3(df, minp):
        return df.rolling("2s", min_periods=minp, on="C")[["A", "B"]].mean()

    # variable window with UDF
    def impl4(df, minp):
        return df.rolling("2s", min_periods=minp, on="C")[["A", "B"]].apply(
            lambda a: a.sum()
        )

    df = pd.DataFrame(
        {
            "A": [5, 12, np.nan, np.nan, 3, np.nan],
            "B": [0, 1, 2, np.nan, 4, np.nan],
            "C": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
                pd.Timestamp("20130101 09:00:07"),
            ],
        }
    )
    check_func(impl1, (df, False, 1))
    check_func(impl1, (df, True, 1))
    check_func(impl1, (df, False, 2))
    check_func(impl1, (df, True, 2))

    check_func(impl2, (df, False, 1))
    check_func(impl2, (df, True, 1))
    check_func(impl2, (df, False, 2))
    check_func(impl2, (df, True, 2))

    check_func(impl3, (df, 1))
    check_func(impl3, (df, 2))
    check_func(impl4, (df, 1))
    check_func(impl4, (df, 2))


@pytest.mark.slow
def test_apply_raw_false(memory_leak_check):
    """make sure raw=False argument of apply() works (which is the default)"""

    def impl1(df):
        return df.rolling(2)["B"].apply(lambda a: a.idxmin())

    def impl2(df):
        # Change to min. Possible bug in pandas
        return df.rolling("2s", on="C").apply(lambda a: a.min())

    df = pd.DataFrame(
        {
            "A": [5, 12, 21, 3.2, 3],
            "B": [0, 1, 2, 1.9, 4],
            "C": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
            ],
        },
        index=[3, 1, 5, 11, 3],
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))


@pytest.mark.slow
def test_nullable_int(memory_leak_check):
    def impl(S):
        return S.rolling(2).sum()

    S = pd.Series([4, 1, 2, 11, 33, -12], dtype="Int64")
    check_func(impl, (S,))


@bodo.jit(distributed=False)
def g(a):
    return a.sum()


@pytest.mark.slow
def test_fixed_apply_nested_func(memory_leak_check):
    """test nested UDF decorated with Bodo (make sure it doesn't hang due to barriers)"""
    # test sequentially with manually created dfs
    def test_impl(df):
        return df.rolling(2).apply(lambda a: g(a))

    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_groupby_rolling(is_slow_run):
    """test groupby rolling combination"""

    def impl1(df):
        return df.groupby("A").C.rolling(2).mean()

    def impl2(df):
        return (
            df.groupby("A")[["C", "D"]]
            .rolling(3, center=True)
            .apply(lambda x: x.sum() / 2)
        )

    def impl3(df):
        return df.groupby("A").rolling("2s", on="time")["B"].mean()

    df = pd.DataFrame(
        {
            "A": [1, 4, 4, 11, 4, 1] * 3,
            "C": [1.1, 2.2, 3.3, 4.4, 5.5, -1.1] * 3,
            "D": [3, 1, 2, 4, 5, 5] * 3,
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    if not is_slow_run:
        return
    check_func(impl2, (df,), sort_output=True, reset_index=True)
    df = pd.DataFrame(
        {
            "A": [1, 4, 4, 11, 4, 1, 1, 1, 4],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
            "time": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
                pd.Timestamp("20130101 09:00:07"),
                pd.Timestamp("20130101 09:00:08"),
                pd.Timestamp("20130101 09:00:10"),
                pd.Timestamp("20130101 09:00:11"),
            ],
        }
    )
    check_func(impl3, (df,), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_skip_non_numeric_columns(memory_leak_check):
    """Make sure non-numeric columns are skipped properly"""

    def impl1(df):
        return df.rolling(2).sum()

    def impl2(df):
        return df.rolling("2s", on="time").sum()

    df = pd.DataFrame(
        {
            "A": [1, 4, 4, 11, 4, 1, 1, 1, 4],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
            "C": ["A", "AB", "C", "D", "E", "AA", "BB", "C", "CC"],
            "D": [True, False, False, True, True, False, False, True, False],
            "time": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
                pd.Timestamp("20130101 09:00:07"),
                pd.Timestamp("20130101 09:00:08"),
                pd.Timestamp("20130101 09:00:10"),
                pd.Timestamp("20130101 09:00:11"),
            ],
        }
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))


@pytest.mark.slow
def test_rolling_error_checking():
    """test error checking in rolling calls"""

    # center should be boolean
    def impl1(df):
        return df.rolling(2, center=2)["B"].mean()

    # min_periods should be int
    def impl2(df):
        return df.rolling(2, min_periods=False)["B"].mean()

    # min_periods should be positive
    def impl3(df):
        return df.rolling("2s", on="C", min_periods=-3)["B"].mean()

    # min_periods should be less than window size
    def impl4(df):
        return df.rolling(2, min_periods=3)["B"].mean()

    # center is not implemented for datetime windows
    def impl5(df):
        return df.rolling("2s", on="C", center=True)["B"].mean()

    # window should be non-negative
    def impl6(df):
        return df.rolling(-2)["B"].mean()

    # 'on' not supported for Series yet
    def impl7(df):
        return df.A.rolling("2s", on="C").mean()

    # 'on' should be constant
    def impl8(df):
        return df.rolling("2s", on=df.B)["B"].mean()

    # 'on' should be datetime64[ns]
    def impl9(df):
        return df.rolling("2s", on="A")["B"].mean()

    # window should be int or offset
    def impl10(df):
        return df.rolling(1.4)["B"].mean()

    # check window to be valid offset
    def impl11(df):
        return df.rolling("22")["B"].mean()

    # input should have numeric types
    def impl12(df):
        return df.rolling(3).mean()

    # func should be function
    def impl13(df):
        return df.rolling(3).apply(4)

    # raw should be bool
    def impl14(df):
        return df.rolling(3).apply(lambda a: a.sum(), raw=3)

    df = pd.DataFrame(
        {
            "A": [5, 12, 21, np.nan, 3],
            "B": [0, 1, 2, np.nan, 4],
            "C": [
                pd.Timestamp("20130101 09:00:00"),
                pd.Timestamp("20130101 09:00:02"),
                pd.Timestamp("20130101 09:00:03"),
                pd.Timestamp("20130101 09:00:05"),
                pd.Timestamp("20130101 09:00:06"),
            ],
        }
    )
    with pytest.raises(BodoError, match=r"rolling\(\): center must be a boolean"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match=r"rolling\(\): min_periods must be an integer"):
        bodo.jit(impl2)(df)
    with pytest.raises(ValueError, match=r"min_periods must be >= 0"):
        bodo.jit(impl3)(df)
    with pytest.raises(ValueError, match=r"min_periods must be <= window"):
        bodo.jit(impl4)(df)
    with pytest.raises(
        NotImplementedError, match=r"center is not implemented for datetimelike"
    ):
        bodo.jit(impl5)(df)
    with pytest.raises(ValueError, match=r"window must be non-negative"):
        bodo.jit(impl6)(df)
    with pytest.raises(BodoError, match=r"'on' not supported for Series yet"):
        bodo.jit(impl7)(df)
    with pytest.raises(BodoError, match=r"'on' should be a constant column name"):
        bodo.jit(impl8)(df)
    with pytest.raises(BodoError, match=r"'on' column should have datetime64 data"):
        bodo.jit(impl9)(df)
    with pytest.raises(BodoError, match=r"'window' should be int or time offset"):
        bodo.jit(impl10)(df)
    with pytest.raises(ValueError, match=r"Invalid offset value"):
        bodo.jit(impl11)(df)
    with pytest.raises(BodoError, match=r"No numeric types to aggregate"):
        bodo.jit(impl12)(df[["C"]])
    with pytest.raises(BodoError, match=r"'func' parameter must be a function"):
        bodo.jit(impl13)(df)
    with pytest.raises(BodoError, match=r"'raw' parameter must be bool"):
        bodo.jit(impl14)(df)


@pytest.mark.slow
class TestRolling(unittest.TestCase):
    def test_fixed1(self):
        # test sequentially with manually created dfs
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        for func_name in test_funcs:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(
                func_name
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)

            for args in itertools.product(wins, centers):
                df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
                pd.testing.assert_frame_equal(
                    bodo_func(df, *args), test_impl(df, *args), check_column_type=False
                )
                df = pd.DataFrame({"B": [0, 1, 2, -2, 4]})
                pd.testing.assert_frame_equal(
                    bodo_func(df, *args), test_impl(df, *args), check_column_type=False
                )

    def test_fixed2(self):
        # test sequentially with generated dfs
        sizes = (121,)
        wins = (3,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 3, 5)
        centers = (False, True)
        for func_name in test_funcs:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(
                func_name
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for n, w, c in itertools.product(sizes, wins, centers):
                df = pd.DataFrame({"B": np.arange(n)})
                pd.testing.assert_frame_equal(
                    bodo_func(df, w, c), test_impl(df, w, c), check_column_type=False
                )

    def test_fixed_apply1(self):
        # test sequentially with manually created dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())

        bodo_func = bodo.jit(test_impl)
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)
        for args in itertools.product(wins, centers):
            df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
            pd.testing.assert_frame_equal(
                bodo_func(df, *args), test_impl(df, *args), check_column_type=False
            )
            df = pd.DataFrame({"B": [0, 1, 2, -2, 4]})
            pd.testing.assert_frame_equal(
                bodo_func(df, *args), test_impl(df, *args), check_column_type=False
            )

    def test_fixed_apply2(self):
        # test sequentially with generated dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())

        bodo_func = bodo.jit(test_impl)
        sizes = (121,)
        wins = (3,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 3, 5)
        centers = (False, True)
        for n, w, c in itertools.product(sizes, wins, centers):
            df = pd.DataFrame({"B": np.arange(n)})
            pd.testing.assert_frame_equal(
                bodo_func(df, w, c), test_impl(df, w, c), check_column_type=False
            )

    def test_fixed_parallel1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({"B": np.arange(n)})
            R = df.rolling(w, center=center).sum()
            return R.B.sum()

        bodo_func = bodo.jit(test_impl)
        sizes = (121,)
        wins = (5,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 4, 5, 10, 11)
        centers = (False, True)
        for args in itertools.product(sizes, wins, centers):
            self.assertEqual(
                bodo_func(*args),
                test_impl(*args),
                "rolling fixed window with {}".format(args),
            )
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_fixed_parallel_apply1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({"B": np.arange(n)})
            R = df.rolling(w, center=center).apply(lambda a: a.sum())
            return R.B.sum()

        bodo_func = bodo.jit(test_impl)
        sizes = (121,)
        wins = (5,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 4, 5, 10, 11)
        centers = (False, True)
        for args in itertools.product(sizes, wins, centers):
            self.assertEqual(
                bodo_func(*args),
                test_impl(*args),
                "rolling fixed window with {}".format(args),
            )
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_variable1(self):
        # test sequentially with manually created dfs
        df1 = pd.DataFrame(
            {
                "B": [0, 1, 2, np.nan, 4],
                "time": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        )
        df2 = pd.DataFrame(
            {
                "B": [0, 1, 2, -2, 4],
                "time": [
                    pd.Timestamp("20130101 09:00:01"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:04"),
                    pd.Timestamp("20130101 09:00:09"),
                ],
            }
        )
        wins = ("2s",)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(
                w, func_name
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            # XXX: skipping min/max for this test since the behavior of Pandas
            # is inconsistent: it assigns NaN to last output instead of 4!
            if func_name not in ("min", "max"):
                pd.testing.assert_frame_equal(
                    bodo_func(df1), test_impl(df1), check_column_type=False
                )
            pd.testing.assert_frame_equal(
                bodo_func(df2), test_impl(df2), check_column_type=False
            )

    def test_variable2(self):
        # test sequentially with generated dfs
        wins = ("2s",)
        sizes = (121,)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
            sizes = (1, 2, 10, 11, 121, 1000)
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(
                w, func_name
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for n in sizes:
                time = pd.date_range(start="1/1/2018", periods=n, freq="s")
                df = pd.DataFrame({"B": np.arange(n), "time": time})
                pd.testing.assert_frame_equal(
                    bodo_func(df), test_impl(df), check_column_type=False
                )

    def test_variable_apply1(self):
        # test sequentially with manually created dfs
        df1 = pd.DataFrame(
            {
                "B": [0, 1, 2, np.nan, 4],
                "time": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        )
        df2 = pd.DataFrame(
            {
                "B": [0, 1, 2, -2, 4],
                "time": [
                    pd.Timestamp("20130101 09:00:01"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:04"),
                    pd.Timestamp("20130101 09:00:09"),
                ],
            }
        )
        wins = ("2s",)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').apply(lambda a: a.sum(), raw=True)\n".format(
                w
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            pd.testing.assert_frame_equal(
                bodo_func(df1), test_impl(df1), check_column_type=False
            )
            pd.testing.assert_frame_equal(
                bodo_func(df2), test_impl(df2), check_column_type=False
            )

    def test_variable_apply2(self):
        # test sequentially with generated dfs
        wins = ("2s",)
        sizes = (121,)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
            # TODO: this crashes on Travis (3 process config) with size 1
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(
                w
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for n in sizes:
                time = pd.date_range(start="1/1/2018", periods=n, freq="s")
                df = pd.DataFrame({"B": np.arange(n), "time": time})
                pd.testing.assert_frame_equal(
                    bodo_func(df), test_impl(df), check_column_type=False
                )

    def test_variable_parallel1(self):
        wins = ("2s",)
        sizes = (121,)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
            # XXX: Pandas returns time = [np.nan] for size==1 for some reason
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(n):\n"
            func_text += "  df = pd.DataFrame({'B': np.arange(n), 'time': "
            func_text += "    pd.DatetimeIndex(np.arange(n) * 1000000000)})\n"
            func_text += "  res = df.rolling('{}', on='time').{}()\n".format(
                w, func_name
            )
            func_text += "  return res.B.sum()\n"
            loc_vars = {}
            exec(func_text, {"pd": pd, "np": np, "bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_variable_apply_parallel1(self):
        wins = ("2s",)
        sizes = (121,)
        if LONG_TEST:
            wins = ("1s", "2s", "3s", "4s")
            # XXX: Pandas returns time = [np.nan] for size==1 for some reason
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(n):\n"
            func_text += "  df = pd.DataFrame({'B': np.arange(n), 'time': "
            func_text += "    pd.DatetimeIndex(np.arange(n) * 1000000000)})\n"
            func_text += (
                "  res = df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(
                    w
                )
            )
            func_text += "  return res.B.sum()\n"
            loc_vars = {}
            exec(func_text, {"pd": pd, "np": np, "bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_fixed1(self):
        # test series rolling functions
        # all functions except apply
        S1 = pd.Series([0, 1, 2, np.nan, 4])
        S2 = pd.Series([0, 1, 2, -2, 4])
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)
        for func_name in test_funcs:
            func_text = f"def test_impl(S, w, c):\n  return S.rolling(w, center=c).{func_name}()\n"
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            for args in itertools.product(wins, centers):
                pd.testing.assert_series_equal(
                    bodo_func(S1, *args), test_impl(S1, *args)
                )
                pd.testing.assert_series_equal(
                    bodo_func(S2, *args), test_impl(S2, *args)
                )
        # test apply
        def apply_test_impl(S, w, c):
            return S.rolling(w, center=c).apply(lambda a: a.sum())

        bodo_func = bodo.jit(apply_test_impl)
        for args in itertools.product(wins, centers):
            pd.testing.assert_series_equal(
                bodo_func(S1, *args), apply_test_impl(S1, *args)
            )
            pd.testing.assert_series_equal(
                bodo_func(S2, *args), apply_test_impl(S2, *args)
            )

    def test_series_cov1(self):
        # test series rolling functions
        # all functions except apply
        S1 = pd.Series([0, 1, 2, np.nan, 4])
        S2 = pd.Series([0, 1, 2, -2, 4])
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        def test_impl(S, S2, w, c):
            return S.rolling(w, center=c).cov(S2)

        bodo_func = bodo.jit(test_impl)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(bodo_func(*args), test_impl(*args))
            pd.testing.assert_series_equal(bodo_func(*args), test_impl(*args))

        def test_impl2(S, S2, w, c):
            return S.rolling(w, center=c).corr(S2)

        bodo_func = bodo.jit(test_impl2)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(bodo_func(*args), test_impl2(*args))
            pd.testing.assert_series_equal(bodo_func(*args), test_impl2(*args))

    def test_df_cov1(self):
        # test series rolling functions
        # all functions except apply
        df1 = pd.DataFrame({"A": [0, 1, 2, np.nan, 4], "B": np.ones(5)})
        df2 = pd.DataFrame({"A": [0, 1, 2, -2, 4], "C": np.ones(5)})
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        def test_impl(df, df2, w, c):
            return df.rolling(w, center=c).cov(df2)

        bodo_func = bodo.jit(test_impl)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(
                bodo_func(*args), test_impl(*args), check_column_type=False
            )
            pd.testing.assert_frame_equal(
                bodo_func(*args), test_impl(*args), check_column_type=False
            )

        def test_impl2(df, df2, w, c):
            return df.rolling(w, center=c).corr(df2)

        bodo_func = bodo.jit(test_impl2)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(
                bodo_func(*args), test_impl2(*args), check_column_type=False
            )
            pd.testing.assert_frame_equal(
                bodo_func(*args), test_impl2(*args), check_column_type=False
            )


if __name__ == "__main__":
    unittest.main()
