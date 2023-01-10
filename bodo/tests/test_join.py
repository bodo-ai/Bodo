# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test join operations like df.merge(), df.join(), pd.merge_asof() ...
"""
import io
import os
import random
import string
import unittest
from decimal import Decimal

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    _gather_output,
    _get_dist_arg,
    _test_equal,
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    gen_nonascii_list,
    gen_random_decimal_array,
    gen_random_list_string_array,
    get_start_end,
)


# pytest fixture for df3 = df1.merge(df2, on="A") tests with nested arrays.
# These pairs of arrays are used to generate column "B" of df1 and df2 respectively
@pytest.fixture(
    params=[
        (
            np.array(
                [
                    [[[1, 2], [3]], [[2, None]]],
                    [[[3], [], [1, None, 4]]],
                    [[[3], [], [1, None, 4]]],
                    [[[4, 5, 6], []], [[1]], [[1, 2]]],
                    [],
                    [[[], [1]], None, [[1, 4]], []],
                ],
                object,
            ),
            np.array(
                [
                    [[[6, 8, 3], [1]], [[0, None]]],
                    [[[30], [1], [], [10, None, 4]]],
                    [[[1, 2, 3, 4], [1]], [[]], []],
                    [[[1, 2, 3, 4], [1]], [[]], []],
                    [[[1]], None, [[500, 4]], [[33]]],
                    [],
                ],
                object,
            ),
        ),
        # TODO None value fields in structs is not supported in typing.
        # Using -1 instead of None. Can change to None in the future
        pytest.param(
            (
                np.array(
                    [
                        [{"A": 1, "B": 2}, {"A": 10, "B": 20}],
                        [{"A": 3, "B": -1}],
                        [{"A": 5, "B": 6}, {"A": 50, "B": 60}, {"A": 500, "B": 600}],
                        [{"A": 10, "B": 20}, {"A": 100, "B": 200}],
                        [{"A": 30, "B": 40}],
                        [
                            {"A": -1, "B": 60},
                            {"A": 500, "B": 600},
                            {"A": 5000, "B": 6000},
                        ],
                    ],
                    object,
                ),
                np.array(
                    [
                        [{"A": 3, "B": 45}],
                        [
                            {"A": -1, "B": 60},
                            {"A": -1, "B": 60},
                            {"A": 3, "B": 3},
                            {"A": -7, "B": 13},
                        ],
                        [{"A": 11, "B": 60}, {"A": 500, "B": 33}, {"A": 55, "B": 57}],
                        [{"A": 10, "B": 20}, {"A": 5, "B": -1}],
                        [{"A": 3, "B": 4}, {"A": 10, "B": 20}],
                        [{"A": 1, "B": 7}],
                    ],
                    object,
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"A": 1.0, "B": True},
                        {"A": 3.0, "B": False},
                        {"A": 5.0, "B": True},
                        {"A": 10.0, "B": True},
                        {"A": 30.0, "B": False},
                        {"A": -1.0, "B": True},
                    ]
                ),
                np.array(
                    [
                        {"A": 3.0, "B": False},
                        {"A": -1.0, "B": True},
                        {"A": 11.0, "B": False},
                        {"A": 10.0, "B": False},
                        {"A": 3.0, "B": False},
                        {"A": 1.0, "B": True},
                    ]
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"A": [1, 2], "B": [2, 3]},
                        {"A": [3, 5, 7], "B": [-1]},
                        {"A": [5], "B": [6, 9, 12, 15]},
                        {"A": [10, 13], "B": [20]},
                        {"A": [30, 40], "B": [40, 50, 51, 52, 53]},
                        {"A": [-1, -2, -3], "B": [60]},
                    ]
                ),
                np.array(
                    [
                        {"A": [3, 2, 1, 0], "B": [45, 46]},
                        {"A": [-1, -10, 20], "B": [60, 70, 80]},
                        {"A": [11, 7, 4], "B": [33, 2]},
                        {"A": [10], "B": [20]},
                        {"A": [3, 5], "B": [4]},
                        {"A": [1], "B": [7, 8]},
                    ]
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"A": [Decimal("1.0"), Decimal("2.2")], "B": [Decimal("4.14")]},
                        {
                            "A": [Decimal("0"), Decimal("3.2"), Decimal("4")],
                            "B": [Decimal("-1")],
                        },
                        {
                            "A": [Decimal("5")],
                            "B": [Decimal("644"), Decimal("9.1"), Decimal("154")],
                        },
                        {
                            "A": [Decimal("10.0"), Decimal("13.4")],
                            "B": [Decimal("3.14159")],
                        },
                        {
                            "A": [Decimal("30"), Decimal("5.2")],
                            "B": [Decimal("0"), Decimal("2")],
                        },
                        {
                            "A": [Decimal("-1"), None, Decimal("-3")],
                            "B": [Decimal("60")],
                        },
                    ]
                ),
                np.array(
                    [
                        {
                            "A": [Decimal("2.78"), Decimal("3"), Decimal("2")],
                            "B": [Decimal("45")],
                        },
                        {"A": [Decimal("-1"), Decimal("-10")], "B": [Decimal("4568")]},
                        {
                            "A": [Decimal("11"), None, Decimal("-44.7")],
                            "B": [Decimal("33")],
                        },
                        {"A": [Decimal("10.4")], "B": [Decimal("20.6")]},
                        {"A": [Decimal("3.5"), Decimal("5.5")], "B": [Decimal("4")]},
                        {"A": [Decimal("1")], "B": [Decimal("7"), None]},
                    ]
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"A": {"A1": 10, "A2": 2}, "B": {"B1": -11, "B2": 4}},
                        {"A": {"A1": -19, "A2": 5}, "B": {"B1": 5, "B2": 19}},
                        {"A": {"A1": -5, "A2": -9}, "B": {"B1": -15, "B2": 13}},
                        {"A": {"A1": -12, "A2": 2}, "B": {"B1": 14, "B2": 2}},
                        {"A": {"A1": 17, "A2": -12}, "B": {"B1": 14, "B2": -18}},
                        {"A": {"A1": 17, "A2": 10}, "B": {"B1": -13, "B2": 18}},
                    ]
                ),
                np.array(
                    [
                        {"A": {"A1": 11, "A2": -1}, "B": {"B1": -10, "B2": -6}},
                        {"A": {"A1": 12, "A2": 17}, "B": {"B1": 12, "B2": -1}},
                        {"A": {"A1": 17, "A2": 2}, "B": {"B1": 19, "B2": -2}},
                        {"A": {"A1": 17, "A2": 16}, "B": {"B1": 6, "B2": -20}},
                        {"A": {"A1": 4, "A2": -12}, "B": {"B1": 10, "B2": 3}},
                        {"A": {"A1": 19, "A2": 11}, "B": {"B1": 3, "B2": 16}},
                    ]
                ),
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def nested_arrays_value(request):
    return request.param


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


def _gen_df_binary(n, seed=None):
    """
    helper function that generate DataFrame with int and binary columns.
    Takes an optional seed argument for use with np.random.seed.
    """
    if seed == None:
        np.random.seed(14)
    else:
        np.random.seed(seed)

    bin_vals = []
    for _ in range(n):
        # store NA with 30% chance
        if np.random.ranf() < 0.3:
            bin_vals.append(np.nan)
            continue

        bin_vals.append(bytes(np.random.randint(1, 100)))

    A = np.random.randint(0, 100, n)
    df = pd.DataFrame({"A": A, "B": bin_vals})
    return df


# ------------------------------ merge() ------------------------------ #
def test_merge_nonascii_values(memory_leak_check):
    """
    Test merge(): make sure merge works with non-ASCII key and data values
    """

    def test_impl1(df1, df2):
        o1 = df1.merge(df2, on=["key"])
        return o1

    def test_impl2(df1, df2):
        o1 = df1.merge(df2, on=["value"])
        return o1

    def test_impl3(df1, df2):
        o1 = df1.merge(df2, on=["key"], how="outer")
        return o1

    df1 = pd.DataFrame({"key": gen_nonascii_list(5), "value": [1, 2, 3, 4, 5]})

    df2 = pd.DataFrame(
        {"key": ["a", "bb", "ccc", "DDDD", "e"], "value": [6, 7, 8, 9, 10]}
    )

    df3 = pd.DataFrame({"key": gen_nonascii_list(5), "value": [6, 7, 8, 9, 10]})

    pairs = ((df1, df2), (df1, df3), (df2, df3))
    for left, right in pairs:
        check_func(test_impl1, (left, right), sort_output=True, reset_index=True)
        check_func(test_impl2, (left, right), sort_output=True, reset_index=True)
        check_func(test_impl3, (left, right), sort_output=True, reset_index=True)


def test_merge_key_change(memory_leak_check):
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
        bodo_func(df1, df2, df3, df4)[0],
        test_impl(df1, df2, df3, df4)[0],
        check_column_type=False,
    )
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2, df3, df4)[1],
        test_impl(df1, df2, df3, df4)[1],
        check_column_type=False,
    )


@pytest.mark.slow
def test_merge_suffixes_bracket(memory_leak_check):
    """
    Test merge(): test the suffixes functionality for inner/left/right/outer with bracket
    """

    def test_impl1(df1, df2):
        o1 = df1.merge(df2, on="key", how="inner", suffixes=["_a", "_b"])
        return o1

    def test_impl2(df1, df2):
        df3 = df1.merge(df2, on="key", how="left", suffixes=["_a", "_b"])
        return df3

    def test_impl3(df1, df2):
        df3 = df1.merge(df2, on="key", how="right", suffixes=["_a", "_b"])
        return df3

    def test_impl4(df1, df2):
        o1 = df1.merge(df2, on="key", how="outer", suffixes=["_a", "_b"])
        return o1

    df1 = pd.DataFrame({"key": [0, 1, 2, 0], "value": [1, 2, 3, 4]})
    df2 = pd.DataFrame({"key": [0, 1, 2, 0], "value": [5, 6, 7, 8]})
    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(
        test_impl2, (df1, df2), sort_output=True, check_dtype=False, reset_index=True
    )
    check_func(
        test_impl3, (df1, df2), sort_output=True, check_dtype=False, reset_index=True
    )
    check_func(
        test_impl4, (df1, df2), sort_output=True, check_dtype=False, reset_index=True
    )


def test_merge_suffixes_parenthesis(memory_leak_check):
    """
    Test merge(): test the suffixes functionality with parenthesis
    """

    def test_impl(df1, df2):
        o1 = df1.merge(df2, on=3, how="inner", suffixes=("_a", "_b"))
        return o1

    df1 = pd.DataFrame({3: [0, 1, 2, 0], "value": [1, 2, 3, 5]})
    df2 = pd.DataFrame({3: [0, 1, 2, 0], "value": [5, 6, 7, 8]})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_join_datetime(memory_leak_check):
    """
    merging with some missing datetime and date.
    ---Datetime requires its own handling.
    ---Date can be considered as array of int64 without problems.
    """

    def test_impl(df1, df2):
        df3 = df2.merge(df1, how="left")
        return df3

    siz = 10
    datearr_1 = pd.date_range("2017-01-02", periods=siz)
    datearr_2 = pd.date_range("2015-01-02", periods=siz)
    timedelta_arr = datearr_1 - datearr_2
    df1_timedelta = pd.DataFrame({"A": 1 + np.arange(siz), "B": timedelta_arr})
    df1_datetime = pd.DataFrame({"A": 1 + np.arange(siz), "B": datearr_1})
    df1_date = pd.DataFrame({"A": 1 + np.arange(siz), "B": datearr_1.date})
    df2 = pd.DataFrame({"A": siz - 5 + np.arange(siz)})

    check_func(test_impl, (df1_timedelta, df2), sort_output=True, reset_index=True)
    check_func(test_impl, (df1_datetime, df2), sort_output=True, reset_index=True)
    check_func(test_impl, (df1_date, df2), sort_output=True, reset_index=True)


def test_merge_decimal(memory_leak_check):
    def f(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    random.seed(5)
    n = 50
    df1 = pd.DataFrame(
        {
            "A": gen_random_decimal_array(1, n),
            "B": gen_random_decimal_array(1, n),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": gen_random_decimal_array(1, n),
            "D": gen_random_decimal_array(1, n),
        }
    )
    check_func(f, (df1, df2), sort_output=True, reset_index=True)


def test_merge_empty_suffix_keys(memory_leak_check):
    """
    Test merge(): merging on keys and having an empty sufix.
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_on="A", right_on="C", suffixes=("", "_t"))
        return df3

    def f2(df1, df2):
        df3 = df1.merge(df2, left_on="A", right_on="C", suffixes=("", "_t"))
        return df3[["A", "C"]]

    def f3(df1, df2):
        df3 = df1.merge(df2, left_on="A", right_on="C", suffixes=("", "_t"))
        return df3[["A_t", "B"]]

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
            "C": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
        }
    )
    check_func(f1, (df1, df2), sort_output=True, reset_index=True)
    check_func(f2, (df1, df2), sort_output=True, reset_index=True)
    check_func(f3, (df1, df2), sort_output=True, reset_index=True)

    merge_func1 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f1)
    merge_func2 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f2)
    merge_func3 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f3)

    # calling the functions to get function IR
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func1(df1, df2)
        check_logger_msg(stream, "Output columns: ['A', 'C', 'A_t', 'B', 'C_t']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func2(df1, df2)
        check_logger_msg(stream, "Output columns: ['A', 'C']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func3(df1, df2)
        check_logger_msg(stream, "Output columns: ['A_t', 'B']")
    # Verify the index is dead.
    func_sigs = [merge_func1, merge_func2, merge_func3]
    for func_sig in func_sigs:
        confirmed_dead_index = False
        fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
        for block in fir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, bodo.ir.join.Join):
                    confirmed_dead_index = not stmt.has_live_out_index_var
        assert confirmed_dead_index, "Index not confirmed dead in join node"


def test_merge_left_right_index(memory_leak_check):
    """
    Test merge(): merging on index and use of suffices on output.
    """

    def f(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
        }
    )

    check_func(f, (df1, df2), sort_output=True)


def test_merge_left_index_dce(memory_leak_check):
    """
    Test merge(): merging on left_index and only returning
    one of the output Series.
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3[["A", "B"]]

    def f2(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3[["A_x", "B"]]

    def f3(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3[["A_y", "B"]]

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
        }
    )

    check_func(f1, (df1, df2), sort_output=True)
    check_func(f2, (df1, df2), sort_output=True)
    check_func(f3, (df1, df2), sort_output=True)

    merge_func1 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f1)
    merge_func2 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f2)
    merge_func3 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f3)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func1(df1, df2)
        check_logger_msg(stream, "Output columns: ['A', 'B']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func2(df1, df2)
        check_logger_msg(stream, "Output columns: ['A_x', 'B']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func3(df1, df2)
        check_logger_msg(stream, "Output columns: ['A_y', 'B']")
    # Verify the index is alive.
    func_sigs = [merge_func1, merge_func2, merge_func3]
    for func_sig in func_sigs:
        confirmed_live_index = False
        fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
        for block in fir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, bodo.ir.join.Join):
                    confirmed_live_index = stmt.has_live_out_index_var
        assert confirmed_live_index, "Index not confirmed alive in join node"


def test_merge_right_index_dce(memory_leak_check):
    """
    Test merge(): merging on right_index and only returning
    one of the output Series.
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, right_index=True, left_on=["A"])
        return df3[["A", "B"]]

    def f2(df1, df2):
        df3 = df1.merge(df2, right_index=True, left_on=["A"])
        return df3[["A_x", "B"]]

    def f3(df1, df2):
        df3 = df1.merge(df2, right_index=True, left_on=["A"])
        return df3[["A_y", "B"]]

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
        }
    )

    check_func(f1, (df1, df2), sort_output=True)
    check_func(f2, (df1, df2), sort_output=True)
    check_func(f3, (df1, df2), sort_output=True)

    merge_func1 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f1)
    merge_func2 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f2)
    merge_func3 = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f3)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func1(df1, df2)
        check_logger_msg(stream, "Output columns: ['A', 'B']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func2(df1, df2)
        check_logger_msg(stream, "Output columns: ['A_x', 'B']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func3(df1, df2)
        check_logger_msg(stream, "Output columns: ['A_y', 'B']")
    # Verify the index is alive.
    func_sigs = [merge_func1, merge_func2, merge_func3]
    for func_sig in func_sigs:
        confirmed_live_index = False
        fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
        for block in fir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, bodo.ir.join.Join):
                    confirmed_live_index = stmt.has_live_out_index_var
        assert confirmed_live_index, "Index not confirmed alive in join node"


def test_merge_left_right_index_dce(memory_leak_check):
    """
    Test merge(): merging on index and only returning
    one of the output Series.
    """

    def f(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3["B"].values

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
        }
    )

    check_func(f, (df1, df2), sort_output=True)

    merge_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func(df1, df2)
        check_logger_msg(stream, "Output columns: ['B']")
    # Verify the index is dead.
    func_sig = merge_func
    confirmed_dead_index = False
    fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.join.Join):
                confirmed_dead_index = not stmt.has_live_out_index_var
    assert confirmed_dead_index, "Index not confirmed dead in join node"


def test_merge_left_right_only_index(memory_leak_check):
    """
    Test merge(): merging on index and only returning
    the output index.
    """

    def f(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3.index

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 11, 3, 8] * 2,
            "C": ["a", "b", "A", "C", "CC", "A", "LL", "D"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 4, 3, 5, 11, 8, 3] * 2,
            "B": ["c", "d", "VV", "DD", "", "D", "SS", "A"] * 2,
        }
    )

    check_func(f, (df1, df2), sort_output=True)

    merge_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(f)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func(df1, df2)
        check_logger_msg(stream, "Output columns: []")
    # Verify the index is alive and the table is dead.
    func_sig = merge_func
    confirmed_live_index = False
    confirmed_dead_out_table = False
    confirmed_dead_left_table = False
    confirmed_dead_right_table = False
    fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.join.Join):
                confirmed_live_index = stmt.has_live_out_index_var
                confirmed_dead_out_table = not stmt.has_live_out_table_var
                confirmed_dead_left_table = not stmt.has_live_left_table_var
                confirmed_dead_right_table = not stmt.has_live_right_table_var
    assert confirmed_live_index, "Index not confirmed alive in join node"
    assert confirmed_dead_out_table, "Output Table not confirmed dead in join node"
    assert confirmed_dead_left_table, "Left Table not confirmed dead in join node"
    assert confirmed_dead_right_table, "Right Table not confirmed dead in join node"


def test_list_string_array_type_specific(memory_leak_check):
    """Test with the column type of type  list_string_array"""

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on="A", how="outer")
        return df3

    df1 = pd.DataFrame({"A": [["AB"], np.nan, ["A", "B", "C"]], "C": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [["A", "B", "C"], ["AB", "EF"], np.nan], "D": [4, 5, 6]})
    bodo_impl = bodo.jit(test_impl)
    df3_bodo = bodo_impl(df1, df2)
    df3_target = pd.DataFrame(
        {
            "A": [["AB"], np.nan, ["A", "B", "C"], ["AB", "EF"]],
            "C": [1, 2, 3, np.nan],
            "D": [np.nan, 6, 4, 5],
        }
    )
    pd.testing.assert_frame_equal(
        df3_bodo.reset_index(drop=True),
        df3_target.reset_index(drop=True),
        check_dtype=False,
        check_column_type=False,
    )


@pytest.mark.slow
def test_list_string_array_type_random(memory_leak_check):
    def test_impl(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    random.seed(5)
    n1 = 50
    n2 = 100
    df1 = pd.DataFrame(
        {
            "A": gen_random_list_string_array(1, n1),
            "C": gen_random_list_string_array(1, n1),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": gen_random_list_string_array(1, n2),
            "D": gen_random_list_string_array(1, n2),
        }
    )
    check_func(
        test_impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        convert_columns_to_pandas=True,
    )


def test_merge_left_right_nontrivial_index(memory_leak_check):
    """
    Test merge(): merging on non-trivial index and use of suffices on output.
    """

    def f(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3

    df1 = pd.DataFrame({"A": [1, 2], "C": [3.2, 1.2]}, index=[3, 4])
    df2 = pd.DataFrame({"A": [1, 2], "B": [1.1, 3.1]}, index=[4, 5])

    check_func(f, (df1, df2), sort_output=True)


@pytest.mark.slow
def test_merge_empty_suffix_underscore(memory_leak_check):
    """
    Test merge(): test the suffixes functionality with a pathological example
    """

    def test_impl(df1, df2):
        o1 = df1.merge(df2, on="key", how="inner", suffixes=["", "_"])
        return o1

    df1 = pd.DataFrame({"key": [0, 1, 2, 0], "value": [1, 2, 3, 5]})
    df2 = pd.DataFrame({"key": [0, 1, 2, 0], "value": [5, 6, 7, 8]})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3]}),
        pytest.param(
            pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}),
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"A": [-1, 1, 3]}),
        pytest.param(
            pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1]}), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.DataFrame({"A": [-1, 1, 3], "B": [-1, 0, 1], "C": [-11, 0, 4]}),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_merge_common_cols(df1, df2, memory_leak_check):
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
        check_column_type=False,
    )


def test_merge_cross(memory_leak_check, df_value):
    """
    Test merge() with how="cross" with various data types and values
    """

    def test_impl1(df1, df2):
        return df1.merge(df2, how="cross")

    rng = np.random.default_rng(100)
    df2 = pd.DataFrame({"RAND": rng.integers(11, 33, 6)})
    check_func(test_impl1, (df_value, df2), sort_output=True, reset_index=True)
    check_func(test_impl1, (df2, df_value), sort_output=True, reset_index=True)


def test_merge_cross_len_only(memory_leak_check):
    """
    Test merge() with how="cross" with only length of output used (corner case)
    """

    def impl(df1, df2):
        return len(df1.merge(df2, how="cross"))

    df1 = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 3, 4, 11]})
    df2 = pd.DataFrame({"C": [3, 4, 5], "D": [6, 7, 8]})
    check_func(impl, (df1, df2))

    # test only one side being distributed
    py_out = len(df1) * len(df2)
    bodo_out = bodo.jit(distributed=["df1"])(impl)(_get_dist_arg(df1, True, True), df2)
    assert bodo_out == py_out
    bodo_out = bodo.jit(distributed=["df2"])(impl)(df1, _get_dist_arg(df2, True, True))
    assert bodo_out == py_out


def test_merge_cross_dead_input(memory_leak_check):
    """
    Test merge() with how="cross" when all columns from one side are dead
    """

    # left side is dead
    def impl1(df1, df2):
        df3 = df1.merge(df2, how="cross")
        return df3[["D"]]

    # right side is dead
    def impl2(df1, df2):
        df3 = df1.merge(df2, how="cross")
        return df3[["A"]]

    df1 = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 3, 4, 11]})
    df2 = pd.DataFrame({"C": [3, 4, 5], "D": [6, 7, 8]})
    check_func(impl1, (df1, df2), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True, check_dtype=False)

    # test only one side being distributed
    out_df = _gather_output(
        bodo.jit(distributed=["df1"])(impl1)(_get_dist_arg(df1, True, True), df2)
    )
    if bodo.get_rank() == 0:
        _test_equal(
            out_df,
            impl1(df1, df2),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
        )
    out_df = _gather_output(
        bodo.jit(distributed=["df2"])(impl2)(df1, _get_dist_arg(df2, True, True))
    )
    if bodo.get_rank() == 0:
        _test_equal(
            out_df,
            impl2(df1, df2),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
        )


@pytest.mark.slow
def test_merge_suffix_included(memory_leak_check):
    """
    Merge between two data columns that would conflict
    if a column incorrectly tried to add a suffix.
    """
    df_right = pd.DataFrame(
        {
            "key": np.arange(20),
            "name_x": np.arange(20),
        }
    )
    df_left = pd.DataFrame(
        {
            "key": np.arange(20),
            "name_y": np.arange(20),
        }
    )
    df = pd.DataFrame(
        {
            "key": np.arange(20),
            "name": ["fwfwe", "#424"] * 10,
        }
    )

    def impl(df1, df2):
        return df1.merge(df2, on="key")

    # Make sure name doesn't check for name_x
    check_func(impl, (df, df_right), sort_output=True, reset_index=True)
    # Make sure name doesn't check for name_y
    check_func(impl, (df_left, df), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_suffix_collision(memory_leak_check):
    """
    Merge between two data columns where a suffix
    conflicts with the original columns.
    """
    df_left = pd.DataFrame(
        {
            "key": np.arange(20),
            "name": np.arange(20),
            "name_x": np.arange(20),
            "name_y": np.arange(20),
        }
    )
    df_right = pd.DataFrame(
        {
            "key": np.arange(20),
            "name": ["fwfwe", "#424"] * 10,
            "name_x": ["fwfwe", "#424"] * 10,
            "name_y": ["fwfwe", "#424"] * 10,
        }
    )

    def impl(df1, df2):
        return df1.merge(df2, on="key")

    check_func(impl, (df_left, df_right), sort_output=True, reset_index=True)


def test_merge_disjoint_keys1(memory_leak_check):
    """
    Test merge(): 'how' = inner on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, left_on=["a", "b"], right_on=["a", "d"], how="left")
        return df3

    df1 = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 4.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0], "d": [2.0, 3.0]})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_disjoint_keys2(memory_leak_check):
    """
    Test merge(): 'how' = inner on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, left_on=["a", "b"], right_on=["c", "d"], how="left")
        return df3

    df1 = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 4.0]})
    df2 = pd.DataFrame({"c": [1.0, 2.0], "d": [2.0, 3.0]})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.smoke
def test_merge_inner(memory_leak_check):
    """
    Test merge(): 'how' = inner on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="inner", on="key")
        return df3

    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)}
    )
    df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_left1(memory_leak_check):
    """
    Test merge(): 'how' = left on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="left", on="key")
        return df3

    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)}
    )
    df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_left2(memory_leak_check):
    """
    Test merge(): 'how' = left on specified integer column
    where a key is repeated on left but not right side
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="left", on="key")
        return df3

    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 3, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)}
    )
    df2 = pd.DataFrame({"key": [1, 2, 9, 3, 10], "B": np.array([1, 7, 2, 6, 5], float)})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_right(memory_leak_check):
    """
    Test merge(): 'how' = right on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="right", on="key")
        return df3

    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)}
    )
    df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_outer(memory_leak_check):
    """
    Test merge(): 'how' = outer on specified integer column
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, how="outer", on="key")
        return df3

    df1 = pd.DataFrame(
        {"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)}
    )
    df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_overlap(memory_leak_check):
    """
    Test merge(): column overlapping behavior
    """

    def test_impl(df1):
        df3 = df1.merge(df1, on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [3, 1, 1, 4], "B": [1, 2, 3, 2], "C": [7, 8, 9, 4]})
    check_func(test_impl, (df1,), sort_output=True, reset_index=True)


@pytest.mark.slow
@pytest.mark.parametrize("n", [11, 11111])
def test_merge_int_key(n, memory_leak_check):
    """
    Test merge(): key column is of type int
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, left_on="key1", right_on="key2")
        return df3

    df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
    df2 = pd.DataFrame({"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "n1, n2, len_siz",
    [
        pytest.param(5, 4, 2, marks=pytest.mark.slow),
        pytest.param(10, 12, 3, marks=pytest.mark.slow),
        (120, 100, 10),
        pytest.param(40, 30, 7, marks=pytest.mark.slow),
        pytest.param(1000, 900, 10, marks=pytest.mark.slow),
    ],
)
def test_merge_nullable_int_bool(n1, n2, len_siz, memory_leak_check):
    """
    Test merge(): test of nullable_int_bool for inner/left/right/outer and random input
    """

    def test_impl1(df1, df2):
        df3 = df1.merge(df2, on="A", how="inner")
        return df3

    def test_impl2(df1, df2):
        df3 = df1.merge(df2, on="A", how="left")
        return df3

    def test_impl3(df1, df2):
        df3 = df1.merge(df2, on="A", how="right")
        return df3

    def test_impl4(df1, df2):
        df3 = df1.merge(df2, on="A", how="outer")
        return df3

    def get_random_column(n, len_siz):
        elist = []
        for _ in range(n):
            prob = random.randint(1, len_siz)
            if prob == 1:
                elist.append(None)
            else:
                elist.append(prob)
        return pd.array(elist, dtype="UInt16")

    def get_random_dataframe(n, len_siz):
        elist1 = get_random_column(n, len_siz)
        elist2 = get_random_column(n, len_siz)
        return pd.DataFrame({"A": elist1, "B": elist2})

    random.seed(5)
    df1 = get_random_dataframe(n1, len_siz)
    df2 = get_random_dataframe(n2, len_siz)
    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl3, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl4, (df1, df2), sort_output=True, reset_index=True)


def test_merge_multi_int_key(memory_leak_check):
    """
    Test merge(): sequentially merge on more than one integer key columns
    """

    def test_impl1(df1, df2):
        df3 = df1.merge(df2, on=["A", "B"])
        return df3

    # test multiple column names passed as a tuple
    def test_impl2(df1, df2):
        df3 = df1.merge(df2, on=("A", "B"))
        return df3

    # test constant list inference for join keys
    def test_impl3(df1, df2):
        df3 = df1.merge(df2, on=list(set(df1.columns) - set(["C"])))
        return df3

    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
    )

    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl3, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_literal_arg(memory_leak_check):
    """
    Test forcing merge() args to be literals if jit arguments
    """

    # 'on' and 'suffixes'
    def test_impl1(df1, df2, on, suffixes):
        return df1.merge(df2, on=on, suffixes=suffixes)

    def test_impl2(df1, df2, left_on, right_index, how):
        return df1.merge(df2, how, left_on=left_on, right_index=right_index)

    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
    )

    check_func(
        test_impl1,
        (df1, df2.rename({"D": "C"}, axis=1), ["A", "B"], ["", "_"]),
        sort_output=True,
        reset_index=True,
    )
    # TODO: enable test when #1013 is resolved
    # check_func(test_impl2, (df1, df2.set_index("A"), "A", True, "inner"), sort_output=True)


def test_merge_right_index_rm_dead(memory_leak_check):
    """test joins with an index (rm dead bug reported by user)"""

    def cal_average_by_payment_type(df):
        payments = pd.DataFrame(
            {
                "payment_name": [
                    "Credit Card",
                    "Cash",
                    "No Charge",
                    "Dispute",
                    "Unknown",
                    "Voided trip",
                ]
            },
            index=[1, 2, 3, 4, 5, 6],
        )
        df2 = df.merge(payments, left_on="payment_type", right_index=True)
        result = df2.groupby("payment_name").tip_amount.mean()
        return result

    df = pd.DataFrame({"payment_type": [3, 4, 1], "tip_amount": [1.1, 2.2, 3.3]})
    check_func(
        cal_average_by_payment_type,
        (df,),
        sort_output=True,
    )


def test_merge_right_key_nullable(memory_leak_check):
    """Tests a bug where converting right key to nullable in left join output would
    throw an error.
    """

    def impl(df1, df2):
        return df1.merge(df2, left_on="A1", right_on="A2", how="left")

    df1 = pd.DataFrame({"A1": [1, 2, 4]})
    df2 = pd.DataFrame({"A2": [1, 2, 3]})
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        check_dtype=False,
        reset_index=True,
    )


def test_merge_bool_to_nullable(memory_leak_check):
    """Tests converting non-nullable bool input to nullable bool in output"""

    def impl(df1):
        n = len(df1)
        df2 = pd.DataFrame({"A": np.arange(n), "B": np.ones(n, np.bool_)})
        return df1.merge(df2, on="A", how="left")

    df1 = pd.DataFrame({"A": [1, 2, 4]})
    check_func(
        impl,
        (df1,),
        sort_output=True,
        check_dtype=False,
        # no dist test since we are creating a dataframe inside the function
        dist_test=False,
    )


def test_merge_key_type_change(memory_leak_check):
    """
    Test merge() key type check when key type changes in the program (handled in partial
    typing pass)
    """

    def test_impl():
        df1 = pd.DataFrame(
            {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
        )
        df2 = pd.DataFrame(
            {"A": ["2", "1", "4", "4", "3"], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
        )
        df1["A"] = df1["A"].astype(str)
        df3 = df1.merge(df2, on="A")
        return df3

    check_func(
        test_impl, (), sort_output=True, is_out_distributed=False, reset_index=True
    )


def test_merge_schema_change(memory_leak_check):
    """
    Test merge() key check when schema changes in the program (handled in partial
    typing pass)
    """

    def test_impl():
        df1 = pd.DataFrame({"B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]})
        df2 = pd.DataFrame(
            {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
        )
        df1["A"] = [3, 1, 1, 3, 4]
        df3 = df1.merge(df2, on="A")
        return df3

    check_func(
        test_impl, (), sort_output=True, is_out_distributed=False, reset_index=True
    )


def test_merge_str_key(memory_leak_check):
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


def test_merge_str_nan1(memory_leak_check):
    """
    test merging dataframes containing string columns with nan values
    """

    def test_impl(df1, df2):
        return pd.merge(df1, df2, left_on="key1", right_on="key2")

    df1 = pd.DataFrame(
        {
            "key1": ["foo", "bar", "baz", "baz", "c4", "c7"],
            "A": ["b", "", "ss", "a", "b2", np.nan],
        }
    )
    df2 = pd.DataFrame(
        {
            "key2": ["baz", "bar", "baz", "foo", "c4", "c7"],
            "B": ["b", np.nan, "", "AA", "c", "a1"],
        }
    )

    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_str_nan2(memory_leak_check):
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
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_binary_key(memory_leak_check):
    """
    Test merge(): sequentially merge on key column of type binary
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, left_on="key1", right_on="key2")
        return df3

    df1 = pd.DataFrame({"key1": [b"foo", b"bar", b"baz"] * 3})
    df2 = pd.DataFrame(
        {"key2": [b"baz", b"bar", b"baz"] * 3, "B": [b"b", b"zzz", b"ss"] * 3}
    )

    bodo_func = bodo.jit(test_impl)
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_binary_nan1(memory_leak_check):
    """
    test merging dataframes containing binary columns with nan values
    """

    def test_impl(df1, df2):
        return pd.merge(df1, df2, left_on="key1", right_on="key2")

    df1 = pd.DataFrame(
        {
            "key1": [b"foo", b"bar", b"baz", b"baz", b"c4", b"c7"],
            "A": [b"b", b"", b"ss", b"a", b"b2", np.nan],
        }
    )
    df2 = pd.DataFrame(
        {
            "key2": [b"baz", b"bar", b"baz", b"foo", b"c4", b"c7"],
            "B": [b"b", np.nan, b"", b"AA", b"c", b"a1"],
        }
    )

    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_binary_nan2(memory_leak_check):
    """
    test merging dataframes containing binary columns with nan values
    on larger dataframes
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on="A")

    # seeds should be the same on different processors for consistent input
    n = 1211
    df1 = _gen_df_binary(n, 3)
    df2 = _gen_df_binary(n, 13)
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_bool_nan(memory_leak_check):
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
    check_func(
        test_impl, (df1, df2), sort_output=True, check_dtype=False, reset_index=True
    )


def test_merge_nontrivial_index(memory_leak_check):
    """
    Test merge(): merging on columns with dataframe having non-trivial indexes.
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, left_on="key1", right_on="key2", how="left")
        return df3

    df1 = pd.DataFrame(
        {"key1": ["foo", "c7", "bar", "baz", "c11"]}, index=[0.0, 1.1, 3.0, 5.0, 2.2]
    )
    df2 = pd.DataFrame(
        {"key2": ["baz", "c7", "bar", "baz", "c11"]}, index=["a", "d", "bb", "ccc", "1"]
    )

    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_out_str_na(memory_leak_check):
    """
    Test merge(): setting NA in output string data column.
    Sorting has to be done in the code since option sort_output=True does not
    work for series.
    reset_index(drop=True) is needed so as to avoid index collisions.
    Size of the test has been extended so that empty bins do not occur for test on 2 or 3 processors.
    """

    def test_impl1(df1, df2):
        df4 = (
            df1.merge(df2, left_on="key1", right_on="key2", how="inner")
            .sort_values(by="B")
            .reset_index(drop=True)
        )
        return df4.B

    def test_impl2(df1, df2):
        df4 = (
            df1.merge(df2, left_on="key1", right_on="key2", how="inner")
            .sort_values(by="B")
            .reset_index(drop=True)
        )
        return df4

    df1 = pd.DataFrame({"key1": ["foo", "bar", "baz", "fab", "faz", "fay"]})
    df2 = pd.DataFrame(
        {
            "key2": ["baz", "bar", "baz", "faq", "fau", "fab"],
            "B": ["b", "zzz", "ss", "aaa", "bb", "wuk"],
        }
    )

    check_func(test_impl1, (df1, df2), check_typing_issues=False)
    check_func(test_impl2, (df1, df2), check_typing_issues=False)


def test_merge_out_binary_na(memory_leak_check):
    """
    Test merge(): setting NA in output binary data column.
    Sorting has to be done in the code since option sort_output=True does not
    work for series.
    reset_index(drop=True) is needed so as to avoid index collisions.
    Size of the test has been extended so that empty bins do not occur for test on 2 or 3 processors.
    """

    def test_impl(df1, df2):
        df4 = df1.merge(df2, left_on="key1", right_on="key2", how="inner")
        return df4

    df1 = pd.DataFrame({"key1": [b"foo", b"bar", b"baz", b"fab", b"faz", b"fay"] * 2})
    df2 = pd.DataFrame(
        {
            "key2": [b"baz", b"bar", b"baz", b"faq", b"fau", b"fab"] * 2,
            "B": ["b", "zzz", "ss", "aaa", "bb", "wuk"] * 2,
        }
    )

    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_datetime(memory_leak_check):
    """
    Test merge(): merge on key column of type DatetimeIndex
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, on="time")
        return df3

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
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_datetime_parallel(memory_leak_check):
    """
    Test merge(): merge on key column of type DatetimeIndex
    ensure parallelism
    """

    def test_impl(df1, df2):
        df3 = pd.merge(df1, df2, on="time")
        return (df3.A.sum(), df3.time.max(), df3.B.sum())

    bodo_func = bodo.jit(distributed_block=["df1", "df2"])(test_impl)
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


@pytest.mark.slow
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
def test_merge_suffix(df1, df2, memory_leak_check):
    """
    test merge() default behavior:
    column name overlaps, require adding suffix('_x', '_y') to column names
    """

    def impl1(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    def impl2(df1, df2):
        df3 = df1.merge(df2, on=["B", "A"])
        return df3

    check_func(impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
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
def test_merge_index1(df1, df2, memory_leak_check):
    """
    test merge(): with left_index and right_index specified, merge using index
    """

    def impl1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3

    check_func(impl1, (df1, df2), sort_output=True)


def test_merge_index_left(memory_leak_check):
    """
    test merge(): with left_index and right_index specified
    with all on = left
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True, how="left")
        return df3

    df1 = pd.DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
    df2 = pd.DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])

    check_func(impl, (df1, df2), sort_output=True, check_dtype=False)


def test_merge_index_right(memory_leak_check):
    """
    test merge(): with left_index and right_index specified
    with all on = right
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True, how="right")
        return df3

    df1 = pd.DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
    df2 = pd.DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])
    check_func(impl, (df1, df2), sort_output=True, check_dtype=False)


def test_merge_index_outer(memory_leak_check):
    """
    test merge(): with left_index and right_index specified
    with all on = outer
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True, how="outer")
        return df3

    df1 = pd.DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
    df2 = pd.DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])

    check_func(impl, (df1, df2), sort_output=True, check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df1, df2",
    [
        (
            pd.DataFrame(
                {"A": ["foo", "bar", "baz", "bat", "fab"], "B": [3, 4, 5, 6, 7]},
                index=[-1, -1, -2, -3, -4],
            ),
            pd.DataFrame(
                {"A": ["baz", "bat", "fab"], "B": [1, 3, 5]}, index=[-1, 4, 5]
            ),
        ),
        (
            pd.DataFrame({1: ["foo", "bar", "baz"], "B": [3, 4, 5]}, index=[0, -1, 2]),
            pd.DataFrame(
                {1: ["baz", "baz", "foo", "foo", "bar"], "B": [1, 3, 2, 3, 4]},
                index=[-1, 2, 2, 0, 0],
            ),
        ),
        (
            pd.DataFrame({"A": ["foo", "bar"], "B": [3, 4]}, index=[-1, -1]),
            pd.DataFrame({"A": ["baz", "foo"], "B": [1, 3]}, index=[-1, -1]),
        ),
        (
            pd.DataFrame(
                {"A": ["foo", "bar", "baz", "bar"], "B": [3, 4, 5, 3]},
                index=[0, 0, -1, -1],
            ),
            pd.DataFrame(
                {"A": ["baz", "baz", "foo", "foo", "bar"], "B": [1, 3, 2, 3, 4]},
                index=[-1, -1, 2, 0, 0],
            ),
        ),
    ],
)
def test_merge_non_unique_index(df1, df2, memory_leak_check):
    """
    test merge(): merge on left and right non-unique index
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_index=True)
        return df3

    check_func(impl, (df1, df2), sort_output=True, check_typing_issues=False)


# TODO: Add memory leak check when constant lowering memory leak is fixed
def test_indicator_true():
    """
    test merge(): indicator=True
    """

    def impl1(df1, df2):
        return df1.merge(df2, how="inner", on="join_keys", indicator=True)

    def impl2(df1, df2):
        return df1.merge(df2, how="left", on="join_keys", indicator=True)

    def impl3(df1, df2):
        return df1.merge(df2, how="right", on="join_keys", indicator=True)

    def impl4(df1, df2):
        return df1.merge(df2, how="outer", on="join_keys", indicator=True)

    df1 = pd.DataFrame(
        {
            "join_keys": ["a", "b", "c", "d"] * 3,
            "value_x": np.array([98, 45, 45, 63, 13, 53, 48, 13, 49, 19, 82, 88]),
        }
    )
    df2 = pd.DataFrame(
        {
            "join_keys": ["b", "d", "e", "f"] * 3,
            "value_y": np.array([49, 46, 28, 13, 86, 65, 71, 64, 14, 28, 41, 78]),
        }
    )
    check_func(impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl3, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl4, (df1, df2), sort_output=True, reset_index=True)


def test_indicator_true_deadcol(memory_leak_check):
    """
    test merge(): indicator=True where the indicator column can be optimized out
    as a dead column.
    """

    def test_impl(df1, df2):
        merged_df = df1.merge(df2, how="outer", on="join_keys", indicator=True)
        return merged_df["value_x"]

    df1 = pd.DataFrame(
        {
            "join_keys": ["a", "b", "c", "d"] * 3,
            "value_x": np.array([98, 45, 45, 63, 13, 53, 48, 13, 49, 19, 82, 88]),
        }
    )
    df2 = pd.DataFrame(
        {
            "join_keys": ["b", "d", "e", "f"] * 3,
            "value_y": np.array([49, 46, 28, 13, 86, 65, 71, 64, 14, 28, 41, 78]),
        }
    )
    merge_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
        test_impl
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        merge_func(df1, df2)
        check_logger_msg(stream, "Output columns: ['value_x']")
    # Verify the index is dead.
    func_sig = merge_func
    confirmed_dead_index = False
    fir = func_sig.overloads[func_sig.signatures[0]].metadata["preserved_ir"]
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.join.Join):
                confirmed_dead_index = not stmt.has_live_out_index_var
    assert confirmed_dead_index, "Index not confirmed dead in join node"


def test_merge_all_nan_cols(memory_leak_check):
    """
    test merge(): all columns to merge on are null
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    df1 = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [0, 1, 2]})
    df2 = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [0, 1, 2]})

    check_func(impl, (df1, df2), sort_output=True, reset_index=True)


def test_merge_general_cond(memory_leak_check):
    """
    test merge(): with general condition expressions like "left.A == right.A"
    """

    # single equality term
    def impl1(df1, df2):
        return df1.merge(df2, on="left.A == right.A")

    # multiple equality terms
    def impl2(df1, df2):
        return df1.merge(df2, on="left.A == right.A & right.C == left.B")

    # single non-equality term
    def impl3(df1, df2):
        return df1.merge(df2, on="right.D <= left.B + 1 & left.A == right.A")

    # single non-equality term, multiple equal terms
    def impl4(df1, df2):
        return df1.merge(
            df2, on="left.B == right.C & right.D-5 >= left.B & left.A == right.A"
        )

    # multiple non-equality terms
    def impl5(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.D < left.B + 1 | right.C > left.B) & left.A == right.A",
            how=how,
        )

    # single non-equality term, no equality
    def impl6(df1, df2):
        return df1.merge(df2, on="right.D <= left.B + 1")

    # multiple non-equality terms, no equality
    def impl7(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.D < left.B + 1 | right.C > left.B)",
            how=how,
        )

    df1 = pd.DataFrame({"A": [1, 2, 1, 1, 3, 2, 3], "B": [1, 2, 3, 1, 2, 3, 1]})
    df2 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": [3, 2, 1, 3, 2, 1, 2],
            "D": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    # larger tables to test short/long table control flow in _join.cpp
    df3 = pd.concat([df1] * 10)
    df4 = pd.concat([df2] * 10)
    # nullable float
    df5 = pd.DataFrame(
        {
            "A": pd.array([1, 2, 1, 1, 3, 2, 3], "Float64"),
            "B": pd.array([1, 2, 3, 1, 2, 3, 1], "Float32"),
        }
    )
    df6 = pd.DataFrame(
        {
            "A": pd.array([4, 1, 2, 3, 2, 1, 4], "Float64"),
            "C": pd.array([3, 2, 1, 3, 2, 1, 2], "Float32"),
            "D": pd.array([1, 2, 3, 4, 5, 6, 7], "Float64"),
        }
    )

    py_out = df1.merge(df2, on="A")
    check_func(
        impl1,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    check_func(
        impl2,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D <= B + 1")
    check_func(
        impl3,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df3.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    py_out = py_out.query("D - 5 >= B")
    check_func(
        impl4,
        (df3, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B + 1 | C > B")
    check_func(
        impl5,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # # hit corner case
    df11 = pd.DataFrame({"A": [1, 1], "B": [1, 3]})
    df22 = pd.DataFrame(
        {
            "A": [1, 1],
            "C": [2, 1],
            "D": [2, 2],
        }
    )
    py_out = df11.merge(df22, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B + 1 | C > B")
    check_func(
        impl5,
        (df11, df22, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    py_out = df1.merge(df2, how="cross")
    py_out = py_out.query("D <= B + 1")
    check_func(
        impl6,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    py_out = df1.merge(df4, how="cross")
    py_out = py_out.query("D < B + 1 | C > B")
    check_func(
        impl7,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # testing left/right/outer cases needs Spark to generate reference output
    # avoid duplicated names for non-equi case
    df3 = df2.rename(columns={"A": "A1"})
    df7 = df6.rename(columns={"A": "A1"})
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    sdf1 = spark.createDataFrame(df1)
    sdf2 = spark.createDataFrame(df2)
    sdf3 = spark.createDataFrame(df3)
    sdf5 = spark.createDataFrame(df5)
    sdf6 = spark.createDataFrame(df6)
    sdf7 = spark.createDataFrame(df7)
    sdf1.createOrReplaceTempView("table1")
    sdf2.createOrReplaceTempView("table2")
    sdf3.createOrReplaceTempView("table3")
    sdf5.createOrReplaceTempView("table5")
    sdf6.createOrReplaceTempView("table6")
    sdf7.createOrReplaceTempView("table7")
    for how in ("left", "right", "outer"):
        # Spark requires "full outer" for some reason
        spark_how = "full outer" if how == "outer" else how
        # test with equality
        py_out = spark.sql(
            f"select * from table1 {spark_how} join table2 on (table2.D < table1.B + 1 or table2.C > table1.B) and table1.A == table2.A"
        ).toPandas()
        # spark duplicates key columns with nulls
        py_out_A = py_out.A.iloc[:, 0].combine_first(py_out.A.iloc[:, 1])
        py_out = py_out.drop(columns="A")
        py_out.insert(0, "A", py_out_A)
        check_func(
            impl5,
            (df1, df2, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )
        # test without equality
        py_out = spark.sql(
            f"select * from table1 {spark_how} join table3 on (table3.D < table1.B + 1 or table3.C > table1.B)"
        ).toPandas()
        check_func(
            impl7,
            (df1, df3, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )
        # test only one side being distributed
        out_df = _gather_output(
            bodo.jit(distributed=["df1"])(impl7)(
                _get_dist_arg(df1, True, True), df3, how
            )
        )
        if bodo.get_rank() == 0:
            _test_equal(
                out_df, py_out, sort_output=True, reset_index=True, check_dtype=False
            )
        out_df = _gather_output(
            bodo.jit(distributed=["df2"])(impl7)(
                df1, _get_dist_arg(df3, True, True), how
            )
        )
        if bodo.get_rank() == 0:
            _test_equal(
                out_df, py_out, sort_output=True, reset_index=True, check_dtype=False
            )
        # ---- Test nullable float arrays ----
        # test with equality
        py_out = spark.sql(
            f"select * from table5 {spark_how} join table6 on (table6.D < table5.B + 1 or table6.C > table5.B) and table5.A == table6.A"
        ).toPandas()
        # spark duplicates key columns with nulls
        py_out_A = py_out.A.iloc[:, 0].combine_first(py_out.A.iloc[:, 1])
        py_out = py_out.drop(columns="A")
        py_out.insert(0, "A", py_out_A)
        check_func(
            impl5,
            (df5, df6, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )
        # test without equality
        py_out = spark.sql(
            f"select * from table5 {spark_how} join table7 on (table7.D < table5.B + 1 or table7.C > table5.B)"
        ).toPandas()
        check_func(
            impl7,
            (df5, df7, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )


def test_merge_general_cond_na(memory_leak_check):
    """
    test merge(): with general condition expressions that include
    various NA values
    """

    # single non-equality term
    def impl1(df1, df2):
        return df1.merge(df2, on="right.D <= left.B + 1 & left.A == right.A")

    # single non-equality term, multiple equal terms
    def impl2(df1, df2):
        return df1.merge(df2, on="left.A == right.A & right.C+right.D-5 >= left.B")

    # multiple non-equality terms
    def impl3(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.C > left.B | right.D < left.B + 1) & left.A == right.A",
            how=how,
        )

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": pd.Series([1, None, 3, None, 2, 3, 1], dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": [3, 2, 1, 3, 2, 1, 2],
            "D": pd.Series([None, 2, 3, 4, 5, 6, 7], dtype="Int64"),
        }
    )
    # larger tables to test short/long table control flow in _join.cpp
    df3 = pd.concat([df1] * 10)
    df4 = pd.concat([df2] * 10)

    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    # Note we can't use df.query in Pandas because it can't
    # handle NA types
    filter_cond = py_out["D"] <= (py_out["B"] + 1)
    py_out = py_out[filter_cond]
    check_func(
        impl1,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df3.merge(df2, left_on=["A"], right_on=["A"])
    filter_cond = (py_out["C"] + py_out["D"] - 5) >= py_out["B"]
    py_out = py_out[filter_cond]
    check_func(
        impl2,
        (df3, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    filter_cond = (py_out["C"] > py_out["B"]) | (py_out["D"] < (py_out["B"] + 1))
    py_out = py_out[filter_cond]
    check_func(
        impl3,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_cond_na_float(memory_leak_check):
    """
    test merge(): with general condition expressions that include
    various NA float values
    """

    # single non-equality term
    def impl1(df1, df2):
        return df1.merge(df2, on="right.D <= left.B + 1 & left.A == right.A")

    # single non-equality term, multiple equal terms
    def impl2(df1, df2):
        return df1.merge(df2, on="left.A == right.A & right.C+right.D-5 >= left.B")

    # multiple non-equality terms
    def impl3(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.C > left.B | right.D < left.B + 1) & left.A == right.A",
            how=how,
        )

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": pd.Series([1, None, 3, None, 2, 3, 1], dtype="float64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": [3, 2, 1, 3, 2, 1, 2],
            "D": pd.Series([None, 2, 3, 4, 5, 6, 7], dtype="float32"),
        }
    )
    # larger tables to test short/long table control flow in _join.cpp
    df3 = pd.concat([df1] * 10)
    df4 = pd.concat([df2] * 10)
    # nullable float input
    df5 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": pd.Series([1, None, 3, None, 2, 3, 1], dtype="Float64"),
        }
    )
    df6 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": [3, 2, 1, 3, 2, 1, 2],
            "D": pd.Series([None, 2, 3, 4, 5, 6, 7], dtype="Float32"),
        }
    )

    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    # Note we can't use df.query in Pandas because it can't
    # handle NA types
    filter_cond = py_out["D"] <= (py_out["B"] + 1)
    py_out = py_out[filter_cond]
    check_func(
        impl1,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    py_out = df5.merge(df6, left_on=["A"], right_on=["A"])
    filter_cond = py_out["D"] <= (py_out["B"] + 1)
    py_out = py_out[filter_cond]
    check_func(
        impl1,
        (df5, df6),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    py_out = df3.merge(df2, left_on=["A"], right_on=["A"])
    filter_cond = (py_out["C"] + py_out["D"] - 5) >= py_out["B"]
    py_out = py_out[filter_cond]
    check_func(
        impl2,
        (df3, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    filter_cond = (py_out["C"] > py_out["B"]) | (py_out["D"] < (py_out["B"] + 1))
    py_out = py_out[filter_cond]
    check_func(
        impl3,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_cond_na_dt64(memory_leak_check):
    """
    test merge(): with general condition expressions that include
    various NA values stored in data1
    """

    # multiple non-equality terms
    def impl(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.C > left.B | right.D < left.B) & left.A == right.A",
            how=how,
        )

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": pd.Series(
                [
                    pd.Timestamp(2021, 10, 2),
                    None,
                    pd.Timestamp(2021, 11, 3),
                    None,
                    pd.Timestamp(2021, 11, 2),
                    3,
                    pd.Timestamp(2021, 10, 2),
                ],
                dtype="datetime64[ns]",
            ),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [4, 1, 2, 3, 2, 1, 4],
            "C": pd.Series(
                [
                    pd.Timestamp(2021, 11, 3),
                    pd.Timestamp(2021, 11, 2),
                    pd.Timestamp(2021, 10, 2),
                    3,
                    pd.Timestamp(2021, 11, 2),
                    pd.Timestamp(2021, 10, 2),
                    pd.Timestamp(2021, 11, 2),
                ],
                dtype="datetime64[ns]",
            ),
            "D": pd.Series(
                [
                    None,
                    pd.Timestamp(2021, 11, 2),
                    pd.Timestamp(2021, 11, 3),
                    pd.Timestamp(2021, 11, 4),
                    pd.Timestamp(2021, 11, 5),
                    pd.Timestamp(2021, 11, 6),
                    pd.Timestamp(2021, 11, 7),
                ],
                dtype="datetime64[ns]",
            ),
        }
    )
    # larger tables to test short/long table control flow in _join.cpp
    df4 = pd.concat([df2] * 10)

    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    filter_cond = (py_out["C"] > py_out["B"]) | (py_out["D"] < (py_out["B"]))
    py_out = py_out[filter_cond]
    check_func(
        impl,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_cond_binary(memory_leak_check):
    """
    test merge(): with general condition expressions like "left.A == right.A"
    with binary columns
    """

    # single equality term
    def impl1(df1, df2):
        return df1.merge(df2, on="left.A == right.A")

    # multiple equality terms
    def impl2(df1, df2):
        return df1.merge(df2, on="left.A == right.A & right.C == left.B")

    # single non-equality term
    def impl3(df1, df2):
        return df1.merge(df2, on="right.D <= left.B & left.A == right.A")

    # single non-equality term, multiple equal terms
    def impl4(df1, df2):
        return df1.merge(
            df2, on="left.B == right.C & right.D >= left.B & left.A == right.A"
        )

    # multiple non-equality terms
    def impl5(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.D < left.B | right.C < left.B) & left.A == right.A",
            how=how,
        )

    # multiple non-equality terms compare to spark
    def impl6(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.D < left.B | right.C < left.B) & left.A == right.A",
            how=how,
            _bodo_na_equal=False,
        )

    df1 = pd.DataFrame(
        {
            "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, object),
            "B": np.array([bytes(32), b"abcde", b"ihohi04324", None] * 3, object),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": np.array([b"cew", b"ce2r", b"r2r", None] * 3, object),
            "C": np.array([bytes(12), b"abcde", b"ihohi04324", None] * 3, object),
            "D": np.array([b"r32r2", b"poiu", b"3r32", b"3f3"] * 3, object),
        }
    )

    # larger tables to test short/long table control flow in _join.cpp
    df3 = pd.concat([df1] * 10)
    df4 = pd.concat([df2] * 10)

    py_out = df1.merge(df2, on="A")
    check_func(
        impl1,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    check_func(
        impl2,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D <= B")
    check_func(
        impl3,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df3.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    py_out = py_out.query("D >= B")
    check_func(
        impl4,
        (df3, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B | C < B")
    check_func(
        impl5,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # hit corner case
    df11 = pd.DataFrame(
        {
            "A": [b"a", b"a"] * 5,
            "B": [b"a", b"A"] * 5,
        }
    )
    df22 = pd.DataFrame(
        {
            "A": [b"a", b"a"] * 5,
            "C": [b"bcew", b"a"] * 5,
            "D": [b"bcew", b"bcew"] * 5,
        }
    )
    py_out = df11.merge(df22, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B | C < B")
    check_func(
        impl5,
        (df11, df22, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # test left/right/outer cases, needs Spark to generate reference output
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    sdf1 = spark.createDataFrame(df1)
    sdf2 = spark.createDataFrame(df2)
    sdf1.createOrReplaceTempView("table1")
    sdf2.createOrReplaceTempView("table2")
    for how in ("left", "right", "outer"):
        # Spark requires "full outer" for some reason
        spark_how = "full outer" if how == "outer" else how
        py_out = spark.sql(
            f"select * from table1 {spark_how} join table2 on (table2.D < table1.B or table2.C < table1.B) and table1.A == table2.A"
        ).toPandas()
        # spark duplicates key columns with nulls
        py_out_A = py_out.A.iloc[:, 0].combine_first(py_out.A.iloc[:, 1])
        py_out = py_out.drop(columns="A")
        py_out.insert(0, "A", py_out_A)
        # Spark uses a different array type from Bodo
        py_out[py_out.columns] = py_out[py_out.columns].apply(
            lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
            axis=1,
            result_type="expand",
        )
        check_func(
            impl6,
            (df1, df2, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )


def test_merge_general_cond_strings(memory_leak_check):
    """
    test merge(): with general condition expressions like "left.A == right.A"
    with string columns
    """

    # single equality term
    def impl1(df1, df2):
        return df1.merge(df2, on="left.A == right.A")

    # multiple equality terms
    def impl2(df1, df2):
        return df1.merge(df2, on="left.A == right.A & right.C == left.B")

    # single non-equality term
    def impl3(df1, df2):
        return df1.merge(df2, on="right.D <= left.B & left.A == right.A")

    # single non-equality term, multiple equal terms
    def impl4(df1, df2):
        return df1.merge(
            df2, on="left.B == right.C & right.D >= left.B & left.A == right.A"
        )

    # multiple non-equality terms
    def impl5(df1, df2, how):
        return df1.merge(
            df2,
            on="(right.D < left.B | right.C < left.B) & left.A == right.A",
            how=how,
        )

    df1 = pd.DataFrame(
        {
            "A": ["a", "bcew", "a", "a", "A", "bcew", "A"],
            "B": ["a", "bcew", "A", "a", "bcew", "A", "a"],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": ["re2f", "a", "bcew", "A", "bcew", "a", "bcew"],
            "C": ["A", "bcew", "a", "A", "bcew", "a", "bcew"],
            "D": ["a", "bcew", "A", "re2f", "", "Z", "Fe"],
        }
    )
    # larger tables to test short/long table control flow in _join.cpp
    df3 = pd.concat([df1] * 10)
    df4 = pd.concat([df2] * 10)

    py_out = df1.merge(df2, on="A")
    check_func(
        impl1,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    check_func(
        impl2,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D <= B")
    check_func(
        impl3,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df3.merge(df2, left_on=["A", "B"], right_on=["A", "C"])
    py_out = py_out.query("D >= B")
    check_func(
        impl4,
        (df3, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df4, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B | C < B")
    check_func(
        impl5,
        (df1, df4, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # hit corner case
    df11 = pd.DataFrame(
        {
            "A": ["a", "a"] * 5,
            "B": ["a", "A"] * 5,
        }
    )
    df22 = pd.DataFrame(
        {
            "A": ["a", "a"] * 5,
            "C": ["bcew", "a"] * 5,
            "D": ["bcew", "bcew"] * 5,
        }
    )
    py_out = df11.merge(df22, left_on=["A"], right_on=["A"])
    py_out = py_out.query("D < B | C < B")
    check_func(
        impl5,
        (df11, df22, "inner"),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )

    # test left/right/outer cases, needs Spark to generate reference output
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    sdf1 = spark.createDataFrame(df1)
    sdf2 = spark.createDataFrame(df2)
    sdf1.createOrReplaceTempView("table1")
    sdf2.createOrReplaceTempView("table2")
    for how in ("left", "right", "outer"):
        # Spark requires "full outer" for some reason
        spark_how = "full outer" if how == "outer" else how
        py_out = spark.sql(
            f"select * from table1 {spark_how} join table2 on (table2.D < table1.B or table2.C < table1.B) and table1.A == table2.A"
        ).toPandas()
        # spark duplicates key columns with nulls
        py_out_A = py_out.A.iloc[:, 0].combine_first(py_out.A.iloc[:, 1])
        py_out = py_out.drop(columns="A")
        py_out.insert(0, "A", py_out_A)
        check_func(
            impl5,
            (df1, df2, how),
            sort_output=True,
            reset_index=True,
            check_dtype=False,
            py_output=py_out,
        )


def test_merge_general_cond_all_keys(memory_leak_check):
    """
    test merge(): with general condition expressions where all non-equal
    conditions use only key columns.
    """
    # single equality term
    def impl(df1, df2):
        return df1.merge(
            df2, on="left.A == right.A & left.B == right.B & left.A > right.B"
        )

    df1 = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": np.arange(100) % 12,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": np.arange(100) % 14,
            "B": np.arange(100) % 11,
        }
    )
    py_out = df1.merge(df2, on=["A", "B"])
    py_out = py_out.query("A > B")
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_cond_rm_dead(memory_leak_check):
    """
    test dead code elimination in merge() with general condition expressions
    """

    # columns used in non-equality conditions are unused otherwise but shouldn't be
    # removed in dead code elimination
    def impl(df1, df2):
        df3 = df1.merge(df2, on="right.D2-5 >= left.C & left.A == right.A2")
        return df3[["A2", "B", "C", "E"]]

    def impl2(df1, df2):
        df3 = df1.merge(df2, on="right.D2-5 >= left.C")
        return df3[["A2", "B", "C", "E"]]

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": [1, 2, 3, 1, 2, 3, 1],
            "C": [1, 2, 3, 1, 2, 3, 1],
            "D": [1, 2, 3, 1, 2, 3, 1],
            "E": [1, 2, 3, 1, 2, 3, 1],
        }
    )
    df2 = pd.DataFrame(
        {
            "A2": [4, 1, 2, 3, 2, 1, 4],
            "B2": [3, 2, 1, 3, 2, 1, 2],
            "C2": [3, 2, 1, 3, 2, 1, 2],
            "D2": [1, 2, 3, 4, 5, 6, 7],
            "E2": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    py_out = df1.merge(df2, left_on="A", right_on="A2")
    py_out = py_out.query("D2-5 >= C")[["A2", "B", "C", "E"]]
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )
    py_out = df1.merge(df2, how="cross")
    py_out = py_out.query("D2-5 >= C")[["A2", "B", "C", "E"]]
    check_func(
        impl2,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_eq_cond(memory_leak_check):
    """
    test equality condition that can't be extracted in merge() with general condition
    expressions
    """

    # left.B == (right.C2 + 1) is equality but can't be extracted to left_on/right_on
    def impl(df1, df2):
        df3 = df1.merge(df2, on="left.B == (right.C2 + 1) & left.A == right.A2")
        return df3

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B": [1, 2, 3, 1, 2, 3, 1],
            "C": [1, 2, 3, 1, 2, 3, 1],
        }
    )
    df2 = pd.DataFrame(
        {
            "A2": [4, 1, 2, 3, 2, 1, 4],
            "B2": [3, 2, 1, 3, 2, 1, 2],
            "C2": [3, 2, 1, 3, 2, 1, 2],
        }
    )
    py_out = df1.merge(df2, left_on="A", right_on="A2")
    py_out = py_out.query("B == C2 + 1")
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_non_identifier_columns(memory_leak_check):
    """
    tests merge() with columns that are not valid python identifiers,
    encapsulated in a backtic
    """

    def impl(df1, df2):
        return df1.merge(
            df2, on="left.`$A` == right.`apply` and left.`exec` == right.`!@$%^&*`"
        )

    df1 = pd.DataFrame(
        {
            "$A": [1, 2, 1, 1, 3, 2, 3],
            "exec": [1, 2, 1, 1, 3, 2, 3],
        }
    )
    df2 = pd.DataFrame(
        {
            "apply": [3, 2, 1, 3, 2, 1, 2],
            "!@$%^&*": [1, 2, 1, 1, 3, 2, 3],
        }
    )
    py_out = df1.merge(df2, left_on=["$A", "exec"], right_on=["apply", "!@$%^&*"])
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_non_identifier_cond_columns(memory_leak_check):
    """
    Test non-identifier columns used directly as a condition rather than in a comparison.
    This helps test that name escapes work properly.
    """

    def impl(df1, df2):
        return df1.merge(df2, on="left.`A` == right.`B` & left.`__bodo_dummy__4=_left`")

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "__bodo_dummy__4=_left": [True, False, True, True, True, True, False],
        }
    )
    df2 = pd.DataFrame(
        {
            "B": [1, 2, 2, 1, 3, 2, 1],
        }
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["B"])
    py_out = py_out[py_out["__bodo_dummy__4=_left"]]
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


def test_merge_general_bool_columns(memory_leak_check):
    """
    tests merge() with boolean columns
    """

    def impl(df1, df2):
        return df1.merge(df2, on="left.`A` == right.`A` and left.`B1` and right.`B2`")

    df1 = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 3, 2, 3],
            "B1": [True, True, True, True, False, True, False],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [3, 2, 1, 3, 2, 1, 2],
            "B2": [False, True, True, True, False, True, False],
        }
    )
    py_out = df1.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out[py_out["B1"] & py_out["B2"]]
    check_func(
        impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


@pytest.mark.slow
def test_merge_match_key_types(memory_leak_check):
    """
    test merge(): where key types mismatch but values can be equal
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
    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl1, (df2, df1), sort_output=True, reset_index=True)
    check_func(test_impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl2, (df2, df1), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_match_key_types2(memory_leak_check):
    """
    test merge(): where key types mismatch in precision
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
    df2["A"] = df2.A.astype(np.int32)
    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl1, (df2, df1), sort_output=True, reset_index=True)
    check_func(test_impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl2, (df2, df1), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_match_key_types_nullable(memory_leak_check):
    """
    test merge(): where key types mismatch in precision and one is nullable
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
    df2["A"] = df2.A.astype("Int32")
    check_func(test_impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl1, (df2, df1), sort_output=True, reset_index=True)
    check_func(test_impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(test_impl2, (df2, df1), sort_output=True, reset_index=True)


def test_merge_cat_identical(memory_leak_check):
    """
    Test merge(): merge identical dataframes on categorical column
    """

    def test_impl(df1):
        df3 = df1.merge(df1, on="C2")
        return df3

    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")
    ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
    dtypes = {"C1": int, "C2": ct_dtype}
    df1 = pd.read_csv(fname, names=["C1", "C2"], dtype=dtypes, usecols=[0, 1])
    check_func(test_impl, (df1,), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_cat_multi_cols(memory_leak_check):
    """
    Test merge(): merge dataframes containing mutilple categorical cols
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on=["C1", "C2"])
        return df3

    ct_dtype1 = pd.CategoricalDtype(["2", "3", "4", "5", "t-"])
    ct_dtype2 = pd.CategoricalDtype(["A", "B", "C", "p"])
    dtypes = {"C1": ct_dtype1, "C2": ct_dtype2, "C3": str}
    df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes, skiprows=[0])
    df2 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes, skiprows=[1])
    # add extra rows to avoid empty output dataframes on 3 process test
    df3 = pd.DataFrame(
        {
            "C1": pd.Categorical(["t-"], ct_dtype1.categories),
            "C2": pd.Categorical(["p"], ct_dtype2.categories),
            "C3": ["A"],
        }
    )
    df1 = df1.append((df3, df1))
    df2 = df2.append((df3, df2))
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_cat1_inner(memory_leak_check):
    """
    Test merge(): merge dataframes containing categorical values
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": int, "C2": ct_dtype, "C3": str}
        df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        df1["C1"] = df1.C1 * 7 + 1
        n = len(df1) * 100
        df2 = pd.DataFrame({"C1": np.arange(n), "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1")
        return df3

    check_func(test_impl, (), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_cat1_right_2cols1(memory_leak_check):
    """
    Test merge(): setting NaN in categorical array
    a smaller test case for test_join_cat1_right()
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat3.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": int, "C2": ct_dtype}
        df1 = pd.read_csv(fname, names=["C1", "C2"], dtype=dtypes)
        n = len(df1)
        df2 = pd.DataFrame({"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    check_func(test_impl, (), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_cat1_right_2cols2(memory_leak_check):
    """
    Test merge(): setting NaN in categorical array
    bug fixed: some get_item_size that did not work for strings
    a smaller test case for test_join_cat1_right()
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat4.csv")

    def test_impl():
        dtypes = {"C1": int, "C2": str}
        df1 = pd.read_csv(fname, names=["C1", "C2"], dtype=dtypes)
        df1["C1"] = df1.C1 * 7 + 1
        n = len(df1) * 100
        df2 = pd.DataFrame({"C1": np.arange(n), "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    check_func(test_impl, (), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_merge_cat1_right(memory_leak_check):
    """
    Test merge(): setting NaN in categorical array
    """
    fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": int, "C2": ct_dtype, "C3": str}
        df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        df1["C1"] = df1.C1 * 7 + 1
        n = len(df1) * 100
        df2 = pd.DataFrame({"C1": np.arange(n), "AAA": n + np.arange(n) + 1.0})
        df3 = df1.merge(df2, on="C1", how="right")
        return df3

    check_func(test_impl, (), sort_output=True, reset_index=True)


@pytest.mark.slow
@pytest.mark.parametrize("n", [11, 11111])
def test_merge_parallel_optimize(n, memory_leak_check):
    """
    Test merge(): ensure parallelism optimization
    """

    def test_impl(n):
        df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
        df2 = pd.DataFrame({"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0})
        df3 = pd.merge(df1, df2, left_on="key1", right_on="key2")
        return df3.B.sum()

    check_func(test_impl, (n,))
    assert count_array_REPs() == 0  # assert parallelism
    assert count_parfor_REPs() == 0  # assert parallelism


@pytest.mark.slow
def test_merge_left_parallel(memory_leak_check):
    """
    Test merge(): merge with only left dataframe columns distributed
    ensure parallelism
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on=["A", "B"])
        return df3.C.sum() + df3.D.sum()

    bodo_func = bodo.jit(distributed_block=["df1"])(test_impl)
    df1 = pd.DataFrame(
        {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
    )

    df2 = pd.DataFrame(
        {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
    )
    start, end = get_start_end(len(df1))
    assert test_impl(df1, df2) == bodo_func(df1.iloc[start:end], df2)


def test_join_rm_dead_data_name_overlap1(memory_leak_check):
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


def test_join_rm_dead_data_name_overlap2(memory_leak_check):
    """
    Test join dead code elimination when there are matching names in data columns of
    input tables but only one of them is actually used.
    """

    def test_impl(df1, df2):
        return df1.merge(df2, left_on=["id"], right_on=["user_id"])

    df1 = pd.DataFrame({"id": [3, 4, 1]})
    df2 = pd.DataFrame({"id": [3, 4, 2], "user_id": [3, 5, 6]})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_join_deadcode_cleanup(memory_leak_check):
    """
    Test join dead code elimination when a merged dataframe is never used,
    merge() is not executed
    """

    def test_impl(df1, df2):  # pragma: no cover
        df3 = df1.merge(df2, on=["A"])
        return

    def test_impl_with_join(df1, df2):  # pragma: no cover
        df3 = df1.merge(df2, on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "C": [4, 5, 6]})

    j_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(test_impl)
    j_func_with_join = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"A": [1, 11, 3], "B": [4, 5, 1]}, index=[1, 4, 3]),
        pytest.param(
            pd.DataFrame(
                {"A": [1, 11, 3], "B": [4, 5, 1], "C": [-1, 3, 4]}, index=[1, 4, 3]
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame({"D": [-1.0, 1.0, 3.0]}, index=[-1, 1, 3]),
        pytest.param(
            pd.DataFrame(
                {"D": [-1.0, 1.0, 3.0], "E": [-1.0, 0.0, 1.0]}, index=[-1, 1, 3]
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_join_call(df1, df2, memory_leak_check):
    """
    test join() default behavior:
    impl1(): join on index when key columns not provided
    impl2(): left on key column, right on index
    """

    def impl1(df1, df2):
        df3 = df1.join(df2)
        return df3

    check_func(impl1, (df1, df2), sort_output=True, check_dtype=False)

    def impl2(df1, df2):
        return df1.join(df2, on="A")

    check_func(impl2, (df1, df2), sort_output=True, check_dtype=False)


@pytest.mark.slow
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
        pd.DataFrame({"D": [1, 9, 3], "E": [-1.0, 1.0, 3.0]}, index=[-1, 1, 3]),
        pd.DataFrame(
            {"D": [4, 6, 1], "E": [-1.0, 1.0, 3.0], "F": [-1.0, 0.0, 1.0]},
            index=[-1, 1, 3],
        ),
    ],
)
def test_join_how(df1, df2, memory_leak_check):
    """
    test join() 'how'
    """

    def impl_left(df1, df2):
        df3 = df1.join(df2, how="left")
        return df3

    def impl_right(df1, df2):
        df3 = df1.join(df2, how="right")
        return df3

    def impl_outer(df1, df2):
        df3 = df1.join(df2, how="outer")
        return df3

    def impl_inner(df1, df2):
        df3 = df1.join(df2, how="inner")
        return df3

    check_func(impl_left, (df1, df2), sort_output=True, check_dtype=False)
    check_func(impl_right, (df1, df2), sort_output=True, check_dtype=False)
    check_func(impl_outer, (df1, df2), sort_output=True, check_dtype=False)
    check_func(impl_inner, (df1, df2), sort_output=True, check_dtype=False)


# ------------------------------ merge on the index and column ------------------------------ #


@pytest.mark.slow
@pytest.mark.parametrize(
    "df1",
    [
        # left dataframes
        pd.DataFrame(
            {"A": [3, 1, 1, 3, 4], "B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}
        ).set_index("A"),
        pd.DataFrame(
            {"B": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}, index=[3, 1, 1, 3, 4]
        ),
        pd.DataFrame(
            {"X": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}, index=[3, 1, 1, 3, 4]
        ),
        pd.DataFrame(
            {"A": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5]}, index=[3, 1, 1, 3, 4]
        ),
        pd.DataFrame(
            {"A": [1, 2, 3, 2, 3], "C": [7, 8, 9, 4, 5], "Z": [3, 1, 1, 3, 4]}
        ).set_index("Z"),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        # right dataframes
        pd.DataFrame(
            {"A": [2, 1, 4, 4, 3], "B": [1, 3, 2, 3, 2], "D": [1, 2, 3, 4, 8]}
        ),
        pd.DataFrame(
            {
                "A": [2, 1, 4, 4, 3],
                "B": [1, 3, 2, 3, 2],
                "D": [1, 2, 3, 4, 8],
                "Y": [10, 11, 12, 13, 14],
            }
        ).set_index("Y"),
        pd.DataFrame(
            {
                "A": [2, 1, 4, 4, 3],
                "B": [1, 3, 2, 3, 2],
                "D": [1, 2, 3, 4, 8],
                "Y": ["a", "b", "c", "d", "e"],
            }
        ).set_index("Y"),
    ],
)
def test_merge_index_column_second(df1, df2, memory_leak_check):
    """
    Test merge(): test the merging with one key on the index and the other on the column
    This is another pattern.
    """

    def f(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    check_func(f, (df1, df2), sort_output=True)


def test_merge_index_column(memory_leak_check):
    """
    Test merge(): test the merging with one key on the index and the other on the column
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    def f2(df1, df2):
        df3 = df1.merge(df2, right_index=True, left_on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [1, 2], "C": ["a", "b"]})
    df2 = pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})

    check_func(f1, (df1, df2), sort_output=True, check_typing_issues=False)
    check_func(f2, (df1, df2), sort_output=True, check_typing_issues=False)


def test_merge_index_column_returning_empty(memory_leak_check):
    """
    Test merge(): Same as first, but returning empty dataframe
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [1, 2], "C": ["a", "b"]})
    df2 = pd.DataFrame({"A": [4, 5], "B": ["c", "d"]})

    # We need reset_index=True because the dataframe is empty.
    check_func(
        f1, (df1, df2), sort_output=True, reset_index=True, check_typing_issues=False
    )


@pytest.mark.slow
def test_merge_index_column_nontrivial_index(memory_leak_check):
    """
    Test merge(): Same as first but with a non-trivial index
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [4, 5], "C": ["a", "b"]}, index=[4, 5])
    df2 = pd.DataFrame({"A": [4, 5], "B": ["c", "d"]}, index=[4, 5])

    check_func(f1, (df1, df2), sort_output=True, check_typing_issues=False)


@pytest.mark.slow
def test_merge_index_column_double_index(memory_leak_check):
    """
    Test merge(): Same as first but with an index being double.
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    df1 = pd.DataFrame({"A": [4.0, 5.0], "C": ["a", "b"]}, index=[0.0, 4.0])
    df2 = pd.DataFrame({"A": [4.0, 5.0], "C": ["a", "b"]}, index=[0.0, 4.0])

    check_func(f1, (df1, df2), sort_output=True, check_typing_issues=False)


@pytest.mark.slow
def test_merge_index_column_string_index(memory_leak_check):
    """
    Test merge(): Same as first but with a string index
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    df1 = pd.DataFrame({"A": ["aa", "bb"], "C": ["a", "b"]}, index=["zz", "aa"])
    df2 = pd.DataFrame({"A": ["aa", "bb"], "C": ["a", "b"]}, index=["zz", "aa"])

    check_func(f1, (df1, df2), sort_output=True, check_typing_issues=False)


def test_merge_index_column_binary_index(memory_leak_check):
    """
    Test merge(): Same as first but with a binary index
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"])
        return df3

    df1 = pd.DataFrame(
        {"A": [b"aa", b"bb", b"cc"] * 3, "C": [b"a", b"b", b"c"] * 3},
        index=[b"z", b"a", b"c", b"zz", b"aa", b"cc", b"zzz", b"aaa", b"ccc"],
    )
    df2 = pd.DataFrame(
        {"A": [b"aa", b"bb", b"cc"] * 3, "C": [b"a", b"b", b"c"] * 3},
        index=[b"z", b"a", b"c", b"zz", b"aa", b"cc", b"zzz", b"aaa", b"ccc"],
    )

    check_func(f1, (df1, df2), sort_output=True)


def test_merge_index_column_how(memory_leak_check):
    """
    Test merge(): Same as first but with a variety of how merging.
    We need to have an index column that is made of floats since
    otherwise we would need to accept an index made of IntegerArrayType
    and that is not possible right now.
    The check_dtype=False is here because otherwise we have a collision
    between the dtype: one side is "Int64" and the other "int64".
    """

    def f1(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"], how="inner")
        return df3

    def f2(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"], how="left")
        return df3

    def f3(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"], how="right")
        return df3

    def f4(df1, df2):
        df3 = df1.merge(df2, left_index=True, right_on=["A"], how="outer")
        return df3

    np.random.seed(5)

    siz = 40
    df1 = pd.DataFrame(
        {"A": 0.0 + np.random.randint(0, 10, siz), "C": np.random.randint(0, 10, siz)},
        index=0.0 + np.random.randint(0, 10, siz),
    )
    df2 = pd.DataFrame(
        {"A": 0.0 + np.random.randint(0, 10, siz), "C": np.random.randint(0, 10, siz)},
        index=0.0 + np.random.randint(0, 10, siz),
    )

    check_func(f1, (df1, df2), sort_output=True, check_dtype=False)
    check_func(f2, (df1, df2), sort_output=True, check_dtype=False)
    check_func(f3, (df1, df2), sort_output=True, check_dtype=False)
    check_func(f4, (df1, df2), sort_output=True, check_dtype=False)


def test_merge_partial_distributed(memory_leak_check):
    """Only one dataframe is distributed, the other fixed.
    In that case in principle we do not need shuffle exchanges.
    However, in case of having say left table replicated and how='left',
    we need to handle how the rows of the left table are matched.
    """

    def test_impl(df1, df2):
        df3 = df1.merge(df2, on="A", how="outer")
        return df3

    np.random.seed(5)
    siz = 50
    df1 = pd.DataFrame(
        {"A": np.random.randint(0, 10, siz), "C": np.random.randint(0, 10, siz)}
    )
    df2 = pd.DataFrame(
        {"A": np.random.randint(0, 10, siz), "D": np.random.randint(0, 10, siz)}
    )
    start, end = get_start_end(len(df1))
    bdf1 = df1.iloc[start:end]

    bodo_impl = bodo.jit(distributed_block={"df1", "df3"})(test_impl)
    df3_bodo1 = bodo_impl(bdf1, df2)
    df3_bodo2 = (
        bodo.gatherv(df3_bodo1).sort_values(by=["A", "C", "D"]).reset_index(drop=True)
    )
    df3_pd = test_impl(df1, df2).sort_values(by=["A", "C", "D"]).reset_index(drop=True)
    if bodo.get_rank() == 0:
        pd.testing.assert_frame_equal(
            df3_bodo2, df3_pd, check_dtype=False, check_column_type=False
        )


def _gen_df_rand_col_names():
    """
    generate a dataframe with random column names
    """
    random.seed(3)
    df = pd.DataFrame()
    # 20 columns should have significant probability of hash difference in set ops
    for i in range(20):
        k = random.randint(2, 20)
        name = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
        # different value types in columns
        val = np.int32(2)
        if i % 2 == 0:
            val = 3.2
        df[name] = val

    return df


def test_merge_common_col_ordering(memory_leak_check):
    """
    Test merge() with several common column names as keys to make sure ordering of set
    operations to find key names is consistent across processors
    (it will hang or crash otherwise)
    """

    def impl(df1, df2):
        return df1.merge(df2)

    df1 = _gen_df_rand_col_names()
    df2 = df1.copy()
    df2["C"] = 3
    # We need reset_index=True because the dataframe is empty.
    check_func(impl, (df1, df2), sort_output=True, reset_index=True)


# TODO: add memory_leak_check
def test_merge_nested_arrays_non_keys(nested_arrays_value):
    def test_impl(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3

    df1 = pd.DataFrame(
        {"A": [0, 10, 200, 3000, 40000, 500000], "B": nested_arrays_value[0]}
    )
    df2 = pd.DataFrame(
        {"A": [0, 200, 200, 500000, 0, 3000], "B": nested_arrays_value[1]}
    )

    check_func(
        test_impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
        convert_columns_to_pandas=True,
    )


# ------------------------------ merge_asof() ------------------------------ #


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_seq(memory_leak_check):
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
    pd.testing.assert_frame_equal(
        bodo_func(df1, df2), test_impl(df1, df2), check_column_type=False
    )


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_parallel(datapath, memory_leak_check):
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "df1, df2, expected_output",
    [
        (
            pd.DataFrame(
                {
                    "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="Int64"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Int64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Int64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="float64"),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="float64"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="float64"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Int64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Int64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Float64"),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Float64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Float64"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Float64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="Float64"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Float64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Float64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="float32"),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="float32"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="float32"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Int64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Int64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [5, None, 1, 0, None, 7] * 2, dtype="datetime64[ns]"
                    ),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="datetime64[ns]"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="datetime64[ns]"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Int64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Int64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [5, None, 1, 0, None, 7] * 2, dtype="timedelta64[ns]"
                    ),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="timedelta64[ns]"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([5, 5, 5, 5, 1, 1, 1, 1], dtype="timedelta64[ns]"),
                    "B1": pd.Series(
                        [1, 1, None, None, None, None, 1, 1], dtype="Int64"
                    ),
                    "B2": pd.Series([2, 3, 2, 3, 2, 3, 2, 3], dtype="Int64"),
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "A": pd.Series([True, None, False, False], dtype="boolean"),
                    "B1": pd.Series([1, 2, None, 3], dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([False, False, None], dtype="boolean"),
                    "B2": pd.Series([None, 2, 4], dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([False, False, False, False], dtype="boolean"),
                    "B1": pd.Series([None, None, 3, 3], dtype="Int64"),
                    "B2": pd.Series([None, 2, None, 2], dtype="Int64"),
                }
            ),
        ),
        # Test that int64 works the same as normal
        (
            pd.DataFrame(
                {
                    "A": pd.Series([5, 231, 1, 0, 231, 7] * 2, dtype="int64"),
                    "B1": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([2, 5, 6, 6, 231, 1] * 2, dtype="int64"),
                    "B2": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            5,
                            5,
                            5,
                            5,
                            231,
                            231,
                            231,
                            231,
                            231,
                            231,
                            231,
                            231,
                            1,
                            1,
                            1,
                            1,
                        ],
                        dtype="int64",
                    ),
                    "B1": pd.Series(
                        [
                            1,
                            1,
                            None,
                            None,
                            2,
                            2,
                            1,
                            1,
                            3,
                            3,
                            None,
                            None,
                            None,
                            None,
                            1,
                            1,
                        ],
                        dtype="Int64",
                    ),
                    "B2": pd.Series(
                        [2, 3, 2, 3, None, 4, None, 4, None, 4, None, 4, 2, 3, 2, 3],
                        dtype="Int64",
                    ),
                }
            ),
        ),
    ],
)
def test_merge_nan_ne(df1, df2, expected_output, memory_leak_check):
    """
    Test for Bodo Extension that marks keys that
    are both NA as not equal.

    This is implemented for nullable types (Int, Bool),
    float, timedelta64, and datetime64.

    For all other arrays, this parameter should have no impact.
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on="A", _bodo_na_equal=False)

    # Using py_output because this extension doesn't exist
    # in Pandas
    check_func(
        test_impl,
        (df1, df2),
        py_output=expected_output,
        sort_output=True,
        reset_index=True,
    )


def test_merge_dead_keys(memory_leak_check):
    """tests pd.merge when eliminating dead keys"""

    def test_impl(df1, df2):
        return df1.merge(df2, left_on="A", right_on="B")[["key2", "A"]]

    n = 100
    df1 = pd.DataFrame({"key1": np.arange(n) + 3, "A": np.arange(n) + 1.0})
    df2 = pd.DataFrame({"key2": 2 * np.arange(n) + 1, "B": n + np.arange(n) + 1.0})
    check_func(
        test_impl,
        (df1, df2),
        sort_output=True,
        reset_index=True,
    )


def test_merge_repeat_key(memory_leak_check):
    """tests pd.merge when the same key is repeated twice"""

    def impl1(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["A", "B"])

    def impl2(df1, df2):
        return df2.merge(df1, left_on=["A", "B"], right_on=["A", "A"])

    def impl3(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["A", "B"], how="left")

    def impl4(df1, df2):
        return df2.merge(df1, left_on=["A", "B"], right_on=["A", "A"], how="left")

    def impl5(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["A", "B"], how="right")

    def impl6(df1, df2):
        return df2.merge(df1, left_on=["A", "B"], right_on=["A", "A"], how="right")

    def impl7(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["A", "B"], how="outer")

    def impl8(df1, df2):
        return df2.merge(df1, left_on=["A", "B"], right_on=["A", "A"], how="outer")

    df1 = pd.DataFrame(
        {
            "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
            "B": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
        }
    )
    check_func(impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl3, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl4, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl5, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl6, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl7, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl8, (df1, df2), sort_output=True, reset_index=True)


def test_merge_repeat_key_same_frame(memory_leak_check):
    """tests pd.merge when the same key is repeated twice and keys
    aren't shared between DataFrames."""

    def impl1(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["B", "B"])

    def impl2(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["B", "B"])

    def impl3(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["B", "B"])

    def impl4(df1, df2):
        return df1.merge(df2, left_on=["A", "A"], right_on=["B", "B"])

    df1 = pd.DataFrame(
        {
            "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
            "B": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
        }
    )
    check_func(impl1, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl3, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl4, (df1, df2), sort_output=True, reset_index=True)


def test_merge_output_cast_key_order(memory_leak_check):
    """Test for the issue in [BE-3979], where a key in the output
    needs to be cast back to a smaller type but the key number and
    input column numbers differ.
    """

    def impl(df1, df2):
        return df1.merge(df2, left_on=["l1", "l0"], right_on=["r0", "r1"], how="outer")

    df1 = pd.DataFrame(
        {
            "l0": pd.arrays.ArrowStringArray(
                pa.array(
                    ["abc", "b", "h", "abc", "h", "b", "cde"],
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
            "l1": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="Int32"),
        }
    )
    df2 = pd.DataFrame(
        {
            "r0": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="Int64"),
            "r1": pd.arrays.ArrowStringArray(
                pa.array(
                    ["abc", "b", "h", "abc", "h", "b", "cde"],
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )
    check_func(impl, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
class TestJoin(unittest.TestCase):
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
            dtypes = {"C1": int, "C2": ct_dtype, "C3": str}
            df1 = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
            n = len(df1)
            df2 = pd.DataFrame(
                {"C1": 2 * np.arange(n) + 1, "AAA": n + np.arange(n) + 1.0}
            )
            df3 = df1.merge(df2, on="C1")
            return df3

        bodo_func = bodo.jit(distributed_block=["df3"])(test_impl)
        # TODO: check results
        self.assertTrue((bodo_func().columns == test_impl().columns).all())


if __name__ == "__main__":
    unittest.main()
