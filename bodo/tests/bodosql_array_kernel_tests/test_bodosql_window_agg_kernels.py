# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL window/aggregation functions
"""

import math

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    [10, 10, 10, 11, 15, 15, 15, 16, 16, 16, 19, 19],
                    dtype=pd.UInt8Dtype(),
                ),
                pd.Series([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4]),
            ),
            id="uint8_sorted_no_null_with_duplicates",
        ),
        pytest.param(
            (
                pd.Series([0, 10, 30, 31, 40, 41, 50, 89], dtype=pd.Int8Dtype()),
                pd.Series([0, 1, 2, 3, 4, 5, 6, 7]),
            ),
            id="int8_sorted_no_null_no_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [None, None, None, None, 25, 25, 50, 75, 75, 75, 100],
                    dtype=pd.UInt16Dtype(),
                ),
                pd.Series([0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 3]),
            ),
            id="uint16_sorted_with_null_with_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([None, 100, 200, 300, 400, 500], dtype=pd.Int16Dtype()),
                pd.Series([0, 0, 1, 2, 3, 4]),
            ),
            id="int16_sorted_with_null_no_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [25, 25, 50, 50, 50, 75, 25, 75, 75, 75, 50, 50],
                    dtype=pd.UInt32Dtype(),
                ),
                pd.Series([0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5]),
            ),
            id="uint32_unsorted_no_null_with_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([100, 0, 75, 25, 50], dtype=pd.Int32Dtype()),
                pd.Series([0, 1, 2, 3, 4]),
            ),
            id="int32_unsorted_no_null_no_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [None, None, None, None, 64, None, 2, 2, 4, None, None, 4, 8],
                    dtype=pd.UInt64Dtype(),
                ),
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3]),
            ),
            id="uint64_unsorted_with_null_with_duplicates",
        ),
        pytest.param(
            (
                pd.Series([13, 2, 7, None, 5, 3], dtype=pd.Int64Dtype()),
                pd.Series([0, 1, 2, 2, 3, 4]),
            ),
            id="int64_unsorted_with_null_no_duplicates",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([None, None, None, None, None, 42], dtype=pd.Int32Dtype()),
                pd.Series([0] * 6),
            ),
            id="int32_almost_all_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([None] * 6, dtype=pd.Int32Dtype()),
                pd.Series([0] * 6),
            ),
            id="int32_all_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [None, "A", None, None, None, "A", None] + list("AAACCCCBAA")
                ),
                pd.Series([0] * 10 + [1, 1, 1, 1, 2, 3, 3]),
            ),
            id="string_unsorted_with_null_with_duplicates",
        ),
        pytest.param(
            (
                pd.Series([1.0, None, 2.0, None, 4.0, None, 3.0]),
                pd.Series([0, 0, 1, 1, 2, 2, 3]),
            ),
            id="float_unsorted_with_null_with_duplicates",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if y == "n" else pd.Timestamp(f"201{y}")
                        for y in "nn08n000n08155n"
                    ]
                ),
                pd.Series([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 5, 5]),
            ),
            id="timestamp_unsorted_with_null_with_duplicates",
        ),
        pytest.param(
            (
                pd.Series([True, False, False, False, True, False, None, False, True]),
                pd.Series([0, 1, 1, 1, 2, 3, 3, 3, 4]),
            ),
            id="bool_unsorted_with_null_with_duplicates",
        ),
    ],
)
def test_change_event(args):
    def impl(S):
        return bodo.libs.bodosql_window_agg_array_kernels.change_event(S)

    S, answer = args
    check_func(
        impl,
        (S,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )


def window_refsol(S, lower, upper, func):
    L = []
    for i in range(len(S)):
        if upper < lower:
            result = None
        else:
            # Extract the window frame of elements by slicing about the current
            # index using the lower/upper bounds (without going out of bounds)
            elems = [
                elem
                for elem in S.iloc[
                    np.clip(i + lower, 0, len(S)) : np.clip(i + upper + 1, 0, len(S))
                ]
                if not pd.isna(elem)
            ]
            if func == "sum":
                result = None if len(elems) == 0 else sum(elems)
            elif func == "count":
                result = len(elems)
            elif func == "avg":
                result = None if len(elems) == 0 else sum(elems) / len(elems)
            elif func == "median":
                result = None if len(elems) == 0 else np.median(elems)
            elif func == "ratio_to_report":
                result = (
                    None
                    if len(elems) == 0 or S[i] == None or sum(elems) == 0
                    else S[i] / sum(elems)
                )
        L.append(result)
    dtype_map = {
        "sum": pd.Int64Dtype() if S.dtype.kind == "i" else None,
        "count": pd.Int64Dtype(),
        "avg": None,
        "median": None,
        "ratio_to_report": None,
    }
    out_dtype = dtype_map[func]
    return pd.Series(L, dtype=out_dtype)


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(
            pd.Series([1, 2, 3, 4, 5], dtype=pd.UInt8Dtype()),
            id="uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [1, None, -2, 3, None, None, None, -4, 5, 6], dtype=pd.Int32Dtype()
            ),
            id="int32",
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i <= 2 or i >= 18 or i % 7 >= 5
                    else (-1) ** i * round(math.pi, i)
                    for i in range(20)
                ]
            ),
            id="float64",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    ["lower_bound", "upper_bound"],
    [
        pytest.param(-1000, 0, id="prefix"),
        pytest.param(0, 1, id="suffix"),
        pytest.param(-1000, -1, id="prefix_exclusive"),
        pytest.param(1, 1000, id="suffix_exclusive", marks=pytest.mark.slow),
        pytest.param(-1000, 1000, id="entire_window"),
        pytest.param(0, 0, id="current"),
        pytest.param(-1, 1, id="rolling_3"),
        pytest.param(-2, 0, id="lagging_3", marks=pytest.mark.slow),
        pytest.param(1, 3, id="leading_3", marks=pytest.mark.slow),
        pytest.param(-1000, -500, id="too_small"),
        pytest.param(100, 200, id="too_large", marks=pytest.mark.slow),
        pytest.param(3, -3, id="backward"),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "median",
        "sum",
        "count",
        "avg",
        "ratio_to_report",
    ],
)
def test_windowed_kernels_numeric(func, S, lower_bound, upper_bound, memory_leak_check):
    def impl1(S, lower, upper):
        return pd.Series(bodo.libs.bodosql_array_kernels.windowed_sum(S, lower, upper))

    def impl2(S, lower, upper):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.windowed_count(S, lower, upper)
        )

    def impl3(S, lower, upper):
        return pd.Series(bodo.libs.bodosql_array_kernels.windowed_avg(S, lower, upper))

    def impl4(S, lower, upper):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.windowed_median(S, lower, upper)
        )

    def impl5(S, lower, upper):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.windowed_ratio_to_report(S, lower, upper)
        )

    implementations = {
        "sum": impl1,
        "count": impl2,
        "avg": impl3,
        "median": impl4,
        "ratio_to_report": impl5,
    }
    impl = implementations[func]

    check_func(
        impl,
        (S, lower_bound, upper_bound),
        py_output=window_refsol(S, lower_bound, upper_bound, func),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            pd.Series(
                [None if "0" in str(i) else (i**2) % 17 for i in range(500)],
                dtype=pd.UInt8Dtype(),
            ),
            id="uint8",
        ),
        pytest.param(
            pd.Series(
                [
                    None if "11" in str(i) else (((13 + i) ** 2) % 41) ** 0.4
                    for i in range(500)
                ]
            ),
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [chr(65 + (i**2) % 5) if i % 7 < 6 else None for i in range(250)]
            ),
            id="string",
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if "1" in str(i) and "2" in str(i)
                    else bytes(bin((i**2) % 47), encoding="utf-8")
                    for i in range(450)
                ]
            ),
            id="binary",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    None if year is None else pd.datetime(2000 + year, 1, 1)
                    for tup in zip(
                        [i for i in range(50)],
                        [((i + 3) ** 2) % 22 for i in range(50)],
                        [None] * 50,
                        [((i + 4) ** 3) % 22 for i in range(50)],
                        [((i + 5) ** 4) % 22 for i in range(50)],
                        [((i + 6) ** 5) % 22 for i in range(50)],
                    )
                    for year in tup
                ]
            ),
            id="datetime",
        ),
    ],
)
@pytest.mark.parametrize(
    ["lower_bound", "upper_bound"],
    [
        pytest.param(-1000, 0, id="prefix"),
        pytest.param(0, 1, id="suffix"),
        pytest.param(-1000, -1, id="prefix_exclusive"),
        pytest.param(1, 1000, id="suffix_exclusive", marks=pytest.mark.slow),
        pytest.param(-1000, 1000, id="entire_window"),
        pytest.param(0, 0, id="current"),
        pytest.param(-20, 20, id="rolling_41"),
        pytest.param(-1, 1, id="rolling_3", marks=pytest.mark.slow),
        pytest.param(1, 3, id="leading_3", marks=pytest.mark.slow),
        pytest.param(-1000, -700, id="too_small"),
        pytest.param(3, -3, id="backward", marks=pytest.mark.slow),
    ],
)
def test_windowed_mode(data, lower_bound, upper_bound, memory_leak_check):
    def impl(S, lower, upper):
        return pd.Series(bodo.libs.bodosql_array_kernels.windowed_mode(S, lower, upper))

    if bodo.get_size() > 1:
        pytest.skip("These kernels are only sequential")

    # Calculates the window function for each row, breaking ties by finding
    # the element that appeared first in the sequence chronologically
    def generate_answers(S, lower, upper):
        L = []
        for i in range(len(S)):
            if upper < lower:
                L.append(None)
            else:
                elems = [
                    elem
                    for elem in S.iloc[
                        np.clip(i + lower, 0, len(S)) : np.clip(
                            i + upper + 1, 0, len(S)
                        )
                    ]
                    if not pd.isna(elem)
                ]
                counts = {}
                bestVal = None
                bestCount = 0
                for elem in elems:
                    counts[elem] = counts.get(elem, 0) + 1
                    if counts[elem] > bestCount or (
                        counts[elem] == bestCount
                        and S[S == elem].index[0] < S[S == bestVal].index[0]
                    ):
                        bestCount = counts[elem]
                        bestVal = elem
                L.append(bestVal)
        return pd.Series(L, dtype=S.dtype)

    check_func(
        impl,
        (data, lower_bound, upper_bound),
        py_output=generate_answers(data, lower_bound, upper_bound),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )
