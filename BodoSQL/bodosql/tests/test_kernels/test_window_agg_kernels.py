"""Test Bodo's array kernel utilities for BodoSQL window/aggregation functions"""

import datetime
import math
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    nullable_float_arr_maker,
    pytest_mark_one_rank,
    pytest_slow_unless_codegen,
)

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


def uniform_distribution(n, seed):
    """Generates an array of random points in a uniform distribution

    Args:
        n (integer): the number of points to generate
        seed (integer): the starting seed for the generation

    Returns:
        np.ndarray[float64]: array of floats in a uniform distribbution
    """
    np.random.seed(seed)
    return np.random.uniform(0, 1000, n)


def gaussian_distribution(n, seed):
    """Generates an array of random points in a gaussian distribution

    Args:
        n (integer): the number of points to generate
        seed (integer): the starting seed for the generation

    Returns:
        np.ndarray[float64]: array of floats in a gaussian distribbution
    """
    np.random.seed(seed)
    return np.random.normal(75, 15, n)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series([math.tan(i) for i in range(30000)]), id="tangent"),
        pytest.param(
            pd.Series(list(range(30000)), dtype=pd.Int32Dtype()),
            id="linear_no_null",
        ),
        pytest.param(
            pd.Series(
                [None if i**0.5 == int(i**0.5) else i for i in range(90000)],
                dtype=pd.Int32Dtype(),
            ),
            id="linear_null",
        ),
        pytest.param(pd.Series([i**0.5 for i in range(100000)]), id="root_no_nan"),
        pytest.param(
            pd.Series([np.nan if i % 7 == 0 else i**0.5 for i in range(100000)]),
            id="root_nan",
        ),
        pytest.param(pd.Series(uniform_distribution(30000, 42)), id="uniform"),
        pytest.param(pd.Series(gaussian_distribution(30000, 42)), id="gaussian"),
    ],
)
def test_approx_percentile(data, memory_leak_check):
    def impl(arr):
        return (
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.001),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.01),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.1),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.25),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.42),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.5),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.63),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.75),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.9),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.99),
            bodo.libs.array_kernels.approx_percentile(arr.values, 0.999),
        )

    percentiles = [0.001, 0.01, 0.1, 0.25, 0.42, 0.5, 0.63, 0.75, 0.9, 0.99, 0.999]
    exact_answer = tuple([float(data.quantile(perc)) for perc in percentiles])
    # There are no strong accuracy guarantees for the t-digest quantile approximation
    # algorithm, so the best we can do is compare to the exact answer using
    # a somewhat large relative & absolute tolerance. 40% was chosen as an
    # arbitrary cutoff for the relative tolerance based on observations.
    check_func(
        impl,
        (data,),
        py_output=exact_answer,
        is_out_distributed=False,
        check_dtype=False,
        reset_index=True,
        atol=0.01,
        rtol=0.4,
    )


@pytest.mark.parametrize(
    "use_default",
    [pytest.param(True, id="with_default"), pytest.param(False, id="no_default")],
)
@pytest.mark.parametrize(
    "shift, answer",
    [
        pytest.param(-1, [1, 1, 1, 2, 3, 4, 4, -1, -1], id="negative"),
        pytest.param(2, [-1, -1, -1, -1, 0, 1, 2, 2, 3], id="positive"),
        pytest.param(0, [0, None, None, 1, 2, 3, None, 4, None], id="zero"),
    ],
)
@pytest.mark.parametrize(
    "values, dtype, default",
    [
        pytest.param([0, 1, 2, 3, 4], pd.Int32Dtype(), -1, id="int32"),
        pytest.param(
            [10.0, 3.1415926, -64.0, 125.0, 2.718281828], np.float64, 0.0, id="float64"
        ),
        pytest.param(
            ["alpha", "beta", "gamma", "delta", "epsilon"], None, "", id="string"
        ),
        pytest.param(
            [b"romeo", b"juliet", b"othello", b"hamlet", b"viola"],
            None,
            b"",
            id="binary",
        ),
        pytest.param(
            [True, False, True, False, True], pd.BooleanDtype(), True, id="boolean"
        ),
        pytest.param(
            [
                datetime.date.fromordinal(i)
                for i in [736879, 737729, 733133, 680082, 688783]
            ],
            None,
            datetime.date(1999, 12, 31),
            id="date",
        ),
        pytest.param(
            [bodo.types.Time(second=i) for i in [45000, 86399, 21600, 1, 1020]],
            None,
            bodo.types.Time(0, 0, 0),
            id="time",
        ),
        pytest.param(
            [
                pd.Timestamp("2023-1-1") + pd.Timedelta(hours=i)
                for i in [0, -10000, 20000, -30000, 40000]
            ],
            None,
            pd.Timestamp("1999-1-1"),
            id="naive_timestamp",
        ),
        pytest.param(
            [
                pd.Timestamp("2023-1-1", tz="US/Pacific") + pd.Timedelta(hours=i)
                for i in [0, 10000, -20000, 30000, -40000]
            ],
            None,
            pd.Timestamp("1999-1-1", tz="US/Pacific"),
            id="tz_timestamp",
        ),
    ],
)
def test_null_ignoring_shift(
    values, dtype, shift, default, use_default, answer, memory_leak_check
):
    """Tests null_ignoring_shift on multiple types with various shifts
    and default values via the following pattern:

     - Values: a list of five distinct values of the type being tested
     - Dtype: the datatype to be used when converting to a Series
     - Default: the default value to provide to the kernel
     - Shift: how much to shif tby
     - Answer: a list of length 9 that provides an output pattern corresponding
       to the input pattern shifted by the amount

     The pattern (defined below) is [0, None, None, 1, 2, 3, None, 4, None].
     Suppose our values were ["a", "b", "c", "d", "e"]. Then this would
     construct the following list:

         ["a", None, None, "b", "c", "d", None, "e", None]

     If we shifted by -1 with a default of "" we would get the following:

         ["b", "b", "b", "c", "d", "e", "e", "", ""]

     So the output pattern to replicate this would be as follows:

         [1, 1, 1, 2, 3, 4, 4, -1, -1]

     Where the numbers 0 to 4 represent which value from the original list
     of 5 values is used, and -1 indicates using the default.

     Note: if use_default is False, then None is used instead of the default
     value provided
    """

    pattern_list = [0, None, None, 1, 2, 3, None, 4, None]
    if not use_default:
        default = None

    input_list = []
    output_list = []
    for i in range(len(pattern_list)):
        if pattern_list[i] is None:
            input_list.append(None)
        else:
            input_list.append(values[pattern_list[i]])
        if answer[i] is None:
            output_list.append(None)
        elif answer[i] == -1:
            output_list.append(default)
        else:
            output_list.append(values[answer[i]])

    def impl(S, shift_amt, default_value):
        return pd.Series(
            bodosql.kernels.null_ignoring_shift(S, shift_amt, default_value)
        )

    check_func(
        impl,
        (pd.Series(input_list, dtype=dtype), shift, default),
        py_output=pd.Series(output_list, dtype=dtype),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )


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
                    ],
                    dtype="datetime64[ns]",
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
        return pd.Series(bodosql.kernels.window_agg_array_kernels.change_event(S))

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
        is_out_distributed=False,
    )


def window_refsol(S, lower, upper, func, use_nans=False):
    L = []
    to_nan = [-1]
    to_null = [-1]
    for i in range(len(S)):
        if upper < lower:
            result = None if func != "count(*)" else 0
        else:
            # Extract the window frame of elements by slicing about the current
            # index using the lower/upper bounds (without going out of bounds)
            elems = pd.Series(
                [
                    elem
                    for elem in S.iloc[
                        np.clip(i + lower, 0, len(S)) : np.clip(
                            i + upper + 1, 0, len(S)
                        )
                    ]
                    if ((str(elem) != "<NA>") if use_nans else not (pd.isna(elem)))
                ]
            )
            if func == "count":
                result = len(elems)
            elif func == "count(*)":
                # Does not filter nulls.
                result = np.clip(i + upper + 1, 0, len(S)) - np.clip(
                    i + lower, 0, len(S)
                )
            elif func == "count_if":
                # Summing over booleans will give us the count of true values.
                result = elems.sum()
            elif func == "boolor":
                result = (
                    None
                    if len(elems) == 0
                    else len(list(filter(lambda x: bool(x), elems))) > 0
                )
            elif func == "booland":
                result = (
                    None
                    if len(elems) == 0
                    else len(list(filter(lambda x: bool(x), elems))) == len(elems)
                )
            elif func == "boolxor":
                result = (
                    None
                    if len(elems) == 0
                    else len(list(filter(lambda x: bool(x), elems))) == 1
                )
            elif func == "bitor_agg":

                def impl(elems):
                    result = 0
                    for e in elems:
                        if pd.isna(e):
                            continue
                        result |= np.int64(e)
                    return result

                result = None if len(elems) == 0 else impl(elems)
            elif func == "bitand_agg":

                def impl(elems):
                    result = np.int64(-1)
                    for e in elems:
                        if pd.isna(e):
                            continue
                        result &= np.int64(e)
                    return result

                result = None if len(elems) == 0 else impl(elems)
            elif func == "bitxor_agg":

                def impl(elems):
                    result = 0
                    for e in elems:
                        if pd.isna(e):
                            continue
                        result ^= np.int64(e)
                    return result

                result = None if len(elems) == 0 else impl(elems)
            elif use_nans and ("nan" in map(str, elems)):
                result = np.nan
            elif func == "sum":
                result = None if len(elems) == 0 else elems.sum()
            elif func == "avg":
                result = None if len(elems) == 0 else elems.mean()
            elif func == "median":
                result = None if len(elems) == 0 else elems.median()
            elif func == "min":
                result = None if len(elems) == 0 else elems.min()
            elif func == "max":
                result = None if len(elems) == 0 else elems.max()
            elif func == "ratio_to_report":
                total = elems.sum()
                result = (
                    None
                    if len(elems) == 0 or S[i] == None or pd.isna(S[i]) or total == 0
                    else S[i] / total
                )
            elif func == "var_pop":
                result = None if len(elems) == 0 else np.var(elems, ddof=0)
            elif func == "var_samp":
                result = None if len(elems) <= 1 else np.var(elems, ddof=1)
            elif func == "stddev_pop":
                result = None if len(elems) == 0 else np.std(elems, ddof=0)
            elif func == "stddev_samp":
                result = None if len(elems) <= 1 else np.std(elems, ddof=1)
            elif func == "skew":
                result = None if len(elems) < 3 else elems.skew()
                # result = None if len(elems) == 0 else elems.skew()
            elif func == "kurtosis":
                result = None if len(elems) < 4 else elems.kurtosis()
        if use_nans:
            default = 0.0
            if func.startswith("bool"):
                default = False
            elif func.startswith("bit"):
                default = 0

            if result is None:
                to_null.append(i)
                L.append(default)
            elif result is np.nan:
                to_nan.append(i)
                L.append(default)
            else:
                L.append(result)
        else:
            L.append(result)
    if use_nans:
        return nullable_float_arr_maker(L, to_null, to_nan)
    else:
        dtype_map = {
            "sum": pd.Int64Dtype() if S.dtype.kind == "i" else None,
            "count": pd.Int64Dtype(),
        }
        out_dtype = dtype_map.get(func, None)
        return pd.Series(L, dtype=out_dtype)


# Calculates the window function Mode, breaking ties by finding
# the element that appeared first in the sequence chronologically
def window_refsol_mode(S, lower, upper, use_nans):
    L = []
    to_null = [-1]
    to_nan = [-1]
    for i in range(len(S)):
        if upper < lower:
            L.append(None)
        else:
            elems = pd.Series(
                [
                    elem
                    for elem in S.iloc[
                        np.clip(i + lower, 0, len(S)) : np.clip(
                            i + upper + 1, 0, len(S)
                        )
                    ]
                    if ((str(elem) != "<NA>") if use_nans else not (pd.isna(elem)))
                ]
            )
            if len(elems) == 0:
                if use_nans:
                    L.append(0.0)
                    to_null.append(i)
                else:
                    L.append(None)
            else:
                counts = {}
                if use_nans:
                    bestVal, bestCount = np.nan, np.isnan(elems).sum()
                else:
                    bestVal, bestCount = None, 0
                for elem in elems:
                    if use_nans and np.isnan(elem):
                        continue
                    counts[elem] = counts.get(elem, 0) + 1
                    if counts[elem] > bestCount or (
                        (not (use_nans and np.isnan(bestVal)))
                        and counts[elem] == bestCount
                        and S[S == elem].index[0] < S[S == bestVal].index[0]
                    ):
                        bestCount = counts[elem]
                        bestVal = elem
                if bestVal is np.nan:
                    L.append(0.0)
                    to_nan.append(i)
                else:
                    L.append(bestVal)
    if use_nans:
        return nullable_float_arr_maker(L, to_null, to_nan)
    else:
        return pd.Series(L, dtype=S.dtype)


def window_refsol_double(Y, X, lower, upper, func, use_nans=False):
    L = []
    to_null = [-1]
    to_nan = [-1]
    for i in range(len(Y)):
        if upper < lower:
            result = None
        else:
            # Extract the window frame of elements by slicing about the current
            # index using the lower/upper bounds (without going out of bounds)
            elems = [
                (y, x)
                for y, x in zip(
                    Y.iloc[
                        np.clip(i + lower, 0, len(Y)) : np.clip(
                            i + upper + 1, 0, len(Y)
                        )
                    ],
                    X.iloc[
                        np.clip(i + lower, 0, len(X)) : np.clip(
                            i + upper + 1, 0, len(X)
                        )
                    ],
                )
                if (
                    (str(x) != "<NA>" and str(y) != "<NA>")
                    if use_nans
                    else not (pd.isna(x) or pd.isna(y))
                )
            ]
            elems_y = pd.Series(y for y, _ in elems)
            elems_x = pd.Series(x for _, x in elems)
            if use_nans and ("nan" in map(str, elems_y) or "nan" in map(str, elems_x)):
                result = np.nan
            elif func == "covar_pop":
                result = None if len(elems) == 0 else elems_y.cov(elems_x, ddof=0)
            elif func == "covar_samp":
                result = None if len(elems) <= 1 else elems_y.cov(elems_x)
            elif func == "corr":
                result = None if len(elems) <= 1 else elems_y.corr(elems_x)
        if use_nans:
            if result is None:
                to_null.append(i)
                L.append(0.0)
            elif result is np.nan:
                to_nan.append(i)
                L.append(0.0)
            else:
                L.append(result)
        else:
            L.append(result)
    if use_nans:
        return nullable_float_arr_maker(L, to_null, to_nan)
    else:
        return pd.Series(L)


@pytest.fixture
def window_kernel_numeric_data():
    return {
        "null": pd.Series([None] * 5, dtype=pd.Int32Dtype()),
        "int32": pd.Series(
            [
                None if math.cos(i**2) < -0.5 else int(2 / math.tan(i + 1))
                for i in range(30)
            ],
            dtype=pd.Int32Dtype(),
        ),
        "float64_nonan": pd.Series(np.arange(500, dtype=np.float64)),
        "float64_nan": nullable_float_arr_maker(
            list(range(100)),
            [i**2 for i in range(10)],
            [i**3 + 5 for i in range(5)],
        ),
    }


@pytest.fixture
def window_kernel_all_types_data():
    return {
        "null": pd.Series([None] * 5, dtype=pd.UInt8Dtype()),
        "uint8": pd.Series(
            [None if "0" in str(i) else (i**2) % 17 for i in range(100)],
            dtype=pd.UInt8Dtype(),
        ),
        "float64_nonan": nullable_float_arr_maker(
            [(((13 + i) ** 2) % 41) ** 0.4 for i in range(100)],
            [i for i in range(100) if "11" in str(i)],
            [-1],
        ),
        "float64_nan": nullable_float_arr_maker(
            [float(i % 13) for i in range(100)],
            [i**2 for i in range(10)],
            [i**3 + 5 for i in range(5)],
        ),
        "string": pd.Series(
            [chr(65 + (i**2) % 5) if i % 7 < 6 else None for i in range(250)]
        ),
        "binary": pd.Series(
            [
                None
                if "1" in str(i) and "2" in str(i)
                else bytes(bin((i**2) % 47), encoding="utf-8")
                for i in range(450)
            ]
        ),
        "timestamp": pd.Series(
            [
                None
                if days is None
                else pd.Timestamp("2026-1-1") - pd.Timedelta(days=days)
                for tup in zip(
                    [(i + 10) ** 2 for i in range(50)],
                    [int(1.2**i) for i in range(50)],
                    [None, 100, 100, None, 256] * 10,
                    [(2**i) % 10000 for i in range(50)],
                    [(i**3) % 10000 for i in range(50)],
                    [7 ** int(math.log2(i + 1)) for i in range(50)],
                )
                for days in tup
            ],
            dtype="datetime64[ns]",
        ),
        "date": pd.Series(
            [
                None if days is None else datetime.date.fromordinal(738705 - days)
                for tup in zip(
                    [(i**3) % 9000 for i in range(50)],
                    [(i + 7) ** 2 for i in range(50)],
                    [6 ** int(math.log2(i + 1)) for i in range(50)],
                    [None, 100, 100, None, 256] * 10,
                    [(2**i) % 9000 for i in range(50)],
                    [int(1.19**i) for i in range(50)],
                )
                for days in tup
            ]
        ),
        "time": pd.Series(
            [
                None
                if ns is None
                else bodo.types.Time(0, 0, 0, nanosecond=123456789 * ns, precision=9)
                for tup in zip(
                    [(i**3) % 900000 for i in range(50)],
                    [(i + 15) ** 2 for i in range(50)],
                    [6 ** int(math.log2(i + 1)) for i in range(50)],
                    [None, 10**12, 10**13, None, 10**13] * 10,
                    [(3**i) % 900000 for i in range(50)],
                    [int(2**i) for i in range(50)],
                )
                for ns in tup
            ]
        ),
    }


@pytest.fixture
def window_kernel_two_arg_data():
    return {
        "null": (
            pd.Series([None] * 13, dtype=pd.Int32Dtype()),
            pd.Series([None] * 13, dtype=pd.Int32Dtype()),
        ),
        "int32": (
            pd.Series(
                [None if "0" in str(i) else (i**2) % 17 for i in range(400)],
                dtype=pd.Int32Dtype(),
            ),
            pd.Series(
                [None if "9" in str(i) else (i**3) % 23 for i in range(400)],
                dtype=pd.Int32Dtype(),
            ),
        ),
        "float64_nonan": (
            pd.Series(
                [
                    None if "11" in str(i) else (((13 + i) ** 2) % 12345) ** 0.7 - 250
                    for i in range(900)
                ]
            ),
            pd.Series(
                [
                    None if "12" in str(i) else (((100 + i) ** 3) % 54321) ** 0.7 - 1234
                    for i in range(900)
                ]
            ),
        ),
        "float64_nan": (
            nullable_float_arr_maker(
                [math.tan(i) for i in range(100)],
                [(i**2) % 100 for i in range(15)],
                [(i**3) % 100 for i in range(15)],
            ),
            nullable_float_arr_maker(
                [1 / (math.tan(i) + 0.1) for i in range(100)],
                [(i**4) % 100 for i in range(15)],
                [(i**5) % 100 for i in range(15)],
            ),
        ),
    }


@pytest.mark.parametrize(
    ["dataset", "lower_bound", "upper_bound"],
    [
        pytest.param("null", -10000, 0, id="null-prefix"),
        pytest.param("null", -10000, 10000, id="null-entire_window"),
        pytest.param("int32", -10000, 0, id="int32-prefix"),
        pytest.param(
            "int32", 1, 10000, id="int32-suffix_exclusive", marks=pytest.mark.slow
        ),
        pytest.param("int32", -5, -1, id="int32-lagging_5"),
        pytest.param("int32", -10000, 10000, id="int32-entire_window"),
        pytest.param("int32", 10000, 5000, id="int32-too_large"),
        pytest.param("float64_nonan", -10000, 0, id="float64_nonan-prefix"),
        pytest.param(
            "float64_nonan", 0, 0, id="float64_nonan-current", marks=pytest.mark.slow
        ),
        pytest.param("float64_nonan", -1, 1, id="float64_nonan-rolling_3"),
        pytest.param(
            "float64_nonan",
            100,
            400,
            id="float64_nonan-leading_300",
            marks=pytest.mark.slow,
        ),
        pytest.param("float64_nonan", -10000, 10000, id="float64_nonan-entire_window"),
        pytest.param(
            "float64_nonan", 3, -3, id="float64_nonan-backward", marks=pytest.mark.slow
        ),
        pytest.param("float64_nan", -10000, 0, id="float64_nan-prefix"),
        pytest.param(
            "float64_nan", -3, 3, id="float64_nan-rolling_7", marks=pytest.mark.slow
        ),
        pytest.param(
            "float64_nan",
            -10000,
            10000,
            id="float64_nan-entire_window",
        ),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "median",
        "sum",
        "count",
        "count(*)",
        "avg",
        "ratio_to_report",
        "var_pop",
        "var_samp",
        "stddev_pop",
        "stddev_samp",
        "boolor",
        "booland",
        "boolxor",
        "skew",
        "kurtosis",
        "bitor_agg",
        "bitand_agg",
        "bitxor_agg",
    ],
)
@pytest.mark.skip("TODO: fix for Pandas 3")
def test_windowed_kernels_numeric(
    func,
    window_kernel_numeric_data,
    dataset,
    lower_bound,
    upper_bound,
    memory_leak_check,
):
    def impl_sum(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_sum(S, lower, upper))

    def impl_count(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_count(S, lower, upper))

    def impl_count_star(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_count_star(len(S), lower, upper))

    def impl_avg(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_avg(S, lower, upper))

    def impl_median(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_median(S, lower, upper))

    def impl_ratio_to_report(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_ratio_to_report(S, lower, upper))

    def impl_var_pop(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_var_pop(S, lower, upper))

    def impl_var_samp(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_var_samp(S, lower, upper))

    def impl_stddev_pop(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_stddev_pop(S, lower, upper))

    def impl_stddev_samp(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_stddev_samp(S, lower, upper))

    def impl_boolor(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_boolor(S, lower, upper))

    def impl_booland(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_booland(S, lower, upper))

    def impl_boolxor(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_boolxor(S, lower, upper))

    def impl_skew(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_skew(S, lower, upper))

    def impl_kurtosis(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_kurtosis(S, lower, upper))

    def impl_bitor_agg(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_bitor_agg(S, lower, upper))

    def impl_bitand_agg(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_bitand_agg(S, lower, upper))

    def impl_bitxor_agg(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_bitxor_agg(S, lower, upper))

    S = window_kernel_numeric_data[dataset]

    implementations = {
        "sum": impl_sum,
        "count": impl_count,
        "count(*)": impl_count_star,
        "avg": impl_avg,
        "median": impl_median,
        "ratio_to_report": impl_ratio_to_report,
        "var_pop": impl_var_pop,
        "var_samp": impl_var_samp,
        "stddev_pop": impl_stddev_pop,
        "stddev_samp": impl_stddev_samp,
        "boolor": impl_boolor,
        "booland": impl_booland,
        "boolxor": impl_boolxor,
        "skew": impl_skew,
        "kurtosis": impl_kurtosis,
        "bitor_agg": impl_bitor_agg,
        "bitand_agg": impl_bitand_agg,
        "bitxor_agg": impl_bitxor_agg,
    }
    impl = implementations[func]

    # bitXXX_agg functions do not allow NaN
    if dataset == "float64_nan" and func.startswith("bit"):
        return

    check_func(
        impl,
        (S, lower_bound, upper_bound),
        py_output=window_refsol(
            S, lower_bound, upper_bound, func, dataset == "float64_nan"
        ),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    ["lower_bound", "upper_bound"],
    [
        pytest.param(-10000, 0, id="prefix"),
        pytest.param(-10000, 10000, id="entire_window"),
        pytest.param(1, 10000, id="suffix_exclusive"),
        pytest.param(-2, 2, id="rolling 5"),
        pytest.param(3, -3, id="backward"),
    ],
)
def test_windowed_count_if(
    lower_bound,
    upper_bound,
    memory_leak_check,
):
    """Tests the bodosql array kernel `windowed_count_if`. The `windowed_count_if` is an optimized implementation
    of the window function version of count_if, utilizing gen_windowed.

    Args:
        lower_bound (int): The lower bound of the window being tested, where 0 is the current row,
            negative values are preceding rows, and positive values are following rows.
        upper_bound (int): The upper bound of the window being tested, with the same logic as above.
        memory_leak_check (): Fixture, see `conftest.py`.
    """

    def impl(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_count_if(S, lower, upper))

    # Generate seeded random series of None | True | False values for count_if to operate on.
    random.seed(42)
    S = pd.Series(
        [random.choice([None, True, False]) for i in range(30)], dtype=pd.BooleanDtype()
    )

    check_func(
        impl,
        (S, lower_bound, upper_bound),
        py_output=window_refsol(S, lower_bound, upper_bound, "count_if"),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    ["dataset", "lower_bound", "upper_bound"],
    [
        pytest.param("null", -1000, 0, id="null-prefix"),
        pytest.param("null", -1000, 1000, id="null-entire_window"),
        pytest.param("null", 0, 0, id="null-current", marks=pytest.mark.slow),
        pytest.param("null", 3, -3, id="null-backward", marks=pytest.mark.slow),
        pytest.param("uint8", -1000, 0, id="uint8-prefix"),
        pytest.param("uint8", -1000, 0, id="uint8-suffix_exclusive"),
        pytest.param("uint8", -1000, 1000, id="uint8-entire_window"),
        pytest.param("uint8", 0, 0, id="uint8-current"),
        pytest.param("uint8", -1000, -700, id="uint8-too_small"),
        pytest.param("uint8", 3, -3, id="uint8-backward"),
        pytest.param(
            "float64_nonan", -1000, 0, id="float64_nonan-prefix", marks=pytest.mark.slow
        ),
        pytest.param(
            "float64_nonan",
            -1000,
            1000,
            id="float64_nonan-entire_window",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "float64_nonan",
            1000,
            7000,
            id="float64_nonan-too_large",
            marks=pytest.mark.slow,
        ),
        pytest.param("float64_nan", -1000, 0, id="float64_nan-prefix"),
        pytest.param(
            "float64_nan",
            -1000,
            1000,
            id="float64_nan-entire_window",
        ),
        pytest.param("string", -1000, 0, id="string-suffix_exclusive"),
        pytest.param("string", -1000, 1000, id="string-entire_window"),
        pytest.param("string", -20, 20, id="string-rolling_41", marks=pytest.mark.slow),
        pytest.param("string", 1, 3, id="string-leading_3"),
        pytest.param("string", 3, -3, id="string-backward", marks=pytest.mark.slow),
        pytest.param("binary", -1000, 0, id="binary-prefix", marks=pytest.mark.slow),
        pytest.param("binary", -1000, 1000, id="binary-entire_window"),
        pytest.param("binary", 0, 0, id="binary-current", marks=pytest.mark.slow),
        pytest.param("binary", 1, 3, id="binary-leading_3"),
        pytest.param("binary", 3, -3, id="binary-backward", marks=pytest.mark.slow),
        pytest.param(
            "timestamp",
            -1000,
            0,
            id="timestamp-suffix_exclusive",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "timestamp",
            -1000,
            1000,
            id="timestamp-entire_window",
            marks=pytest.mark.slow,
        ),
        pytest.param("timestamp", 0, 0, id="timestamp-current", marks=pytest.mark.slow),
        pytest.param(
            "timestamp", -20, 20, id="timestamp-rolling_41", marks=pytest.mark.slow
        ),
        pytest.param("timestamp", -13, -1, id="timestamp-lagging_13"),
        pytest.param(
            "timestamp", 3, -3, id="timestamp-backward", marks=pytest.mark.slow
        ),
        pytest.param("date", -1000, 0, id="date-prefix"),
        pytest.param("date", -10, -1, id="date-leading_10"),
        pytest.param("date", -1000, 1000, id="date-entire_window"),
        pytest.param("date", 7, -7, id="date-backward"),
        pytest.param("time", 0, 1000, id="time-suffix"),
        pytest.param("time", -3, 3, id="time-rolling_7"),
        pytest.param("time", 5, 30, id="time-leading_26"),
        pytest.param("time", -1000, 1000, id="time-entire_window"),
        pytest.param("time", 1000, 2000, id="time-too_large"),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "mode",
        "min",
        "max",
    ],
)
@pytest_mark_one_rank
@pytest.mark.skip("TODO: fix for Pandas 3")
def test_windowed_non_numeric(
    func,
    dataset,
    window_kernel_all_types_data,
    lower_bound,
    upper_bound,
    memory_leak_check,
):
    def impl1(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_mode(S, lower, upper))

    def impl2(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_min(S, lower, upper))

    def impl3(S, lower, upper):
        return pd.Series(bodosql.kernels.windowed_max(S, lower, upper))

    data = window_kernel_all_types_data[dataset]

    implementations = {
        "mode": impl1,
        "min": impl2,
        "max": impl3,
    }
    impl = implementations[func]

    if func == "mode":
        answer = window_refsol_mode(
            data, lower_bound, upper_bound, dataset == "float64_nan"
        )
    else:
        answer = window_refsol(
            data, lower_bound, upper_bound, func, dataset == "float64_nan"
        )

    check_func(
        impl,
        (data, lower_bound, upper_bound),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    ["dataset", "lower_bound", "upper_bound"],
    [
        pytest.param("null", -1000, 0, id="null-prefix"),
        pytest.param("null", -1000, 1000, id="null-entire_window"),
        pytest.param("int32", -1000, 0, id="int32-prefix"),
        pytest.param(
            "int32", 1, 1000, id="int32-suffix_exclusive", marks=pytest.mark.slow
        ),
        pytest.param("int32", -1000, 1000, id="int32-entire_window"),
        pytest.param("int32", -2, 0, id="int32-lagging_3"),
        pytest.param("int32", 1, 70, id="int32-leading_70", marks=pytest.mark.slow),
        pytest.param("int32", -2000, -1000, id="int32-too_small"),
        pytest.param("float64_nonan", -1000, 0, id="float64_nonan-prefix"),
        pytest.param("float64_nonan", -1000, 1000, id="float64_nonan-entire_window"),
        pytest.param("float64_nonan", 0, 0, id="float64_nonan-current"),
        pytest.param(
            "float64_nonan",
            -50,
            50,
            id="float64_nonan-rolling_101",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "float64_nonan",
            1000,
            2000,
            id="float64_nonan-too_large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "float64_nonan", 3, -3, id="float64_nonan-backward", marks=pytest.mark.slow
        ),
        pytest.param("float64_nan", -1000, 0, id="float64_nan-prefix"),
        pytest.param(
            "float64_nan",
            -1000,
            1000,
            id="float64_nan-entire_window",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "float64_nan", -2, 2, id="float64_nan-rolling-5", marks=pytest.mark.slow
        ),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "covar_pop",
        "covar_samp",
        "corr",
    ],
)
def test_windowed_kernels_two_arg(
    func,
    dataset,
    window_kernel_two_arg_data,
    lower_bound,
    upper_bound,
    memory_leak_check,
):
    def impl1(Y, X, lower, upper):
        return pd.Series(bodosql.kernels.windowed_covar_pop(Y, X, lower, upper))

    def impl2(Y, X, lower, upper):
        return pd.Series(bodosql.kernels.windowed_covar_samp(Y, X, lower, upper))

    def impl3(Y, X, lower, upper):
        return pd.Series(bodosql.kernels.windowed_corr(Y, X, lower, upper))

    Y, X = window_kernel_two_arg_data[dataset]

    implementations = {
        "covar_pop": impl1,
        "covar_samp": impl2,
        "corr": impl3,
    }
    impl = implementations[func]

    check_func(
        impl,
        (Y, X, lower_bound, upper_bound),
        py_output=window_refsol_double(
            Y, X, lower_bound, upper_bound, func, dataset == "float64_nan"
        ),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
        is_out_distributed=False,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "arr, q, answer",
    [
        pytest.param(
            pd.Series([10, 20, 30, 40, 50] * 100, dtype=pd.Float64Dtype()),
            0.5,
            pd.Series([30.0] * 500, dtype=pd.Float64Dtype()),
            id="0.5-no_null",
        ),
        pytest.param(
            pd.Series(
                [30, 20, 10, None, 300, None, None, None] * 100, dtype=pd.Float64Dtype()
            ),
            0.5,
            pd.Series([25.0] * 800, dtype=pd.Float64Dtype()),
            id="0.5-some_null",
        ),
        pytest.param(
            pd.Series([None] * 5000, dtype=pd.Float64Dtype()),
            0.5,
            pd.Series([None] * 5000, dtype=pd.Float64Dtype()),
            id="0.5-all_null",
            marks=pytest.mark.skip(
                reason="Pandas doesn't support np.nan data in nullable float arrays"
            ),
        ),
        pytest.param(
            pd.Series(list(range(1000)), dtype=pd.Float64Dtype()),
            0.01,
            pd.Series([10.0] * 1000, dtype=pd.Float64Dtype()),
            id="0.01-no_null",
        ),
        pytest.param(
            pd.Series(
                [None if i % 2 == 0 else i**2 for i in range(500)],
                dtype=pd.Float64Dtype(),
            ),
            0.79,
            pd.Series([140251.96] * 500, dtype=pd.Float64Dtype()),
            id="0.79-some_null",
        ),
    ],
)
def test_windowed_approx_percentile(arr, q, answer, memory_leak_check):
    def impl(arr, q):
        return pd.Series(bodosql.kernels.windowed_approx_percentile(arr, q))

    check_func(
        impl,
        (arr, q),
        py_output=answer,
        only_seq=True,
        check_dtype=False,
        reset_index=True,
        atol=0.01,
        rtol=0.4,
    )


def test_window_dict_min_max(memory_leak_check):
    """Tests windowed_min and windowed_max on dictionary encoded arrays"""

    def impl(arr):
        return (
            pd.Series(bodosql.kernels.windowed_min(arr, -2, 2)),
            pd.Series(bodosql.kernels.windowed_max(arr, -2, 2)),
        )

    arr = pa.array(
        [
            "alpha",
            "beta",
            "gamma",
            None,
            "delta",
            "epsilon",
            "",
            "XYZ",
            "alpha",
            None,
            "delta",
            "beta",
            "",
            "",
            "",
            None,
            "XYZ",
        ],
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    answer = (
        pd.Series(
            [
                "alpha",
                "alpha",
                "alpha",
                "beta",
                "",
                "",
                "",
                "",
                "",
                "XYZ",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        pd.Series(
            [
                "gamma",
                "gamma",
                "gamma",
                "gamma",
                "gamma",
                "epsilon",
                "epsilon",
                "epsilon",
                "delta",
                "delta",
                "delta",
                "delta",
                "delta",
                "beta",
                "XYZ",
                "XYZ",
                "XYZ",
            ]
        ),
    )

    check_func(
        impl,
        (arr,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )


def test_string_min_max(memory_leak_check):
    def impl_max_arr(arr):
        return bodosql.kernels.window_agg_array_kernels.str_arr_max(arr)

    def impl_min_arr(arr):
        return bodosql.kernels.window_agg_array_kernels.str_arr_min(arr)

    def impl_max_series(arr):
        return bodosql.kernels.window_agg_array_kernels.str_arr_max(arr.values)

    def impl_min_series(arr):
        return bodosql.kernels.window_agg_array_kernels.str_arr_min(arr.values)

    args = [
        (pd.Series(["A", "B", "C", "D", "E"]), "E", "A", False),
        (pd.Series(["I", "e", "A", "o", "U"]), "o", "A", False),
        (pd.Series([None, "42", "720", "1024", None]), "720", "1024", False),
        (pd.Series(["a", "B", None, "!", ""]), "a", "", False),
        (pd.Series(["", "⚕¶", "©¡™", "ᖴ∆π", "🐍∫ßå"]), "🐍∫ßå", "", False),
        (pd.Series([b"A", b"B", b"C", b"D", b"E"]), b"E", b"A", False),
        (pd.Series([b"I", b"e", b"A", b"o", b"U"]), b"o", b"A", False),
        (pd.Series([None, b"42", b"720", b"1024", None]), b"720", b"1024", False),
        (pd.Series([b"a", b"B", None, b"!", b""]), b"a", b"", False),
    ]
    for data, data_max, data_min, dict_test in args:
        if dict_test:
            check_func(impl_max_arr, (data,), py_output=data_max, only_seq=True)
            check_func(impl_min_arr, (data,), py_output=data_min, only_seq=True)
        else:
            check_func(impl_max_series, (data,), py_output=data_max, only_seq=True)
            check_func(impl_min_series, (data,), py_output=data_min, only_seq=True)


@pytest.mark.parametrize(
    "value_data, value_dtype",
    [
        pytest.param(
            pd.Series(
                [None if i % 2 == 0 else 2**i for i in range(10)],
                dtype=pd.Int64Dtype(),
            ),
            pa.int64(),
            id="int64",
        ),
        pytest.param(
            pd.Series([None if i % 5 == 4 else chr(65 + i) * i for i in range(10)]),
            pa.large_string(),
            id="string",
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 4 == 3
                    else [datetime.date(2023, 12, 21), None, datetime.date(2022, 7, 4)][
                        : i % 3
                    ]
                    for i in range(10)
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.date32())),
            ),
            pa.large_list(pa.date32()),
            id="array_date",
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 5 == 4
                    else {
                        "A": None if i % 4 == 3 else i,
                        "B": None if i == 7 else [True, False, None, True][: i % 4],
                    }
                    for i in range(10)
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.float64()),
                            pa.field("B", pa.large_list(pa.bool_())),
                        ]
                    )
                ),
            ),
            pa.struct(
                [pa.field("A", pa.float64()), pa.field("B", pa.large_list(pa.bool_()))]
            ),
            id="struct-float_list_bool",
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 4 == 1
                    else {
                        chr(j): None if j % 2 == 0 else j
                        for j in range(65 + i, 65 + i + i % 3)
                    }
                    for i in range(10)
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int8())),
            ),
            pa.map_(pa.string(), pa.int8()),
            id="map-int",
        ),
    ],
)
def test_window_object_agg(value_data, value_dtype, memory_leak_check):
    def impl(K, V):
        return pd.DataFrame({"res": bodosql.kernels.windowed_object_agg(K, V)})

    key_data = pd.Series([None if c in "AEI" else c for c in "ABCDEFGHIJ"])
    json_res = {}
    keep = pd.notna(key_data) & pd.notna(value_data)
    for i in range(len(key_data)):
        if keep[i]:
            json_res[key_data.iloc[i]] = value_data.iloc[i]
    answer = pd.DataFrame(
        {
            "res": pd.array(
                [json_res] * 10, dtype=pd.ArrowDtype(pa.map_(pa.string(), value_dtype))
            )
        }
    )

    check_func(
        impl,
        (key_data, value_data),
        py_output=answer,
        check_dtype=False,
        only_seq=True,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "arr, lower_bound, upper_bound, answer",
    [
        pytest.param(
            pd.array(
                [None if i % 4 < 2 else f"{i}.{i % 10}" for i in range(10)],
                dtype=pd.ArrowDtype(pa.decimal128(38, 1)),
            ),
            -500,
            500,
            pd.Series(["19.8"] * 10),
            id="some_null-partition_wide",
        ),
        pytest.param(
            pd.array([None] * 40, dtype=pd.ArrowDtype(pa.decimal128(38, 1))),
            -500,
            500,
            pd.Series([None] * 40, dtype=pd.ArrowDtype(pa.decimal128(38, 1))),
            id="all_null-partition_wide",
        ),
        pytest.param(
            pd.array(
                [None if i % 4 < 2 else f"{i}.{i % 10}" for i in range(10)],
                dtype=pd.ArrowDtype(pa.decimal128(38, 1)),
            ),
            -500,
            0,
            pd.Series(
                [None, None, "2.2", "5.5", "5.5", "5.5", "12.1", "19.8", "19.8", "19.8"]
            ),
            id="some_null-prefix",
        ),
    ],
)
def test_decimal_window_sum(arr, lower_bound, upper_bound, answer, memory_leak_check):
    def impl(arr, lower_bound, upper):
        return pd.Series(
            bodosql.kernels.to_char(
                bodosql.kernels.windowed_sum(pd.Series(arr), lower_bound, upper_bound),
                is_scalar=True,
            )
        )

    check_func(
        impl,
        (arr, lower_bound, upper_bound),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
        is_out_distributed=False,
    )
