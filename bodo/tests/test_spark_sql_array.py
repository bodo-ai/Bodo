# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on arrays as
column elements.

Test names refer to the names of the spark function they map to.
"""

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [
                    np.array([1.1234, np.nan, 3.31111]),
                    np.array([2.1334, 5.1, -6.3]),
                ]
                * 20,
                "B": [np.array([1.123, 7.2]), np.array([3.3111, 5.1, -2, 4.7])] * 20,
            }
        ),
        pd.DataFrame(
            {
                "A": [np.array([1, 2, 3]), np.array([2, 5, -6])] * 20,
                "B": [np.array([0, -1, 2]), np.array([4, -1, -5])] * 20,
            }
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        np.array(["hi", "its", " me "]),
                        np.array(["who, ", "are", " you"]),
                    ]
                    * 20,
                    "B": [
                        np.array(["hi", "iTs", " you "]),
                        np.array(["who", "are", " you "]),
                    ]
                    * 20,
                }
            ),
            marks=pytest.mark.skip,
        ),
        pd.DataFrame(
            {
                "A": [
                    pd.array([False, True, True, False]),
                    pd.array([False, False, True, True]),
                ]
                * 20,
                "B": [
                    pd.array([False, True, True]),
                    pd.array([False, False]),
                ]
                * 20,
            }
        ),
        pd.DataFrame(
            {
                "A": [pd.array([1, 2, 3]), pd.array([2, 5, -6])] * 20,
                "B": [pd.array([0, -1, 2]), pd.array([4, -1, -5])] * 20,
            }
        ),
        pd.DataFrame(
            {
                "A": [
                    pd.array(["hi", "its", " me "]),
                    pd.array(["who, ", "are", " you"]),
                ]
                * 20,
                "B": [
                    pd.array(["hi", "iTs", " you "]),
                    pd.array(["who", "are", " you "]),
                ]
                * 20,
            }
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.skip(reason="Missing support for in #1851")
def test_array_contains(dataframe_val, memory_leak_check):
    def test_impl_float(df):
        return df.A.map(lambda a: 5.1 in a)

    def test_impl_int(df):
        return df.A.map(lambda a: 1 in a)

    def test_impl_str(df):
        return df.A.map(lambda a: "you" in a)

    def test_impl_bool(df):
        return df.A.map(lambda a: True in a)

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        test_impl = test_impl_float
    elif isinstance(df.A[0][0], np.int64):
        test_impl = test_impl_int
    elif isinstance(df.A[0][0], str):
        test_impl = test_impl_str
    elif isinstance(df.A[0][0], (bool, np.bool_)):
        test_impl = test_impl_bool

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="sort_outputs doesn't work with array elems in series #1771")
def test_array_distinct(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.unique(x))

    df = dataframe_val
    check_func(test_impl, (df,), sort_output=True)


@pytest.mark.skip("Issue with our string array #1925")
def test_array_except(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.setdiff1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,), dist_test=False)


@pytest.mark.skip("Issue with our string array #1925")
def test_array_intersect(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.intersect1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,), dist_test=False)


@pytest.mark.slow
def test_array_join(dataframe_val, memory_leak_check):
    def test_impl_comma(df):
        return df.A.map(lambda x: ",".join(x))

    def test_impl_empty(df):
        return df.A.map(lambda x: "".join(x))

    def test_impl_space(df):
        return df.A.map(lambda x: " ".join(x))

    df = dataframe_val

    # Only test on str values
    if isinstance(df.A[0][0], str):
        check_func(test_impl_comma, (df,))
        check_func(test_impl_empty, (df,))
        check_func(test_impl_space, (df,))


@pytest.mark.slow
def test_array_max(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.nanmax(x))

    df = dataframe_val
    # np.max is not supported on pd.array in pandas
    if not isinstance(df.A[0][0], str) and not isinstance(
        df.A[0], (pd.arrays.BooleanArray, pd.arrays.IntegerArray)
    ):
        check_func(test_impl, (df,))


@pytest.mark.slow
def test_array_min(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.nanmin(x))

    df = dataframe_val
    # np.min is not supported on pd.array in pandas
    if not isinstance(df.A[0][0], str) and not isinstance(
        df.A[0], (pd.arrays.BooleanArray, pd.arrays.IntegerArray)
    ):
        check_func(test_impl, (df,))


@pytest.mark.slow
def test_array_position(dataframe_val, memory_leak_check):
    def test_impl_float(df):
        return df.A.map(lambda x: np.append(np.where(x == 3.31111)[0], -1)[0])

    def test_impl_int(df):
        return df.A.map(lambda x: np.append(np.where(x == 1)[0], -1)[0])

    def test_impl_str(df):
        return df.A.map(lambda x: np.append(np.where(x == "are")[0], -1)[0])

    def test_impl_bool(df):
        return df.A.map(lambda x: np.append(np.where(x == True)[0], -1)[0])

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        test_impl = test_impl_float
    elif isinstance(df.A[0][0], np.int64):
        test_impl = test_impl_int
    elif isinstance(df.A[0][0], str):
        test_impl = test_impl_str
    elif isinstance(df.A[0][0], (bool, np.bool_)):
        test_impl = test_impl_bool

    check_func(test_impl, (df,))


@pytest.mark.skip("Issue with our string array #1925")
def test_array_remove(dataframe_val, memory_leak_check):
    def test_impl(df, arr):
        return df.A.apply(lambda x, arr: np.setdiff1d(x, arr), arr=arr)

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        arr = np.array([5.1])
    elif isinstance(df.A.values, pd.arrays.IntegerArray):
        arr = pd.array([3])
    elif isinstance(df.A[0][0], np.int64):
        arr = np.array([3])
    elif isinstance(df.A[0][0], str):
        arr = np.array(["hi"], dtype=object)
    elif isinstance(df.A[0][0], (bool, np.bool_)):
        arr = pd.array([True])

    check_func(test_impl, (df, arr), dist_test=False)


@pytest.mark.slow
def test_array_repeat(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.repeat(x, 3))

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_array_sort(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: np.sort(x))

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_array_union(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.union1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip("Issue with our string array #1925")
def test_arrays_overlap(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B"]].apply(
            lambda x: len(np.intersect1d(x[0], x[1])) > 0, axis=1
        )

    df = dataframe_val
    check_func(test_impl, (df,), dist_test=False)


@pytest.mark.slow
def test_size(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: len(x))

    df = dataframe_val
    check_func(test_impl, (df,))


def test_concat_arrays(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.hstack(x), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_concat_arrays_heterogenous(memory_leak_check):
    """Tests that columns with different array types that can be merged succeed."""

    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.hstack(x), axis=1)

    df = pd.DataFrame(
        {
            "A": [
                np.array([1.1234, np.nan, 3.31111]),
                np.array([2.1334, 5.1, -6.3]),
            ]
            * 20,
            "B": [np.array([0, -1, 2]), np.array([4, -1, -5])] * 20,
        }
    )
    check_func(test_impl, (df,))
