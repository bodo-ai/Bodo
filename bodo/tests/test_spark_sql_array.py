# Copyright (C) 2020 Bodo Inc. All rights reserved.
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
                    ["hi", "its", " me "],
                    ["who, ", "are", " you"],
                ]
                * 20,
                "B": [
                    ["hi", "iTs", " you "],
                    ["who", "are", " you "],
                ]
                * 20,
            }
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.skip(reason="known int array bug #1511")
def test_array_contains(dataframe_val):
    def test_impl_float(df):
        return df.A.map(lambda a: 5.1 in list(a))

    def test_impl_int(df):
        return df.A.map(lambda a: 1 in list(a))

    def test_impl_str(df):
        return df.A.map(lambda a: "you" in list(a))

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        test_impl = test_impl_float
    elif isinstance(df.A[0][0], np.int64):
        test_impl = test_impl_int
    elif isinstance(df.A[0][0], str):
        test_impl = test_impl_str

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="sort_outputs doesn't work with array elems in series #1771")
def test_array_distinct(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: np.unique(x))

    df = dataframe_val
    check_func(test_impl, (df,), sort_output=True)


@pytest.mark.skip(reason="Map operation not yet supported #1534")
def test_array_except(dataframe_val):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.setdiff1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1534")
def test_array_intersect(dataframe_val):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.intersect1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Numba bug with no support for str.join(numpy_array) #1571")
def test_array_join(dataframe_val):
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


@pytest.mark.skip(reason="Map operation not yet supported #1569")
def test_array_max(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: np.max(x))

    df = dataframe_val
    if not isinstance(df.A[0][0], str):
        check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1569")
def test_array_min(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: np.min(x))

    df = dataframe_val
    if not isinstance(df.A[0][0], str):
        check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1570")
def test_array_position(dataframe_val):
    def test_impl_float(df):
        return df.A.map(lambda x: np.append(np.where(x == 3.31111)[0], -1)[0])

    def test_impl_int(df):
        return df.A.map(lambda x: np.append(np.where(x == 1)[0], -1)[0])

    def test_impl_str(df):
        return df.A.map(lambda x: np.append(np.where(x == "are")[0], -1)[0])

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        test_impl = test_impl_float
    elif isinstance(df.A[0][0], np.int64):
        test_impl = test_impl_int
    elif isinstance(df.A[0][0], str):
        test_impl = test_impl_str

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1534")
def test_array_remove(dataframe_val):
    def test_impl_float(df):
        return df.A.map(lambda x: np.setdiff1d(x, np.array([5.1])))

    def test_impl_int(df):
        return df.A.map(lambda x: np.setdiff1d(x, np.array([3])))

    def test_impl_str(df):
        return df.A.map(lambda x: np.setdiff1d(x, np.array(["hi"])))

    df = dataframe_val
    if isinstance(df.A[0][0], np.float64):
        test_impl = test_impl_float
    elif isinstance(df.A[0][0], np.int64):
        test_impl = test_impl_int
    elif isinstance(df.A[0][0], str):
        test_impl = test_impl_str

    check_func(test_impl, (df,))


def test_array_repeat(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: np.repeat(x, 3))

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1548")
def test_array_sort(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: np.sort(x))

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1534")
def test_array_union(dataframe_val):
    def test_impl(df):
        return df[["A", "B"]].apply(lambda x: np.union1d(x[0], x[1]), axis=1)

    df = dataframe_val
    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Map operation not yet supported #1534")
def test_arrays_overlap(dataframe_val):
    def test_impl(df):
        return df[["A", "B"]].apply(
            lambda x: len(np.intersect1d(x[0], x[1])) > 0, axis=1
        )

    df = dataframe_val
    check_func(test_impl, (df,))
