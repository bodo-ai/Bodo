# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Measure performance of various operations that uses the unordered_map/unordered_set
"""
import pandas as pd
import numpy as np
import bodo
from bodo.tests.utils import check_timing_func
import random
import time
import string
import pytest


def get_random_int_numpy_column(n, len_siz):
    """Function returning a random numpy array with NaN occurring randomly"""
    elist = []
    for _ in range(n):
        val = random.randint(1, len_siz)
        if val == 1:
            val = np.nan
        elist.append(val)
    return elist


def get_random_array_string(n, len_siz):
    """Returns a random array of strings"""
    elist = []
    for _ in range(n):
        k = random.randint(1, len_siz)
        if k == 1:
            val = np.nan
        else:
            val = "".join(random.choices(string.ascii_uppercase, k=k))
        elist.append(val)
    return elist


def get_random_nullable_int_array(n, len_siz):
    """Returns an array of random values"""
    elist = []
    for _ in range(n):
        val = random.randint(1, len_siz)
        if val == 1:
            val = None
        elist.append(val)
    return pd.array(elist, dtype="UInt16")


def get_random_dataframe(list_names, func, args):
    """Returned a random dataframe with the names on input and the chosen input function"""
    list_columns = {}
    for eName in list_names:
        list_columns[eName] = func(*args)
    return pd.DataFrame(list_columns)


@pytest.mark.parametrize(
    "n, len_siz",
    [
        (10000, 10),
        (100000, 100),
        (200000, 1000),
        (400000, 1000),
        (1000000, 1000),
        (2000000, 1000),
        (4000000, 2000),
    ],
)
def test_dd_numpy_int_nan(n, len_siz):
    """
    Test drop_duplicates(): 2 columns of random numpy arrays
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return True

    random.seed(5)
    df1 = get_random_dataframe(["A", "B"], get_random_int_numpy_column, (n, len_siz))
    check_timing_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_siz",
    [(10000, 10), (10000, 10), (100000, 100), (100000, 100), (200000, 1000)],
)
def test_dd_strings(n, len_siz):
    """
    Test drop_duplicates(): random array of strings
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return True

    random.seed(5)
    df1 = get_random_dataframe(["A", "B"], get_random_array_string, (n, len_siz))
    check_timing_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_siz",
    [
        (10000, 10),
        (10000, 10),
        (10000, 10),
        (100000, 100),
        (100000, 100),
        (100000, 100),
    ],
)
def test_dd_nullable_int_bool(n, len_siz):
    """
    Test drop_duplicates(): 2 arrays of random nullable_int_bool arrays
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return True

    random.seed(5)
    df1 = get_random_dataframe(["A", "B"], get_random_nullable_int_array, (n, len_siz))
    check_timing_func(test_impl, (df1,))


@pytest.mark.parametrize("n, len_siz", [(10000, 10), (100000, 100)])
def test_join_nullable_int(n, len_siz):
    """
    Test inner and outer merge(): 2 arrays of random nullable_int_bool arrays
    """

    def test_impl1(df1, df2):
        df3 = df1.merge(df2, how="inner", on=["A", "B"])
        return True

    def test_impl2(df1, df2):
        df3 = df1.merge(df2, how="outer", on=["A", "B"])
        return True

    random.seed(5)
    df1 = get_random_dataframe(["A", "B"], get_random_nullable_int_array, (n, len_siz))
    df2 = get_random_dataframe(["A", "B"], get_random_nullable_int_array, (n, len_siz))
    check_timing_func(test_impl1, (df1, df2))
    check_timing_func(test_impl2, (df1, df2))


@pytest.mark.parametrize(
    "n, len_siz", [(10000, 10), (100000, 100), (1000000, 1000), (1000000, 10000)]
)
def test_sort_nullable_int(n, len_siz):
    """
    Test sort_values(): two arrays of random nullable_innt_bool arrays
    """

    def test_impl(df1):
        df2 = df1.sort_values(by=["A", "B"])
        return True

    random.seed(5)
    df1 = get_random_dataframe(["A", "B"], get_random_nullable_int_array, (n, len_siz))
    check_timing_func(test_impl, (df1,))
