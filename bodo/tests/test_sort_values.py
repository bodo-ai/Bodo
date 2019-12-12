# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test sort_values opration as called as df.sort_values()
   The C++ implementation uses the timsort which is a stable sort algorithm.
   Therefore, in the test we use mergesort, which guarantees that the equality
   tests can be made sensibly.
"""
import pandas as pd
import numpy as np
import bodo
import random
import string
import pytest
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoWarning, BodoError


def test_sort_values_1col():
    """
    Test sort_values(): with just one column
    """

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by="A", ascending=False, kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = []
        for i in range(n):
            eVal = i * i % 34
            eListA.append(eVal)
        return pd.DataFrame({"A": eListA})

    n = 100
    check_func(test_impl1, (get_quasi_random(n),), sort_output=False)
    check_func(test_impl2, (get_quasi_random(n),), sort_output=False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_sort_values_1col_np_array(dtype):
    """
    Test sort_values(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_dtype(n, dtype):
        eListA = np.array([0] * n, dtype=dtype)
        for i in range(n):
            eVal = i * i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})

    n = 100
    check_func(test_impl, (get_quasi_random_dtype(n, dtype),), sort_output=False)


def test_sort_values_2col():
    """
    Test sort_values(): with just one column
    """

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by=["A", "B"], kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = np.array([0] * n, dtype=np.uint64)
        eListB = np.array([0] * n, dtype=np.uint64)
        for i in range(n):
            eValA = i * i % 34
            eValB = i * i * i % 34
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})

    n = 100
    check_func(test_impl1, (get_quasi_random(n),), sort_output=False)
    check_func(test_impl2, (get_quasi_random(n),), sort_output=False)


@pytest.mark.parametrize(
    "dtype1, dtype2",
    [
        (np.int8, np.int16),
        (np.uint8, np.int32),
        (np.int16, np.float64),
        (np.uint16, np.float32),
    ],
)
def test_sort_values_2col_np_array(dtype1, dtype2):
    """
    Test sort_values(): with two columns, two types
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_dtype(n, dtype1, dtype2):
        eListA = np.array([0] * n, dtype=dtype1)
        eListB = np.array([0] * n, dtype=dtype2)
        for i in range(n):
            eValA = i * i % 34
            eValB = i * (i - 1) % 23
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})

    n = 1000
    check_func(
        test_impl, (get_quasi_random_dtype(n, dtype1, dtype2),), sort_output=False
    )


@pytest.mark.parametrize("n, len_str", [(1000, 2), (100, 1), (300, 2)])
def test_sort_values_strings_constant_length(n, len_str):
    """
    Test sort_values(): with 1 column and strings of constant length
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            val = "".join(random.choices(string.ascii_uppercase, k=len_str))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    check_func(test_impl, (get_random_strings_array(n, len_str),), sort_output=False)


@pytest.mark.parametrize("n, len_str", [(100, 30), (1000, 10), (10, 30)])
def test_sort_values_strings_variable_length(n, len_str):
    """
    Test sort_values(): with 1 column and strings of variable length all of character A.
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_var_length_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            k = random.randint(1, len_str)
            val = "A" * k
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    check_func(
        test_impl, (get_random_var_length_strings_array(n, len_str),), sort_output=False
    )


@pytest.mark.parametrize("n, len_str", [(100, 30), (1000, 10), (10, 30)])
def test_sort_values_strings(n, len_str):
    """
    Test sort_values(): with 1 column and strings of variable length and variable characters.
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            k = random.randint(1, len_str)
            val = "".join(random.choices(string.ascii_uppercase, k=k))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    check_func(test_impl, (get_random_strings_array(n, len_str),), sort_output=False)


# ------------------------------ error checking ------------------------------ #


df = pd.DataFrame({"A": [-1, 3, -3, 0, -1], "B": ["a", "c", "b", "c", "b"]})


def test_sort_values_by_const_str_or_str_list():
    """
    Test sort_values(): 'by' is of type str or list of str
    """

    def impl1(df):
        return df.sort_values(by=None)

    def impl2(df):
        return df.sort_values(by=1)

    with pytest.raises(
        BodoError,
        match="'by' parameter only supports a constant column label or column labels",
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError,
        match="'by' parameter only supports a constant column label or column labels",
    ):
        bodo.jit(impl2)(df)


def test_sort_values_by_labels():
    """
    Test sort_values(): 'by' is a valid label or label lists
    """

    def impl1(df):
        return df.sort_values(by=["C"])

    def impl2(df):
        return df.sort_values(by=["B", "C"])

    with pytest.raises(BodoError, match="invalid key .* for by"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="invalid key .* for by"):
        bodo.jit(impl2)(df)


def test_sort_values_axis_default():
    """
    Test sort_values(): 'axis' cannot be values other than integer value 0
    """

    def impl1(df):
        return df.sort_values(by=["A"], axis=1)

    def impl2(df):
        return df.sort_values(by=["A"], axis="1")

    def impl3(df):
        return df.sort_values(by=["A"], axis=None)

    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl3)(df)


def test_sort_values_ascending_bool_list():
    """
    Test sort_values(): 'ascending' bool list is not supported
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], ascending=[True, False])

    def impl2(df, ascending):
        return df.sort_values(by=["A", "B"], ascending=ascending)

    ascending = [True, False]
    with pytest.raises(BodoError, match="multiple sort orders are not supported"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="multiple sort orders are not supported"):
        bodo.jit(impl2)(df, ascending)


def test_sort_values_ascending_bool():
    """
    Test sort_values(): 'ascending' must be of type bool
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], ascending=None)

    def impl2(df):
        return df.sort_values(by=["A"], ascending=2)

    def impl3(df, ascending):
        return df.sort_values(by=["A", "B"], ascending=ascending)

    with pytest.raises(BodoError, match="'ascending' parameter must be of type bool"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="'ascending' parameter must be of type bool"):
        bodo.jit(impl2)(df)


def test_sort_values_inplace_bool():
    """
    Test sort_values(): 'inplace' must be of type bool
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], inplace=None)

    def impl2(df):
        return df.sort_values(by="A", inplace=9)

    with pytest.raises(BodoError, match="'inplace' parameter must be of type bool"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="'inplace' parameter must be of type bool"):
        bodo.jit(impl2)(df)


def test_sort_values_kind_no_spec():
    """
    Test sort_values(): 'kind' should not be specified by users
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], kind=None)

    def impl2(df):
        return df.sort_values(by=["A"], kind="quicksort")

    def impl3(df):
        return df.sort_values(by=["A"], kind=2)

    with pytest.warns(BodoWarning, match="specifying sorting algorithm"):
        bodo.jit(impl1)(df)
    with pytest.warns(
        BodoWarning, match="specifying sorting algorithm is not supported"
    ):
        bodo.jit(impl2)(df)
    with pytest.warns(
        BodoWarning, match="specifying sorting algorithm is not supported"
    ):
        bodo.jit(impl3)(df)


def test_sort_values_na_position_no_spec():
    """
    Test sort_values(): 'na_position' should not be specified by users
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], na_position=None)

    def impl2(df):
        return df.sort_values(by=["A"], na_position="last")

    def impl3(df):
        return df.sort_values(by=["A"], na_position=0)

    with pytest.raises(BodoError, match="na_position is not currently supported"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="na_position is not currently supported"):
        bodo.jit(impl2)(df)
    with pytest.raises(BodoError, match="na_position is not currently supported"):
        bodo.jit(impl3)(df)
