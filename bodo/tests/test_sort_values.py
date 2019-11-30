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


def test_sort_values_1col_np_array():
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
    check_func(test_impl, (get_quasi_random_dtype(n, np.int8),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.uint8),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.int16),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.uint16),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.int32),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.uint32),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.int64),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.uint64),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.float32),), sort_output=False)
    check_func(test_impl, (get_quasi_random_dtype(n, np.float64),), sort_output=False)


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


def test_sort_values_2col_np_array():
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
        test_impl, (get_quasi_random_dtype(n, np.int8, np.int16),), sort_output=False
    )
    check_func(
        test_impl, (get_quasi_random_dtype(n, np.uint8, np.int32),), sort_output=False
    )
    check_func(
        test_impl, (get_quasi_random_dtype(n, np.int16, np.float64),), sort_output=False
    )
    check_func(
        test_impl,
        (get_quasi_random_dtype(n, np.uint16, np.float32),),
        sort_output=False,
    )


def test_sort_values_strings_constant_length():
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

    check_func(test_impl, (get_random_strings_array(1000, 2),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(100, 1),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(300, 2),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(100, 1),), sort_output=False)


def test_sort_values_strings_variable_length():
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
        test_impl, (get_random_var_length_strings_array(100, 30),), sort_output=False
    )
    check_func(
        test_impl, (get_random_var_length_strings_array(1000, 10),), sort_output=False
    )
    check_func(
        test_impl, (get_random_var_length_strings_array(10, 30),), sort_output=False
    )
    check_func(
        test_impl, (get_random_var_length_strings_array(100, 30),), sort_output=False
    )


def test_sort_values_strings():
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

    check_func(test_impl, (get_random_strings_array(100, 30),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(1000, 10),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(10, 30),), sort_output=False)
    check_func(test_impl, (get_random_strings_array(100, 30),), sort_output=False)
