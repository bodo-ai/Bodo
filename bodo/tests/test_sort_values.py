# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test sort_values opration as called as df.sort_values()
   The C++ implementation uses the timsort which is a stable sort algorithm.
   Therefore, in the test we use mergesort, which guarantees that the equality
   tests can be made sensibly.
"""
import pandas as pd
import numpy as np
import bodo
import pytest
from bodo.tests.utils import check_func


def test_sort_values_1col():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = []
        for i in range(n):
            eVal = i*i % 34
            eListA.append(eVal)
        return pd.DataFrame({"A": eListA})
    n=100
    check_func(test_impl, (get_quasi_random(n),), sort_output=False)


def test_sort_values_1col_np_array():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_int8(n):
        eListA = np.array([0]*n, dtype=np.int8)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    def get_quasi_random_uint8(n):
        eListA = np.array([0]*n, dtype=np.uint8)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    
    def get_quasi_random_int16(n):
        eListA = np.array([0]*n, dtype=np.int16)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    def get_quasi_random_uint16(n):
        eListA = np.array([0]*n, dtype=np.uint16)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    
    def get_quasi_random_int32(n):
        eListA = np.array([0]*n, dtype=np.int32)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    def get_quasi_random_uint32(n):
        eListA = np.array([0]*n, dtype=np.uint32)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    
    def get_quasi_random_int64(n):
        eListA = np.array([0]*n, dtype=np.int64)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    def get_quasi_random_uint64(n):
        eListA = np.array([0]*n, dtype=np.uint64)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    
    def get_quasi_random_float32(n):
        eListA = np.array([0]*n, dtype=np.float32)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    def get_quasi_random_float64(n):
        eListA = np.array([0]*n, dtype=np.float64)
        for i in range(n):
            eVal = i*i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})
    n=100
    check_func(test_impl, (get_quasi_random_int8(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint8(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int16(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint16(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int32(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint32(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int64(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint64(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_float32(n),), sort_output=False)
    check_func(test_impl, (get_quasi_random_float64(n),), sort_output=False)


def test_sort_values_1col_descending():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", ascending=False, kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = []
        for i in range(n):
            eVal = i*i % 34
            eListA.append(eVal)
        return pd.DataFrame({"A": eListA})
    n=100
    check_func(test_impl, (get_quasi_random(n),), sort_output=False)

def test_sort_values_2col():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by=["A","B"], kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = np.array([0]*n, dtype=np.uint64)
        eListB = np.array([0]*n, dtype=np.uint64)
        for i in range(n):
            eValA = i*i % 34
            eValB = i*i*i % 34
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})
    n=100
    check_func(test_impl1, (get_quasi_random(n),), sort_output=False)
    check_func(test_impl2, (get_quasi_random(n),), sort_output=False)
