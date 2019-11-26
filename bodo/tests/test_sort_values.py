# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test drop_duplicate opration as called as df.drop_duplicates()
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
        df2 = df1.sort_values("A")
        return df2

    def get_quasi_random(n):
        eListA = []
        for i in range(n):
            eVal = i*i % 34
            eListA.append(eVal)
        return pd.DataFrame({"A": eListA})
    check_func(test_impl, (get_quasi_random(100),), sort_output=False)


def test_sort_values_1col_np_array():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values("A")
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

    check_func(test_impl, (get_quasi_random_int8(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint8(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int16(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint16(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int32(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint32(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_int64(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_uint64(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_float32(100),), sort_output=False)
    check_func(test_impl, (get_quasi_random_float64(100),), sort_output=False)


