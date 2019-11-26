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
    check_func(test_impl, (get_quasi_random(100),), sort_output=True)
