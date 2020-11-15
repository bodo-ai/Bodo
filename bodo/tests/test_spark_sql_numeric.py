# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on numeric
column elements.

Test names refer to the names of the spark function they map to.
"""

import math

import numpy as np
import pandas as pd
import pytest
import scipy

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 3, 10, 15, 20, 11, 19] * 10,
                    "B": [201, 30, -10, 15, -30, -1000, 100] * 10,
                }
            )
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.slow
def test_factorial(dataframe_val):
    """Factorial limit for a 64 bit integer is 20"""

    def test_impl1(df):
        return df.A.map(lambda x: math.factorial(x))

    def test_impl2(df):
        return df.A.map(lambda x: np.math.factorial(x))

    def test_impl3(df):
        return df.A.map(lambda x: scipy.math.factorial(x))

    check_func(test_impl1, (dataframe_val,))
    check_func(test_impl2, (dataframe_val,))
    check_func(test_impl3, (dataframe_val,))
