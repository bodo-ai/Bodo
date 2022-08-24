# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on maps as
column elements.

Test names refer to the names of the spark function they map to.

These will need to consist of tests for both Struct Array types
and map array types, as both are represented with dictionaries 
in Python.
"""

import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        {"a": 1, "b": 2},
                        {"a": 1001, "b": 45},
                    ]
                    * 20,
                    "B": [
                        {1: 1.4, 2: 3.1},
                        {},
                    ]
                    * 20,
                }
            )
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        {1: 1.4, 2: 3.1},
                        {7: -1.2},
                        {11: 3.4, 21: 3.1, 9: 8.1},
                        {4: 9.4, 6: 4.1},
                        {7: -1.2},
                        {},
                    ]
                    * 10,
                    "B": [
                        {"a": 1, "b": 2},
                        {"a": -111, "b": -45},
                        {"a": 411, "b": 322},
                        {"a": 0, "b": 34245},
                        {"a": 343, "b": 5235245},
                        {"a": 168, "b": 951},
                    ]
                    * 10,
                }
            )
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.slow
def test_size(dataframe_val):
    def test_impl(df):
        return df.A.map(lambda x: len(x))

    df = dataframe_val
    check_func(test_impl, (df,))
