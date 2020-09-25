# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on strings as
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
                    "hello",
                    "world",
                ]
                * 40,
                "B": [
                    "weird",
                    "world",
                ]
                * 40,
                "C": [
                    "krusty",
                    "klown",
                ]
                * 40,
                "D": [1.2, 5.3231] * 40,
            }
        ),
    ]
)
def dataframe_val(request):
    return request.param


def test_concat_strings(dataframe_val):
    def test_impl(df):
        return df[["A", "B", "C"]].apply(lambda x: ",".join(x), axis=1)

    check_func(test_impl, (dataframe_val,))
