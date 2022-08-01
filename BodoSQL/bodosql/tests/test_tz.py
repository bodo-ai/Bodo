# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of Timestamp operations that
include timezone information.

Note: Because Spark does not contain timezones we cannot
compart with SparkSQL for correctness
"""
import pandas as pd

from bodosql.tests.utils import check_query


def test_select(memory_leak_check):
    """
    Test a simple select statement with a table including
    timezone information.
    """
    query = "Select B from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range("2/2/2022", periods=12, freq="1D2H", tz="Poland"),
            "B": pd.date_range("2/25/2021", periods=12, freq="1D2H", tz="US/Pacific"),
            "C": pd.date_range("5/22/2022", periods=12, freq="1D2H", tz="UTC"),
        }
    )
    expected_output = df[["B"]]
    ctx = {"table1": df}
    check_query(query, ctx, None, expected_output=expected_output)
