# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test SQL `date` support
"""
import datetime

import bodosql
import pandas as pd
import pytest
from bodosql.tests.utils import bodosql_use_date_type, check_query


@pytest.mark.parametrize(
    "fn_name",
    [
        pytest.param(
            "DATE", marks=pytest.mark.skip(reason="Waiting on Calcite support")
        ),
        "TO_DATE",
    ],
)
@pytest.mark.parametrize(
    "scalar,expected",
    [
        pytest.param("'1999-01-01'", datetime.date(1999, 1, 1), id="date_string"),
        pytest.param(
            "timestamp '1999-01-01 00:00:00'", datetime.date(1999, 1, 1), id="timestamp"
        ),
        pytest.param("'3234'", datetime.date(1970, 1, 1), id="int_seconds"),
        pytest.param("'31536000001'", datetime.date(1971, 1, 1), id="int_milliseconds"),
        pytest.param(
            "'31536000000001'", datetime.date(1971, 1, 1), id="int_microseconds"
        ),
        pytest.param(
            "'31536000000000001'", datetime.date(1971, 1, 1), id="int_nanoseconds"
        ),
    ],
)
def test_date_to_date_scalar(fn_name, scalar, expected, memory_leak_check):
    query = f"select {fn_name}({scalar}) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": [expected]})
    with bodosql_use_date_type():
        check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "fn_name",
    [
        pytest.param(
            "DATE", marks=pytest.mark.skip(reason="Waiting on Calcite support")
        ),
        "TO_DATE",
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        "'NOT A DATE'",
        "'1999-54-01'",
    ],
)
def test_date_to_date_invalid(fn_name, scalar):
    query = f"select {fn_name}({scalar}) as A"
    ctx = {}
    with bodosql_use_date_type():
        with pytest.raises(
            ValueError, match="Invalid input while converting to date value"
        ):
            bc = bodosql.BodoSQLContext()
            bc.sql(query, ctx)


@pytest.mark.parametrize(
    "scalar_to_cast",
    [
        pytest.param("'1999-01-01'", id="date string"),
    ],
)
def test_date_cast_to_date(scalar_to_cast, memory_leak_check):
    query = f"select CAST({scalar_to_cast} AS DATE) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": [datetime.date(1999, 1, 1)]})
    with bodosql_use_date_type():
        check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "to_type,expected",
    [
        pytest.param("VARCHAR", "1999-01-01", id="varchar"),
        pytest.param("TIMESTAMP", pd.Timestamp("1999-01-01"), id="timestamp"),
    ],
)
def test_date_cast_from_date(to_type, expected, memory_leak_check):
    query = f"select CAST(DATE '1999-01-01' AS {to_type}) as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": [expected]})
    with bodosql_use_date_type():
        check_query(query, ctx, None, expected_output=expected_output)
