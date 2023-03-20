# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test SQL `date` support
"""
import datetime

import bodosql
import pandas as pd
import pytest
from bodosql.context import BodoSQLContext
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


@pytest.mark.parametrize(
    "unit, test_fn_type, answer",
    [
        pytest.param(
            "year",
            "DATE_PART",
            pd.Series([1999, None, 2010, 2021, 2023], dtype=pd.Int32Dtype()),
            id="valid-year-date_part",
        ),
        pytest.param(
            "quarter",
            "EXTRACT",
            pd.Series([4, None, 1, 3, 2], dtype=pd.Int32Dtype()),
            id="valid-quarter-extract",
        ),
        pytest.param(
            "month",
            "MONTH",
            pd.Series([12, None, 1, 7, 4], dtype=pd.Int32Dtype()),
            id="valid-month-date_part",
        ),
        pytest.param(
            "week",
            "EXTRACT",
            pd.Series([52, None, 53, 26, 15], dtype=pd.Int32Dtype()),
            id="valid-week-extract",
        ),
        pytest.param(
            "dayofmonth",
            "DATE_PART",
            pd.Series([31, None, 1, 4, 15], dtype=pd.Int32Dtype()),
            id="valid-dayofmonth-date_part",
        ),
        pytest.param(
            "dayofyear",
            "DAYOFYEAR",
            pd.Series([365, None, 1, 185, 105], dtype=pd.Int32Dtype()),
            id="valid-dayofyear-regular",
        ),
        pytest.param(
            "dow",
            "EXTRACT",
            pd.Series([5, None, 5, 0, 6], dtype=pd.Int32Dtype()),
            id="valid-dayofweek-extract",
        ),
        pytest.param(
            "dayofweekiso",
            "DAYOFWEEKISO",
            pd.Series([5, None, 5, 7, 6], dtype=pd.Int32Dtype()),
            id="valid-dayofweekiso-regular",
        ),
        pytest.param(
            "hour",
            "DATE_PART",
            None,
            id="invalid-hour-date_part",
        ),
        pytest.param(
            "minute",
            "MINUTE",
            None,
            id="invalid-minute-regular",
        ),
        pytest.param(
            "second",
            "EXTRACT",
            None,
            id="invalid-second-extract",
        ),
        pytest.param(
            "nanosecond",
            "DATE_PART",
            None,
            id="invalid-nanosecond-date_part",
        ),
        pytest.param(
            "microsecond",
            "MICROSECOND",
            None,
            id="invalid-microsecond-regular",
        ),
        pytest.param(
            "millisecond",
            "EXTRACT",
            None,
            id="invalid-millisecond-extract",
        ),
    ],
)
def test_date_extract(unit, answer, test_fn_type, memory_leak_check):
    """Tests EXTRACT and EXTRACT-like functions on date data, checking that
    values smaller than DAY raise an exception"""
    if test_fn_type == "EXTRACT":
        query = f"SELECT EXTRACT({unit} FROM D) AS U FROM table1"
    elif test_fn_type == "DATE_PART":
        query = f"SELECT DATE_PART('{unit}', D) AS U FROM table1"
    else:
        query = f"SELECT {test_fn_type}(D) AS U FROM table1"
    ctx = {
        "table1": pd.DataFrame(
            {
                "D": pd.Series(
                    [
                        datetime.date(1999, 12, 31),
                        None,
                        datetime.date(2010, 1, 1),
                        datetime.date(2021, 7, 4),
                        datetime.date(2023, 4, 15),
                    ]
                )
            }
        )
    }
    with bodosql_use_date_type():
        if answer is None:
            bc = BodoSQLContext(ctx)
            with pytest.raises(
                Exception, match=r"Cannot extract unit \w+ from DATE values"
            ):
                bc.sql(query)
        else:
            expected_output = pd.DataFrame({"U": answer})
            check_query(
                query,
                ctx,
                None,
                expected_output=expected_output,
                check_dtype=False,
                sort_output=False,
            )
