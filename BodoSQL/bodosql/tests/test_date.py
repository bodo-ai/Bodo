# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test SQL `date` support
"""
import datetime

import bodosql
import pandas as pd
import pytest
from bodosql.context import BodoSQLContext
from bodosql.tests.utils import check_query

from bodo.tests.conftest import (  # noqa
    date_df,
    day_part_strings,
    time_part_strings,
)
from bodo.tests.test_bodosql_array_kernels.test_bodosql_datetime_array_kernels import (
    diff_fn,
)


@pytest.mark.parametrize(
    "fn_name",
    [
        "DATE",
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
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "fn_name",
    [
        "DATE",
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


@pytest.mark.parametrize(
    "query, expected_output",
    [
        pytest.param(
            "SELECT DATEDIFF('YEAR', TO_DATE('2000-01-01'), TO_DATE('2022-12-31'))",
            pd.DataFrame({"A": pd.Series([22])}),
            id="year",
        ),
        pytest.param(
            "SELECT TIMEDIFF('QUARTER', TO_DATE('2022-06-30'), TO_DATE('2000-01-01'))",
            pd.DataFrame({"A": pd.Series([-89])}),
            id="quarter",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('MONTH', TO_DATE('2022-06-30'), TO_DATE('2020-01-01'))",
            pd.DataFrame({"A": pd.Series([-29])}),
            id="month",
        ),
        pytest.param(
            "SELECT DATEDIFF('WEEK', TO_DATE('2010-01-01'), TO_DATE('2022-12-31'))",
            pd.DataFrame({"A": pd.Series([678])}),
            id="week",
        ),
        pytest.param(
            "SELECT TIMEDIFF('DAY', TO_DATE('2022-06-30'), TO_DATE('2000-01-01'))",
            pd.DataFrame({"A": pd.Series([-8216])}),
            id="day",
        ),
        pytest.param(
            "SELECT DATEDIFF('HOUR', TO_DATE('2020-01-01'), TO_DATE('2022-06-30'))",
            pd.DataFrame({"A": pd.Series([21864])}),
            id="hour",
        ),
        pytest.param(
            "SELECT TIMEDIFF('MINUTE', TO_DATE('2010-01-01'), TO_DATE('2022-12-31'))",
            pd.DataFrame({"A": pd.Series([6835680])}),
            id="minute",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('SECOND', TO_DATE('2022-06-30'), TO_DATE('2020-01-01'))",
            pd.DataFrame({"A": pd.Series([-78710400])}),
            id="second",
        ),
        pytest.param(
            "SELECT DATEDIFF('MILLISECOND', TO_DATE('2000-01-01'), TO_DATE('2022-12-31'))",
            pd.DataFrame({"A": pd.Series([725760000000])}),
            id="millisecond",
        ),
        pytest.param(
            "SELECT TIMEDIFF('MICROSECOND', TO_DATE('2022-12-31'), TO_DATE('2010-01-01'))",
            pd.DataFrame({"A": pd.Series([-410140800000000])}),
            id="microsecond",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('NANOSECOND', TO_DATE('2020-01-01'), TO_DATE('2022-06-30'))",
            pd.DataFrame({"A": pd.Series([78710400000000000])}),
            id="nanosecond",
        ),
    ],
)
@pytest.mark.slow
def test_datediff_date_literals(query, expected_output, basic_df, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF on datetime.date literals behaves as expected.
    Tests all possible datetime parts except for time units.
    """

    check_query(
        query,
        basic_df,
        spark=None,
        expected_output=expected_output,
        check_names=False,
        check_dtype=False,
    )


def test_datediff_date_columns_time_units(
    date_df, time_part_strings, memory_leak_check
):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF with date columns and time units behaves as expected
    """
    fn_name = {
        "HOUR": "DATEDIFF",
        "hr": "TIMEDIFF",
        "MINUTE": "TIMESTAMPDIFF",
        "min": "DATEDIFF",
        "SECOND": "TIMEDIFF",
        "ms": "TIMESTAMPDIFF",
        "microsecond": "DATEDIFF",
        "usec": "TIMEDIFF",
        "nanosecs": "TIMESTAMPDIFF",
    }[time_part_strings]
    query = f"SELECT {fn_name}('{time_part_strings}', A, B) as output from table1"
    output = pd.DataFrame(
        {
            "output": [
                diff_fn(
                    time_part_strings,
                    date_df["table1"]["A"][i],
                    date_df["table1"]["B"][i],
                )
                for i in range(len(date_df["table1"]["A"]))
            ]
        }
    )
    check_query(
        query,
        date_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=output,
    )


@pytest.mark.slow
def test_datediff_date_columns_day_units(date_df, day_part_strings, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF with date columns and date units behaves as expected
    """
    fn_name = {
        "quarter": "DATEDIFF",
        "yyy": "TIMEDIFF",
        "MONTH": "TIMESTAMPDIFF",
        "mon": "DATEDIFF",
        "WEEK": "TIMEDIFF",
        "wk": "TIMESTAMPDIFF",
        "DAY": "DATEDIFF",
        "dd": "TIMEDIFF",
    }[day_part_strings]
    query = f"SELECT {fn_name}('{day_part_strings}', A, B) as output from table1"
    output = pd.DataFrame(
        {
            "output": [
                diff_fn(
                    day_part_strings,
                    date_df["table1"]["A"][i],
                    date_df["table1"]["B"][i],
                )
                for i in range(len(date_df["table1"]["A"]))
            ]
        }
    )
    check_query(
        query,
        date_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "func_name,expected",
    [
        pytest.param("NEXT_DAY", datetime.date(1999, 1, 3), id="next_day"),
        pytest.param("PREVIOUS_DAY", datetime.date(1998, 12, 27), id="prev_day"),
    ],
)
def test_date_next_day(func_name, expected, memory_leak_check):
    query = f"select {func_name}(TO_DATE('1999-01-01'), 'Sunday') as A"
    ctx = {}
    expected_output = pd.DataFrame({"A": [expected]})
    check_query(query, ctx, None, expected_output=expected_output)


def test_max_date(date_df, memory_leak_check):
    """
    Test that max is working for date type columns
    """
    query = "SELECT MAX(A) FROM table1"
    check_query(
        query,
        date_df,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"A": [datetime.date(2024, 1, 1)]}),
        is_out_distributed=False,
    )


def test_min_date(date_df, memory_leak_check):
    """
    Test that min is working for date type columns
    """
    query = "SELECT MIN(B) FROM table1"
    check_query(
        query,
        date_df,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"B": [datetime.date(1700, 2, 4)]}),
        is_out_distributed=False,
    )


def test_max_date_group_by(date_df, spark_info, memory_leak_check):
    """
    Test that max with group by is working for date type columns
    """
    query = "SELECT MAX(A) FROM table1 GROUP BY C"
    check_query(
        query,
        date_df,
        spark_info,
        check_names=False,
    )


def test_min_date_group_by(date_df, spark_info, memory_leak_check):
    """
    Test that min with group by is working for date type columns
    """
    query = "SELECT MIN(B) FROM table1 GROUP BY C"
    check_query(
        query,
        date_df,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_str_to_date_literals(basic_df, memory_leak_check):
    """
    Checks that calling STR_TO_DATE on literals behaves as expected
    """
    query = "SELECT STR_TO_DATE('17-09-2010', '%d-%m-%Y')"
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"output": [datetime.date(2010, 9, 17)]}),
    )


def test_str_to_date_columns(memory_leak_check):
    """
    Checks that calling STR_TO_DATE on columns behaves as expected
    """
    ctx = {
        "table1": pd.DataFrame({"A": ["2003-02-01", "2013-02-11", "2011-11-01"] * 4})
    }
    query = "SELECT STR_TO_DATE(A, '%Y-%m-%d') from table1"
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame(
            {
                "output": [
                    datetime.date(2003, 2, 1),
                    datetime.date(2013, 2, 11),
                    datetime.date(2011, 11, 1),
                ]
                * 4
            }
        ),
    )


@pytest.mark.slow
def test_str_to_date_columns_format(memory_leak_check):
    """
    Checks that calling STR_TO_DATE on columns behaves as expected when
    the format string needs to be replaced. Note this does not test all
    possible conversions.
    """
    ctx = {
        "table1": pd.DataFrame(
            {"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4}
        )
    }
    query = "SELECT STR_TO_DATE(A, '%Y-%m-%d:%h') from table1"
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame(
            {
                "output": [
                    datetime.date(2003, 2, 1),
                    datetime.date(2013, 2, 11),
                    datetime.date(2011, 11, 1),
                ]
                * 4
            }
        ),
    )
