"""
Test correctness of SQL queries that are dependent on time since year 0, or the unix epoch
"""

import numpy as np
import pandas as pd

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


# the difference in days between the unix epoch and the start of year 0
dayDeltaUnixY0 = 719528
# the difference in days seconds between the unix epoch and the start of year 0
secondDeltaUnixY0 = 62167219200


def test_from_days_cols(spark_info, basic_df, memory_leak_check):
    """tests from_days function on column values"""

    query = f"SELECT FROM_DAYS(A + {dayDeltaUnixY0}), FROM_DAYS(B + {dayDeltaUnixY0}), FROM_DAYS(C + {dayDeltaUnixY0}) from table1"
    spark_query = "SELECT DATE_FROM_UNIX_DATE(A), DATE_FROM_UNIX_DATE(B), DATE_FROM_UNIX_DATE(C) from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_from_days_scalar(spark_info, basic_df, memory_leak_check):
    """tests from_days function on scalar values"""

    query = f"SELECT CASE WHEN FROM_DAYS(B + {dayDeltaUnixY0}) = DATE '1970-1-1' then DATE '1970-1-2' ELSE FROM_DAYS(B + {dayDeltaUnixY0}) END from table1"
    spark_query = "SELECT CASE WHEN DATE_FROM_UNIX_DATE(B) = DATE '1970-1-1' then DATE '1970-1-2' ELSE DATE_FROM_UNIX_DATE(B) END from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_to_seconds_cols(spark_info, bodosql_datetime_types, memory_leak_check):
    """tests to_seconds function on column values"""
    query = "SELECT TO_SECONDS(A) AS A_OUT, TO_SECONDS(B) AS B_OUT, TO_SECONDS(C) AS C_OUT from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "A_OUT": (
                bodosql_datetime_types["TABLE1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
            "B_OUT": (
                bodosql_datetime_types["TABLE1"]["B"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
            "C_OUT": (
                bodosql_datetime_types["TABLE1"]["C"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
        }
    )

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_to_seconds_scalars(spark_info, bodosql_datetime_types, memory_leak_check):
    """tests to_seconds function on scalar values"""
    query = "SELECT CASE WHEN TO_SECONDS(A) = 1 THEN -1 ELSE TO_SECONDS(A) END AS A_OUT from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "A_OUT": (
                bodosql_datetime_types["TABLE1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
        }
    )

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_to_days_cols(spark_info, bodosql_datetime_types, memory_leak_check):
    """test_to_days on column values"""
    query = "SELECT TO_DAYS(A), TO_DAYS(B), TO_DAYS(C) from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "a": (
                bodosql_datetime_types["TABLE1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
            "b": (
                bodosql_datetime_types["TABLE1"]["B"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
            "c": (
                bodosql_datetime_types["TABLE1"]["C"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
        }
    )

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_to_days_scalars(spark_info, bodosql_datetime_types, memory_leak_check):
    """test to_days on scalar values"""
    query = "SELECT CASE WHEN TO_DAYS(A) = 0 then -1 ELSE TO_DAYS(A) END from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "a": (
                bodosql_datetime_types["TABLE1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
        }
    )

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_unix_timestamp(basic_df, memory_leak_check):
    """tests the unix_timestamp function."""

    # essentially, omit the last 3 digits during the check.
    # This will sometimes randomly fail, if the test takes place
    # right as the ten thousandths place changes values
    query = "SELECT A, Floor(UNIX_TIMESTAMP() / 10000) as output from table1"
    expected_output = pd.DataFrame(
        {
            "A": basic_df["TABLE1"]["A"],
            "OUTPUT": np.float64(pd.Timestamp.now().value // (10000 * 1000000000)),
        }
    )

    check_query(
        query,
        basic_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_from_unixtime_cols(spark_info, basic_df, memory_leak_check):
    """tests from_unixtime function on column values"""

    seconds_in_day = 86400
    # We need to wrap from_unixtime with TO_DATE since spark doesn't support the timestamp version
    query = f"SELECT TO_DATE(from_unixtime(A * {seconds_in_day})), TO_DATE(from_unixtime(B * {seconds_in_day})), TO_DATE(from_unixtime(C * {seconds_in_day})) from table1"
    spark_query = "SELECT DATE_FROM_UNIX_DATE(A), DATE_FROM_UNIX_DATE(B), DATE_FROM_UNIX_DATE(C) from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_from_unixtime_scalars(spark_info, basic_df, memory_leak_check):
    """tests the from_unixtime function on scalar values"""

    seconds_in_day = 86400
    query = f"SELECT CASE WHEN from_unixtime(B * {seconds_in_day}) = TIMESTAMP '1970-1-1' then TIMESTAMP '1970-1-2' ELSE from_unixtime(B * {seconds_in_day}) END from table1"
    spark_query = "SELECT CASE WHEN DATE_FROM_UNIX_DATE(B) = TIMESTAMP '1970-1-1' then TIMESTAMP '1970-1-2' ELSE DATE_FROM_UNIX_DATE(B) END from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )
