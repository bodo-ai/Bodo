# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries that are depenent on time since year 0, or the unix epoch
"""
import pandas as pd
from bodosql.tests.utils import check_query

# the difference in days between the unix epoch and the start of year 0
dayDeltaUnixY0 = 719528
# the difference in days seconds between the unix epoch and the start of year 0
secondDeltaUnixY0 = 62167219200


def test_fromdays_cols(spark_info, basic_df, memory_leak_check):
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


def test_fromdays_scalar(spark_info, basic_df, memory_leak_check):
    """tests from_days function on scalar values"""

    query = f"SELECT CASE WHEN FROM_DAYS(B + {dayDeltaUnixY0}) = TIMESTAMP '1970-1-1' then TIMESTAMP '1970-1-2' ELSE FROM_DAYS(B + {dayDeltaUnixY0}) END from table1"
    spark_query = "SELECT CASE WHEN DATE_FROM_UNIX_DATE(B) = TIMESTAMP '1970-1-1' then TIMESTAMP '1970-1-2' ELSE DATE_FROM_UNIX_DATE(B) END from table1"

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
    query = "SELECT TO_SECONDS(A), TO_SECONDS(B), TO_SECONDS(C) from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "a": (
                bodosql_datetime_types["table1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
            "b": (
                bodosql_datetime_types["table1"]["B"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
            "c": (
                bodosql_datetime_types["table1"]["C"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
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


def test_to_seconds_scalars(spark_info, bodosql_datetime_types, memory_leak_check):
    """tests to_seconds function on scalar values"""
    query = (
        "SELECT CASE WHEN TO_SECONDS(A) = 1 THEN -1 ELSE TO_SECONDS(A) END from table1"
    )

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "a": (
                bodosql_datetime_types["table1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.total_seconds()
            + secondDeltaUnixY0,
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


def test_to_days_cols(spark_info, bodosql_datetime_types, memory_leak_check):
    """test_to_days on column values"""
    query = "SELECT TO_DAYS(A), TO_DAYS(B), TO_DAYS(C) from table1"

    # Since spark has no equivalent function, we need to manually set the expected output
    expected_output = pd.DataFrame(
        {
            "a": (
                bodosql_datetime_types["table1"]["A"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
            "b": (
                bodosql_datetime_types["table1"]["B"] - pd.Timestamp("1970-1-1")
            ).dt.days
            + dayDeltaUnixY0,
            "c": (
                bodosql_datetime_types["table1"]["C"] - pd.Timestamp("1970-1-1")
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
                bodosql_datetime_types["table1"]["A"] - pd.Timestamp("1970-1-1")
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


def test_unix_timestamp(spark_info, basic_df, memory_leak_check):
    """tests the unix_timestamp function."""

    # essentially, omit the last 3 digits durring the check.
    # This will sometimes randomly fail, if the test takes place
    # right as the ten thousandths place changes values
    # query = "SELECT A, Ceiling(UNIX_TIMESTAMP() / 10000) from table1"
    # spark_query = "SELECT A, Ceiling(to_unix_timestamp(current_timestamp())/10000) from table1"

    query = "SELECT A, UNIX_TIMESTAMP() from table1"
    spark_query = "SELECT A, to_unix_timestamp(current_timestamp()) from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_from_unixtime_cols(spark_info, basic_df, memory_leak_check):
    """tests from_unixtime function on column values"""

    seconds_in_day = 86400
    query = f"SELECT from_unixtime(A * {seconds_in_day}), from_unixtime(B * {seconds_in_day}), from_unixtime(C * {seconds_in_day}) from table1"
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
