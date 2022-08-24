# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL filter queries on BodoSQL
"""

import pytest
from bodosql.tests.utils import check_query


def test_filter_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test filter queries"""

    check_query(
        "select A,C from table1 where C = 2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A = C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,B from table1 where B <> 1.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A <> C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,B from table1 where B < 2.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A < C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,B from table1 where B <= 2.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A <= C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,B from table1 where B > 2.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A > C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,B from table1 where B >= 2.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A >= C",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A >= C and B >= 2.2 and A >=1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A,C from table1 where A >= C or B >= 2.2",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )


def test_filter_null_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test is null on numeric columns"""
    query1 = "select A,C from table1 where A is NULL"
    query2 = "select C from table1 where A is not null"
    check_query(query1, bodosql_numeric_types, spark_info, check_dtype=False)
    check_query(query2, bodosql_numeric_types, spark_info, check_dtype=False)


def test_filter_null_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """test is null on nullable numeric columns"""
    query1 = "select A,C from table1 where A is NULL"
    query2 = "select C from table1 where A is not null"
    check_query(query1, bodosql_nullable_numeric_types, spark_info, check_dtype=False)
    check_query(query2, bodosql_nullable_numeric_types, spark_info, check_dtype=False)


def test_filter_null_string(bodosql_string_types, spark_info, memory_leak_check):
    """test is null on Timestamp columns"""
    query1 = "select A,C from table1 where B is NULL"
    query2 = "select C from table1 where B is not null"
    check_query(query1, bodosql_string_types, spark_info, check_dtype=False)
    check_query(query2, bodosql_string_types, spark_info, check_dtype=False)


def test_filter_null_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """test is null on Timestamp columns"""
    query1 = "select A,C from table1 where B is NULL"
    query2 = "select C from table1 where B is not null"
    check_query(
        query1,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        convert_columns_bytearray=["A", "C"],
    )
    check_query(
        query2,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        convert_columns_bytearray=["C"],
    )


def test_filter_null_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test is null on Timestamp columns"""
    query1 = "select A,C from table1 where B is NULL"
    query2 = "select C from table1 where B is not null"
    check_query(query1, bodosql_datetime_types, spark_info, check_dtype=False)
    check_query(query2, bodosql_datetime_types, spark_info, check_dtype=False)


def test_filter_null_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """test is null on Interval columns"""
    query1 = "select A,C from table1 where B is NULL"
    query2 = "select C from table1 where B is not null"
    check_query(
        query1,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["A", "C"],
    )
    check_query(
        query2,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["C"],
    )


def test_filter_boolean_1(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,B from table1 where B = FALSE",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_filter_boolean_2(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,C from table1 where NOT B = TRUE",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_filter_boolean_3(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,C from table1 where A",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


def test_filter_boolean_4(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,C from table1 where A = C",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_filter_boolean_5(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select B,A from table1 where A <> C",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_filter_boolean_6(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,C from table1 where NOT A and B = TRUE or C <> FALSE",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.skip(
    "BS-162, this test currently fails due to the difference between Spark and BodoSQL's handling of NULL or True"
)
def test_filter_boolean_7(bodosql_boolean_types, spark_info, memory_leak_check):
    check_query(
        "select A,C from table1 where A or B = FALSE and Not FALSE",
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
    )
