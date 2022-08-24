# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests correctness of the 'Greatest' keyword in BodoSQL

Note, a large number of tests on this file generate wildly different column names then the
spark result. See BS-119.
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(params=["GREATEST", "LEAST"])
def greatest_or_least(request):
    return request.param


def test_greatest_integer_literals(
    basic_df, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on integer literals
    """
    query1 = f"""
    SELECT
        A, {greatest_or_least}(1, 1, -1, 2, 3, 5)
    FROM
        table1
    """
    query2 = f"""
    SELECT
        A, {greatest_or_least}(-1, -1, -1, -2, -3, -5)
    FROM
        table1
    """
    check_query(query1, basic_df, spark_info, check_dtype=False, check_names=False)
    check_query(query2, basic_df, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skip("[BS-154], Issue with float comparison")
def test_greatest_float_literals(
    basic_df, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on float literals
    """
    query1 = f"""
    SELECT
        A, {greatest_or_least}(.1, .111, -1.1, 1.75, .5, .01)
    FROM
        table1
    """
    query2 = f"""
    SELECT
        A, {greatest_or_least}(-.1, -.111, -1.1, -1.75, -.5, -.01)
    FROM
        table1
    """
    check_query(query1, basic_df, spark_info, check_dtype=False, check_names=False)
    check_query(query2, basic_df, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skip("[BS-116]")
def test_greatest_string_literals(
    basic_df, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on string literals
    """
    query1 = f"""
    SELECT
        A, {greatest_or_least}('a', 'aa', 'aaa', 'baa', 'za')
    FROM
        table1
    """
    query2 = f"""
    SELECT
        A, {greatest_or_least}('a', 'A', 'AA', 'aAA', 'aa')
    FROM
        table1
    """
    check_query(query1, basic_df, spark_info, check_dtype=False, check_names=False)
    check_query(query2, basic_df, spark_info, check_dtype=False, check_names=False)


def test_greatest_bool_literals(
    basic_df, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on boolean literals
    """
    query = f"""
    SELECT
        A, {greatest_or_least}(true, false, true, false)
    FROM
        table1
    """
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_greatest_numeric_columns(
    bodosql_numeric_types, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on numeric columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C)
    FROM
        table1
    """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_greatest_string_columns(
    bodosql_string_types, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on string columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C)
    FROM
        table1
    """
    check_query(
        query, bodosql_string_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.skip("[BE-959] Support Binary Array as output of apply")
def test_greatest_binary_columns(
    bodosql_binary_types, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on binary columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C)
    FROM
        table1
    """
    check_query(
        query, bodosql_binary_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.skip("[BS-118]")
def test_greatest_bool_columns(
    bodosql_boolean_types, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on boolean columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C)
    FROM
        table1
    """
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )
