# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests that bodoSQL correctly interprets literal values

For MySql reference used, see https://dev.mysql.com/doc/refman/8.0/en/literals.html
"""

import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        #  currently not supported, we may add support for timedeltas greater then one month later
        # ("1 YEAR", pd.Timedelta(365, "D")),
        # ("1 MONTH", pd.Timedelta(31, "D")),
        ("1 DAY", pd.Timedelta(1, "D")),
        ("1 HOUR", pd.Timedelta(1, "H")),
        ("1 MINUTE", pd.Timedelta(1, "m")),
        ("1 SECOND", pd.Timedelta(1, "s")),
    ]
)
def timedelta_equivalent_values(request):
    """fixture that returns a tuple of a timedelta string literal, and the corresponding pd.Timedelta time"""
    return request.param


def test_timestamp_literals(
    basic_df, timestamp_literal_strings, spark_info, memory_leak_check
):
    """
    tests that timestamp literals are correctly parsed by BodoSQL
    """
    query = f"""
    SELECT
        A, TIMESTAMP '{timestamp_literal_strings}'
    FROM
        table1
    """

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip("[BS-173] issues with extract on ")
def test_timestamp_literals_extract(
    basic_df, timestamp_literal_strings, spark_info, memory_leak_check
):
    """
    tests that timestamp literals can be used with basic functions
    """
    query = f"""
    SELECT
        A, EXTRACT(YEAR FROM TIMESTAMP '{timestamp_literal_strings}')
    FROM
        table1
    """

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        optimize_calcite_plan=False,
    )


@pytest.mark.skip(
    "Currently parses with calcite, unusported timestamp due to a pandas error"
)
def test_timestamp_literal_pd_error(basic_df, spark_info, memory_leak_check):
    """This is a specific test case that is parsed correctly by calacite, but generates a runtime pandas error.
    If we want to support this, it'll probably be a quicker fix then the other ones
    """
    query = """
    SELECT
        A, TIMESTAMP '2020-12-01 13:56:03.172'
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip("Currently unsupported timestamp literal formats")
def test_mysql_timestamp_literal(basic_df, spark_info, memory_leak_check):
    """tests a number of different timestamp formats that are currently supported by MySQL, but which we
    may/may not ultimatley end up supporting"""

    spark_query = """
    SELECT
        A, TIMESTAMP '2015-07-21', TIMESTAMP '2015-07-21', TIMESTAMP '2015-07-21', TIMESTAMP '2015-07-21', TIMESTAMP '2015-07-21',
    FROM
        table1
    """

    query = """
    SELECT
        A, TIMESTAMP '15-07-21', TIMESTAMP '2015/07/21', TIMESTAMP '20150721', TIMESTAMP '2020-12-01T01:52:03'
    FROM
        table1
    """

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_date_literal(basic_df, spark_info, memory_leak_check):
    """
    tests that the date keyword is correctly parsed/converted to timestamp by BodoSQL
    """
    query1 = """
    SELECT
        A, DATE '2015-07-21'
    FROM
        table1
    """

    check_query(
        query1,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_interval_literals(
    basic_df, spark_info, timedelta_equivalent_values, memory_leak_check
):
    """
    tests that interval literals are correctly parsed by BodoSQL
    """

    query = f"""
    SELECT
        A, INTERVAL {timedelta_equivalent_values[0]}
    FROM
        table1
    """

    expected = pd.DataFrame(
        {"A": basic_df["table1"]["A"], "time": timedelta_equivalent_values[1]}
    )

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected,
    )


def test_boolean_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that boolean literals are correctly parsed by BodoSQL
    """

    query = """
    SELECT
        A, TRUE, FALSE
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_boolean_literals_case_insensitivity(basic_df, spark_info, memory_leak_check):
    """
    tests that boolean literals are correctly parsed regardless of case by BodoSQL
    """

    query = """
    SELECT
        A, true as B, TrUe as C, false as D, FAlsE as E
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_string_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that string literals are correctly parsed by BodoSQL
    """

    query = """
    SELECT
        A, 'hello', '2015-07-21', 'true'
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip("[BE-957] Support Bytes.fromhex")
def test_binary_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that binary literals are correctly parsed by BodoSQL
    """

    query = """
    SELECT
        A, X'412412', X'STRING'
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_integer_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that integer literals are correctly parsed by BodoSQL
    """
    query = """
    SELECT
        A, 20150721, -13, 0 as Z, 1, -0
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip("[BS-154], float literals not comparing to spark properly")
def test_float_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that float literals are correctly parsed by BodoSQL
    """
    query = """
    SELECT
        A, .0103, -0.0, 13.2
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip("calcite parse error, uncertain if we ever want to support it")
def test_timestamp_null_literal(basic_df, spark_info, memory_leak_check):
    """
    tests that timestamp literals are correctly parsed by BodoSQL
    """
    query1 = """
    SELECT
        A, (CAST TIMESTAMP) NULL
    FROM
        table1
    """
    spark_query = """
    SELECT
        A, NULL
    FROM
        table1
    """
    check_query(
        query1,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.skip(
    "numba error converting none to a numpy dtype, not sure if we want to support this"
)
def test_boolean_null_literals(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    tests that boolean literals are correctly parsed by BodoSQL
    """
    query1 = """
    SELECT
        A and NULL
    FROM
        table1
    """
    query2 = """
    SELECT
        A or NULL
    FROM
        table1
    """
    check_query(
        query1,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        query2,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.skip(
    "numba error converting none to a numpy dtype, not sure if we want to support this"
)
def test_integer_null_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that integer literals are correctly parsed by BodoSQL
    """
    query1 = """
    SELECT
        A + NULL
    FROM
        table1
    """
    query2 = """
    SELECT
        A, -NULL
    FROM
        table1
    """
    check_query(
        query1,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
