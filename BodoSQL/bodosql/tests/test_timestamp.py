# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries specific to Timestamp types on BodoSQL
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


def test_datetime_condition(spark_info, memory_leak_check):
    """test selecting column satisfying condition on timestamp type column"""
    dataframe_dict = {
        "table1": pd.DataFrame(
            {
                "A": [
                    np.datetime64("2011-01-01"),
                    np.datetime64("1971-02-02"),
                    np.datetime64("2021-03-03"),
                    np.datetime64("2004-12-07"),
                ]
                * 3,
            }
        )
    }
    check_query(
        "select A from table1 where A > '2016-02-12'", dataframe_dict, spark_info
    )


@pytest.mark.slow
def test_extract_date(spark_info, memory_leak_check):
    query = "SELECT EXTRACT(YEAR FROM A) FROM table1"
    dataframe_dict = {
        "table1": pd.DataFrame(
            {
                "A": [
                    np.datetime64("2011-01-01"),
                    np.datetime64("1971-02-02"),
                    np.datetime64("2021-03-03"),
                    np.datetime64("2004-12-27"),
                ]
                * 3,
            }
        )
    }
    check_query(query, dataframe_dict, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_datediff_literals(basic_df, spark_info, memory_leak_check):
    """
    Checks that calling DATEDIFF on literals behaves as expected
    """
    query = (
        "SELECT A, DATEDIFF(TIMESTAMP '2017-08-25', TIMESTAMP '2011-08-25') from table1"
    )
    query2 = (
        "SELECT A, DATEDIFF(TIMESTAMP '2017-08-25', TIMESTAMP '2011-08-25') from table1"
    )
    query3 = (
        "SELECT A, DATEDIFF(TIMESTAMP '2017-08-25', TIMESTAMP '2011-08-25') from table1"
    )
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_names=False, check_dtype=False)
    check_query(query3, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_str_to_date_literals(basic_df, spark_info, memory_leak_check):
    """
    Checks that calling STR_TO_DATE on literals behaves as expected
    """
    query = "SELECT A, STR_TO_DATE('17-09-2010', '%d-%m-%Y') from table1"
    spark_query = "SELECT A, TO_DATE('17-09-2010', 'dd-MM-yyyy') from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


def test_datediff_columns(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Checks that calling DATEDIFF on columns behaves as expected
    """
    query = "SELECT DATEDIFF(A, B) from table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_names=False, check_dtype=False
    )


@pytest.mark.slow
def test_datediff_multitable_columns(
    bodosql_datetime_types, spark_info, memory_leak_check
):
    """
    Checks that calling DATEDIFF on literals behaves as expected
    """
    bodosql_datetime_types["table2"] = bodosql_datetime_types["table1"]
    query = "SELECT DATEDIFF(table1.A, table2.B) from table1, table2"

    check_query(
        query, bodosql_datetime_types, spark_info, check_names=False, check_dtype=False
    )


def test_str_to_date_columns(spark_info, memory_leak_check):
    """
    Checks that calling STR_TO_DATE on columns behaves as expected
    """
    ctx = {
        "table1": pd.DataFrame({"A": ["2003-02-01", "2013-02-11", "2011-11-01"] * 4})
    }
    query = "SELECT STR_TO_DATE(A, '%Y-%m-%d') from table1"
    spark_query = "SELECT TO_DATE(A, 'yyyy-MM-dd') from table1"
    check_query(
        query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_str_to_date_columns_format(spark_info, memory_leak_check):
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
    spark_query = "SELECT TO_DATE(A, 'yyyy-MM-dd:hh') from table1"
    check_query(
        query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


def test_str_date_case_stmt(spark_info, memory_leak_check):
    """
    Many sql dialects play fast and loose with what is a string and date/timestamp
    This test checks that a case with a string and timestamp output behaves reasonably.
    """
    ctx = {
        "table1": pd.DataFrame(
            {
                "C": [0, 1, 13] * 4,
                "B": [
                    pd.Timestamp("2021-09-26"),
                    pd.Timestamp("2021-03-25"),
                    pd.Timestamp("2020-01-26"),
                ]
                * 4,
                "A": ["2021-09-26", "2021-03-25", "2020-01-26"] * 4,
            }
        )
    }

    query = "select CASE WHEN C = 1 THEN A ELSE B END from table1"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.fixture(
    params=[
        pytest.param(
            "2013-04-28T20:57:01.123456789+00:00",
            id='YYYY-MM-DD"T"HH24:MI:SS.FFTZH:TZM_no_offset',
        ),
        pytest.param(
            "2013-04-28T20:57:01.123456789+07:00",
            id='YYYY-MM-DD"T"HH24:MI:SS.FFTZH:TZM_with_offset',
        ),
    ]
)
def timestamp_literal(request):
    return request.param


@pytest.mark.skip("Needs calcite support")
def test_timestamp_from_utc_literal(timestamp_literal, memory_leak_check):
    """
    Checks that a timestamp can be created from a literal with a UTC offset
    """
    value = pd.Timestamp(timestamp_literal).tz_convert("UTC").tz_localize(None)
    query = f"SELECT TIMESTAMP '{timestamp_literal}' AS ts"
    ctx = {}
    expected_output = pd.DataFrame({"ts": value}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output)


def test_timestamp_cast_utc_literal(timestamp_literal, memory_leak_check):
    """
    Checks that a timestamp can be cast from a literal with a UTC offset
    """
    value = pd.Timestamp(timestamp_literal).tz_localize(None)
    query = f"SELECT CAST ('{timestamp_literal}' AS TIMESTAMP) AS ts"
    ctx = {}
    expected_output = pd.DataFrame({"ts": value}, index=np.arange(1))
    check_query(query, ctx, None, expected_output=expected_output)
