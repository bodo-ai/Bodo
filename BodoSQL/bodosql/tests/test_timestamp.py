# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries specific to Timestamp types on BodoSQL
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.utils import pytest_slow_unless_codegen

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


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


@pytest.mark.parametrize(
    "query, expected_output",
    [
        pytest.param(
            "SELECT DATEDIFF(YEAR, TIMESTAMP '2017-02-28', TIMESTAMP '2011-12-25')",
            pd.DataFrame({"A": pd.Series([-6])}),
            id="year",
        ),
        pytest.param(
            "SELECT TIMEDIFF('QUARTER', TIMESTAMP '2017-02-28', TIMESTAMP '2011-12-25')",
            pd.DataFrame({"A": pd.Series([-21])}),
            id="quarter",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF(MONTH, TIMESTAMP '2017-02-28', TIMESTAMP '2011-12-25')",
            pd.DataFrame({"A": pd.Series([-62])}),
            id="month",
        ),
        pytest.param(
            "SELECT TIMEDIFF('WEEK', TIMESTAMP '2017-02-28', TIMESTAMP '2011-12-25')",
            pd.DataFrame({"A": pd.Series([-271])}),
            id="week",
        ),
        pytest.param(
            "SELECT DATEDIFF(DAY, TIMESTAMP '2017-02-28', TIMESTAMP '2011-12-25')",
            pd.DataFrame({"A": pd.Series([-1892])}),
            id="day",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('HOUR', TIMESTAMP '2017-02-28 10:10:10', TIMESTAMP '2011-12-25 22:33:33')",
            pd.DataFrame({"A": pd.Series([-45396])}),
            id="hour",
        ),
        pytest.param(
            "SELECT DATEDIFF(MINUTE, TIMESTAMP '2017-02-28 12:10:05', TIMESTAMP '2011-12-25 10:10:10')",
            pd.DataFrame({"A": pd.Series([-2724600])}),
            id="minute",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('SECOND', TIMESTAMP '2011-02-28 22:33:33', TIMESTAMP '2017-12-25 12:10:05')",
            pd.DataFrame({"A": pd.Series([215271392])}),
            id="second",
        ),
    ],
)
@pytest.mark.slow
def test_datediff_args3_literals(query, expected_output, basic_df, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF on timestamp literals behaves as expected.
    Tests all possible datetime parts except for subsecond units.

    Timestamp precision for literals in Calcite is currently limited to seconds
    and will be addressed in a follow-up issue: [BE-4272]

    """

    check_query(
        query,
        basic_df,
        spark=None,
        expected_output=expected_output,
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
    Checks that calling DATEDIFF on columns behaves as expected
    """
    query = "SELECT DATEDIFF(table1.A, table1.B) from table1"

    check_query(
        query, bodosql_datetime_types, spark_info, check_names=False, check_dtype=False
    )


@pytest.mark.parametrize(
    "query, spark_query",
    [
        pytest.param(
            "select DATEDIFF('YEAR', table1.A, table1.B) from table1",
            "select (year(table1.B) - year(table1.A)) from table1",
            id="year",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "select DATEDIFF('DAY', table1.A, table1.B) from table1",
            "select DATEDIFF(table1.B, table1.A) from table1",
            id="day",
        ),
    ],
)
@pytest.mark.slow
def test_datediff_args3_multitable_columns_date(
    query, spark_query, bodosql_datetime_types, spark_info, memory_leak_check
):
    """
    Checks that calling DATEDIFF with date parts on columns behaves as expected.

    Used https://kontext.tech/article/830/spark-date-difference-in-seconds-minutes-hours
    for defining equivalent Spark queries
    """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "query, spark_query",
    [
        pytest.param(
            "select DATEDIFF('SECOND', table1.A, table2.B) from table1, table2",
            "select ((bigint(to_timestamp(table2.B))-bigint(to_timestamp(table1.A)))) from table1, table2",
            id="second",
        ),
        pytest.param(
            "select TIMEDIFF('MILLISECOND', table1.A, table2.B) from table1, table2",
            "select ((bigint(to_timestamp(table2.B))-bigint(to_timestamp(table1.A))) * 1000) from table1, table2",
            id="millisecond",
        ),
        pytest.param(
            "select DATEDIFF('MICROSECOND', table1.A, table2.B) from table1, table2",
            "select ((bigint(to_timestamp(table2.B))-bigint(to_timestamp(table1.A))) * 1000 * 1000) from table1, table2",
            id="microsecond",
        ),
        pytest.param(
            "select TIMESTAMPDIFF('NANOSECOND', table1.A, table2.B) from table1, table2",
            "select ((bigint(to_timestamp(table2.B))-bigint(to_timestamp(table1.A))) * 1000 * 1000 * 1000) from table1, table2",
            id="nanosecond",
        ),
    ],
)
@pytest.mark.slow
def test_datediff_args3_multitable_columns_time(
    query, spark_query, bodosql_datetime_types_small, spark_info, memory_leak_check
):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF with time parts on columns behaves as expected.

    Used https://kontext.tech/article/830/spark-date-difference-in-seconds-minutes-hours
    for defining equivalent Spark queries
    """
    check_query(
        query,
        bodosql_datetime_types_small,
        spark=spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_datediff_args3_multitable_columns_case(
    bodosql_datetime_types, spark_info, memory_leak_check
):
    """
    Checks that calling DATEDIFF on columns in a case statement
    works as expected.
    """
    spark_datediff = "DATEDIFF(table1.B, table1.A)"
    bodo_datediff = "DATEDIFF('DAY', table1.A, table1.B)"

    query = f"SELECT CASE WHEN {bodo_datediff} > 30 THEN {bodo_datediff} ELSE -1 END as res from table1"
    spark_query = f"SELECT CASE WHEN {spark_datediff} > 30 THEN {spark_datediff} ELSE -1 END as res from table1"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


def test_datediff_tz_aware_tz_naive(memory_leak_check):
    """
    Checks that calling DATEDIFF between tz_aware and tz_naive timestamps
    gives a correct result.

    Note this result is copied from an equivalent Snowflake query.
    """
    df = pd.DataFrame(
        {
            "A": pd.array(
                [
                    pd.Timestamp("2021-09-26 01:00:13", tz="US/Pacific"),
                    pd.Timestamp("2023-10-28 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2023-11-05 05:15:17", tz="US/Pacific"),
                ]
                * 5
            ),
            "B": pd.array(
                [
                    pd.Timestamp("2024-09-01 07:23:14"),
                    pd.Timestamp("2023-10-28 00:00:00"),
                    pd.Timestamp("2022-07-15 14:23:51"),
                ]
                * 5
            ),
        }
    )
    ctx = {"table1": df}
    # Note this was verified against Snowflake by running with each scalar directly
    py_output = pd.DataFrame(
        {
            "out1": pd.array([92557381, 0, -41269886] * 5),
            "out2": pd.array([-25710, 0, 11464] * 5),
        }
    )
    query = (
        "SELECT DATEDIFF('s', A, B) as out1, DATEDIFF('H', B, A) as out2 from table1"
    )
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


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
            marks=pytest.mark.slow,
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
