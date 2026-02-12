"""
Tests that bodoSQL correctly interprets literal values

For MySql reference used, see https://dev.mysql.com/doc/refman/8.0/en/literals.html
"""

import datetime

import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    enable_timestamptz,
    pytest_mark_one_rank,
    pytest_slow_unless_codegen,
)
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


# TODO(aneesh) support unquoted literals for subsecond intervals and update this
# test - we should also consider testing all the different abbreviations as well
@pytest.fixture(
    params=[
        ("1 DAY", pd.Timedelta(1, "D")),
        ("1 HOUR", pd.Timedelta(1, "h")),
        ("1 MINUTE", pd.Timedelta(1, "m")),
        ("1 SECOND", pd.Timedelta(1, "s")),
        ("'1 MILLISECOND'", pd.Timedelta(1, "ms")),
        ("'1 MICROSECOND'", pd.Timedelta(1, "us")),
        ("'1 NANOSECOND'", pd.Timedelta(1, "ns")),
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


@pytest.mark.skip(
    "Currently parses with calcite, unsupported timestamp due to a pandas error"
)
def test_timestamp_literal_pd_error(basic_df, spark_info, memory_leak_check):
    """This is a specific test case that is parsed correctly by calcite, but generates a runtime pandas error.
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


@pytest.mark.skip(
    "Unsupported timestamp format in Calcite: https://bodo.atlassian.net/browse/BE-3300"
)
def test_mysql_timestamp_literal(basic_df, spark_info, memory_leak_check):
    """tests a number of different timestamp formats that are currently supported by MySQL, but which we
    may/may not ultimately end up supporting"""

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


def test_timestamptz_literal(basic_df, spark_info, memory_leak_check):
    """
    tests that timestamptz literals are correctly parsed by BodoSQL
    """
    # TZ1 specifies a timezone, TZ2 does not (should use session TZ), and TZ3 specifies UTC
    query = """
    SELECT
        '2020-01-02 03:04:05.123456789 -0800'::timestamptz as tz1,
        '2020-01-02 03:04:05.123456789'::timestamptz as tz2,
        '2020-01-02 03:04:05.123456789 Z'::timestamptz as tz3
    FROM
        table1
    """

    with enable_timestamptz():
        length = basic_df["TABLE1"].shape[0]

        expected_output = pd.DataFrame(
            {
                "TZ1": [
                    bodo.types.TimestampTZ.fromLocal(
                        "2020-01-02 03:04:05.123456789", -480
                    )
                ]
                * length,
                # The session tz is +545
                "TZ2": [
                    bodo.types.TimestampTZ.fromLocal(
                        "2020-01-02 03:04:05.123456789", 345
                    )
                ]
                * length,
                "TZ3": [
                    bodo.types.TimestampTZ.fromUTC("2020-01-02 03:04:05.123456789", 0)
                ]
                * length,
            }
        )
        check_query(
            query,
            basic_df,
            None,
            session_tz="Asia/Kathmandu",
            check_dtype=False,
            expected_output=expected_output,
            enable_timestamp_tz=True,
        )


@pytest.mark.slow
def test_date_literal(basic_df, memory_leak_check):
    """
    tests that the date literal is correctly output by BodoSQL
    """
    query1 = """
    SELECT
        A, DATE '2015-07-21' as LIT
    FROM
        table1
    """
    py_output = pd.DataFrame(
        {"A": basic_df["TABLE1"]["A"], "LIT": datetime.date(2015, 7, 21)}
    )

    check_query(
        query1,
        basic_df,
        None,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_time_literal(basic_df, memory_leak_check):
    """
    tests that the time literal is correctly output by BodoSQL
    """
    query1 = """
    SELECT
        A, time '10:03:56' as LIT
    FROM
        table1
    """
    py_output = pd.DataFrame(
        {"A": basic_df["TABLE1"]["A"], "LIT": bodo.types.Time(10, 3, 56)}
    )

    check_query(
        query1,
        basic_df,
        None,
        expected_output=py_output,
    )


def test_interval_literals(
    basic_df, spark_info, timedelta_equivalent_values, memory_leak_check
):
    """
    tests that interval literals are correctly parsed by BodoSQL

    Note that since we are returning interval literals, we must restrict this
    test to only test the subset of literals that output non-relative times
    (i.e. ones that are compiled to Timedeltas and not DateOffsets)
    """

    query = f"""
    SELECT
        A, INTERVAL {timedelta_equivalent_values[0]}
    FROM
        table1
    """

    expected = pd.DataFrame(
        {"A": basic_df["TABLE1"]["A"], "time": timedelta_equivalent_values[1]}
    )

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected,
    )


@pytest.fixture(
    params=[
        ("1 day", lambda x: x + pd.DateOffset(days=1)),
        ("2 month", lambda x: x + pd.DateOffset(months=2)),
        ("3 year", lambda x: x + pd.DateOffset(years=3)),
        ("1 quarter", lambda x: x + pd.DateOffset(months=3)),
        ("10 seconds", lambda x: x + pd.Timedelta(seconds=10)),
        ("1 year, 1 quarter, 1 month", lambda x: x + pd.DateOffset(years=1, months=4)),
        (
            "1 year, 2 second",
            lambda x: (x + pd.DateOffset(years=1)) + pd.Timedelta(seconds=2),
        ),
        (
            "1 second, 2 year",
            lambda x: (x + pd.DateOffset(years=2)) + pd.Timedelta(seconds=1),
        ),
        (
            "1 year, 2 month, 3 second",
            lambda x: (x + pd.DateOffset(years=1, months=2)) + pd.Timedelta(seconds=3),
        ),
        (
            "1 days, 2 hrs, 3 mins, 4s, 5ms, 6us, 7ns",
            lambda x: x
            + pd.Timedelta(
                days=1,
                hours=2,
                minutes=3,
                seconds=4,
                milliseconds=5,
                microseconds=6,
                nanoseconds=7,
            ),
        ),
        (
            "1 months, 2 days, 3 hrs, 4 mins, 5s, 6ms, 7us, 8ns",
            lambda x: (
                x + pd.DateOffset(months=1, days=2, hours=3, minutes=4, seconds=5)
            )
            + pd.Timedelta(milliseconds=6, microseconds=7, nanoseconds=8),
        ),
    ]
)
def interval_addition_values(request):
    """fixture that returns a tuple of a interval string literal, and a
    corresponding function that would add the eequivalent offset to a
    timestamp"""
    return request.param


def test_interval_literals_addition(interval_addition_values, memory_leak_check):
    """
    tests that interval literal addtion is correctly parsed by BodoSQL
    This allows for testing all possible interval literals.
    """

    query = f"""
    SELECT
        A + INTERVAL {repr(interval_addition_values[0])}
    FROM
        table1
    """

    df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    pd.Timestamp(2020, 1, 2),
                    pd.Timestamp(2020, 1, 2, 3, 4, 5, 6, nanosecond=7),
                    pd.Timestamp(2020, 12, 31, 23, 59, 59, 999999, nanosecond=999),
                    pd.Timestamp(2016, 2, 27, 4, 30, 15, 50, nanosecond=5),
                ],
                dtype="datetime64[ns]",
            )
        }
    )

    expected = pd.DataFrame({"0": df["A"].apply(interval_addition_values[1])})

    check_query(
        query,
        {"TABLE1": df},
        None,
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


def test_binary_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that binary literals are correctly parsed by BodoSQL
    """

    query = """
    SELECT
        A, X'412412', X'A0F32C'
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        # Avoid sorting because bytes are unhashable right now and all rows are the same.
        sort_output=False,
    )


@pytest.mark.slow
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


@pytest.mark.slow
def test_float_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that float literals are correctly parsed by BodoSQL
    """
    query = """
    SELECT
        A, .0103 as B, -0.0 as C, 13.2 as D
    FROM
        table1
    """
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["B", "C", "D"],
    )


@pytest.mark.slow
def test_timestamp_null_literal(basic_df, spark_info, memory_leak_check):
    """
    tests that timestamp literals are correctly parsed by BodoSQL
    """
    query1 = """
    SELECT
        A, CAST(NULL as TIMESTAMP)
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


@pytest.mark.skip("[BE-4406] Support boxing null arrays")
def test_integer_null_literals(basic_df, spark_info, memory_leak_check):
    """
    tests that integer literals are correctly parsed by BodoSQL
    """
    query1 = """
    SELECT
        A, +NULL
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


def test_backslash_literals(spark_info, memory_leak_check):
    """
    tests that integer literals are correctly parsed by BodoSQL
    """
    query1 = r"""
    SELECT
        '\\' as A
    """
    query2 = r"""
    SELECT
        '\n' as A
    """
    check_query(
        query1,
        {},
        spark_info,
    )
    check_query(
        query2,
        {},
        spark_info,
    )


def test_large_day_literals(bodosql_date_types, memory_leak_check):
    """
    tests that Interval literals with large offsets are handled by BodoSQL.

    """
    query = "select A + Interval '180 Days' as output from table1"
    expected_output = pd.DataFrame(
        {"OUTPUT": bodosql_date_types["TABLE1"]["A"] + pd.Timedelta(days=180)}
    )
    check_query(
        query,
        bodosql_date_types,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            "1, 2, 3",
            pd.Series([[1, 2, 3]] * 5, dtype=pd.ArrowDtype(pa.large_list(pa.int64()))),
            id="integer_literals",
        ),
        pytest.param(
            "'a', 'b', 'c'",
            pd.Series(
                [["a", "b", "c"]] * 5, dtype=pd.ArrowDtype(pa.large_list(pa.string()))
            ),
            id="string_literals",
        ),
        pytest.param(
            "OBJECT_CONSTRUCT('a', 0, 'b', '1')",
            pd.Series(
                [[{"a": 0, "b": "1"}]] * 5,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("a", pa.int32()), pa.field("b", pa.string())]
                        )
                    )
                ),
            ),
            marks=pytest.mark.skip(reason="[BSE-2208] MAP array type is unsupported"),
            id="objects",
        ),
        pytest.param(
            "['a'], ['b'], ['c']",
            pd.Series(
                [[["a"], ["b"], ["c"]]] * 5,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.string()))),
            ),
            id="nested_strings",
        ),
        pytest.param(
            "0, A",
            pd.Series([[0, 1]] * 5, dtype=pd.ArrowDtype(pa.large_list(pa.int64()))),
            id="integers_mixed",
        ),
    ],
)
def test_array_literals(args, answer, memory_leak_check):
    query = f"SELECT [{args}] FROM table1"
    check_query(
        query,
        {"TABLE1": pd.DataFrame({"A": [1] * 5})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "select case when A=1 then [1, null, 2] when A=2 then null else [3] end from table1",
            pd.Series(
                [[1, None, 2], None, [1, None, 2], [3]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
            id="integer_literals",
            marks=pytest.mark.skip(
                reason="[BSE-2255] query plan has an invalid cast to non-nullable"
            ),
        ),
        pytest.param(
            "select case when A=1 then [1.2, null, 2.3] else [3.4] end from table1",
            pd.Series(
                [[1.2, None, 2.3], [3.4], [1.2, None, 2.3], [3.4]],
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
            ),
            id="float_literals",
        ),
        pytest.param(
            "select case when A=1 then ['2021-01-01 00:01:00 +0000'::timestamp, null, '2021-01-03 00:02:00 +0000'::timestamp] else ['2021-04-01 00:05:00 +0000'::timestamp] end from table1",
            pd.Series(
                [
                    [
                        pd.Timestamp("2021-01-01 00:01:00 +0000"),
                        None,
                        pd.Timestamp("2021-01-03 00:02:00 +0000"),
                    ],
                    [pd.Timestamp("2021-04-01 00:05:00 +0000")],
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.timestamp("ns"))),
            ),
            id="timestamp_literals",
        ),
    ],
)
def test_array_literals_case(query, answer, memory_leak_check):
    """Test array literals in CASE statements that may convert nullable type to
    non-nullable
    """
    check_query(
        query,
        {"TABLE1": pd.DataFrame({"A": [1, 2, 1, 4]})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest_mark_one_rank
def test_dollar_strings(datapath, memory_leak_check):
    """
    Test that parsing of dollar-enclosed strings matches the behavior
    described here: https://docs.snowflake.com/en/sql-reference/data-types-text#dollar-quoted-string-constants
    """
    with open(datapath("dollar_string_test.sql")) as f:
        query = f.read()
    answer = pd.DataFrame(
        {"idx1": list(range(1, 5)), "idx2": list(range(1, 5)), "L": [25, 55, 31, 7]}
    )
    check_query(
        query,
        {},
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        only_jit_seq=True,
        is_out_distributed=False,
    )


def test_null_variant_literal(memory_leak_check):
    answer = pd.DataFrame({"0": pd.array([None], dtype=pd.ArrowDtype(pa.null()))})
    check_query(
        "SELECT NULL::VARIANT",
        {},
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        only_jit_seq=True,
        is_out_distributed=False,
    )
