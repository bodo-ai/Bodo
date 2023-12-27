# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests that bodoSQL correctly interprets literal values

For MySql reference used, see https://dev.mysql.com/doc/refman/8.0/en/literals.html
"""

import pandas as pd
import pyarrow as pa
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


# TODO(aneesh) support unquoted literals for subsecond intervals and update this
# test - we should also consider testing all the different abbreviations as well
@pytest.fixture(
    params=[
        ("1 DAY", pd.Timedelta(1, "D")),
        ("1 HOUR", pd.Timedelta(1, "H")),
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


@pytest.mark.slow
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
        {"OUTPUT": bodosql_date_types["table1"]["A"] + pd.Timedelta(days=180)}
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
        {"table1": pd.DataFrame({"A": [1] * 5})},
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
        {"table1": pd.DataFrame({"A": [1, 2, 1, 4]})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )
