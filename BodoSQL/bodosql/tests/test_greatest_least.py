"""
Tests correctness of the 'Greatest' keyword in BodoSQL
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.timezone_common import representative_tz  # noqa
from bodosql.tests.utils import check_query


@pytest.fixture(params=["GREATEST", "LEAST"])
def greatest_or_least(request):
    return request.param


def greatest_least_output_func(row, op):
    """
    Semantics for greatest/least to use inside a
    DataFrame.apply. Spark skips null values which differs
    from BodoSQL/typical SQL/Snowflake, so we cannot compare
    directly.
    """
    if (
        row["A"] is None
        or pd.isna(row["A"])
        or row["B"] is None
        or pd.isna(row["B"])
        or row["C"] is None
        or pd.isna(row["C"])
    ):
        return None
    if op == "GREATEST":
        return row.max()
    else:
        return row.min()


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


def test_greatest_float_literals(
    basic_df, spark_info, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on float literals
    """
    query1 = f"""
    SELECT
        A, {greatest_or_least}(.1, .111, -1.1, 1.75, .5, .01) as output
    FROM
        table1
    """
    query2 = f"""
    SELECT
        A, {greatest_or_least}(-.1, -.111, -1.1, -1.75, -.5, -.01) as output
    FROM
        table1
    """
    check_query(
        query1,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_decimal=["output"],
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_decimal=["output"],
    )


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


def test_greatest_binary_columns(
    bodosql_binary_types, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on binary columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as out_col
    FROM
        table1
    """
    df = bodosql_binary_types["TABLE1"]

    expected_output = pd.DataFrame(
        {
            "OUT_COL": df.apply(
                greatest_least_output_func, axis=1, args=(greatest_or_least,)
            )
        }
    )
    check_query(query, bodosql_binary_types, None, expected_output=expected_output)


def test_greatest_bool_columns(
    bodosql_boolean_types, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest and Least work on boolean columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as out_col
    FROM
        table1
    """
    df = bodosql_boolean_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "OUT_COL": df.apply(
                greatest_least_output_func, axis=1, args=(greatest_or_least,)
            )
        }
    )
    check_query(query, bodosql_boolean_types, None, expected_output=expected_output)


def test_greatest_date_literals(spark_info, greatest_or_least, memory_leak_check):
    """
    tests that Greatest works on date literals.
    """
    query = f"""
    SELECT
        {greatest_or_least}(Date '2022-1-1',Date '2022-10-1',Date '2022-1-10')
    """
    check_query(query, {}, spark_info, check_dtype=False, check_names=False)


def test_greatest_date_columns(
    bodosql_date_types, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on date columns.
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as out_col
    FROM
        table1
    """
    df = bodosql_date_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "OUT_COL": df.apply(
                greatest_least_output_func, axis=1, args=(greatest_or_least,)
            )
        }
    )
    check_query(
        query,
        bodosql_date_types,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_greatest_timestamp_literals(spark_info, greatest_or_least, memory_leak_check):
    """
    tests that Greatest works on timestamp literals
    """
    query = f"""
    SELECT
        {greatest_or_least}(Timestamp '2022-1-1', Timestamp '2022-10-1', Timestamp '2022-1-1 10:35:32')
    """
    check_query(query, {}, spark_info, check_dtype=False, check_names=False)


def test_greatest_timestamp_columns(
    bodosql_datetime_types, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on timestamp columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as out_col
    FROM
        table1
    """
    df = bodosql_datetime_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "OUT_COL": df.apply(
                greatest_least_output_func, axis=1, args=(greatest_or_least,)
            )
        }
    )
    check_query(query, bodosql_datetime_types, None, expected_output=expected_output)


def test_greatest_tz_aware_columns(
    representative_tz, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on tz_aware timestamp columns
    """
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="3/1/2022", freq="4h", periods=30, tz=representative_tz, unit="ns"
            ),
            "B": pd.date_range(
                start="2/18/2022",
                freq="1D5h",
                periods=30,
                tz=representative_tz,
                unit="ns",
            ),
            "C": pd.date_range(
                start="1/1/2022", freq="5D", periods=30, tz=representative_tz, unit="ns"
            ),
        }
    )
    ctx = {"TABLE1": df}
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as output
    FROM
        table1
    """
    if greatest_or_least == "GREATEST":
        S = df.max(axis=1)
    else:
        S = df.min(axis=1)
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


def test_least_datetime_strings(memory_leak_check):
    """
    tests that Least works with datetimes + valid strings (to be converted to datetimes)
    """

    df = pd.DataFrame(
        {
            "A": pd.Series(
                [pd.Timestamp("2000-08-17"), pd.Timestamp("1999-08-17")] * 3,
                dtype="datetime64[ns]",
            ),
            "B": pd.Series(
                [None, pd.Timestamp("1999-08-17")] * 3, dtype="datetime64[ns]"
            ),
            "C": pd.Series(["1999-09-17", "1999-09-17"] * 3),
        }
    )
    ctx = {"TABLE1": df}

    query = "SELECT LEAST(A,B,C) as output FROM table1"
    S = (
        df.astype(np.dtype("datetime64[ns]"))
        .min(axis=1, skipna=False)
        .map(lambda s: pd.NA if pd.isna(s) else str(s))
    )
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.slow
def test_single_column_least_greatest(greatest_or_least, memory_leak_check):
    """
    tests that Greatest/Least works with single column
    """

    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    ctx = {"TABLE1": df}

    query = f"SELECT {greatest_or_least}(A) as output FROM table1"
    py_output = pd.DataFrame({"OUTPUT": df.A})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


def test_greatest_time_literals(greatest_or_least, memory_leak_check):
    """
    tests that Greatest works on time literals
    """
    query = f"""
    SELECT
        {greatest_or_least}(TO_TIME('17:24:57'), TO_TIME('04:19:46'), TO_TIME('10:35:32'))
    """
    if greatest_or_least == "GREATEST":
        answer = pd.DataFrame({"A": pd.Series([bodo.types.Time(17, 24, 57)])})
    else:
        answer = pd.DataFrame({"A": pd.Series([bodo.types.Time(4, 19, 46)])})
    check_query(
        query,
        {},
        None,
        check_names=False,
        expected_output=answer,
        is_out_distributed=False,
    )


def test_greatest_time_columns(
    bodosql_time_types, greatest_or_least, memory_leak_check
):
    """
    tests that Greatest works on time columns
    """
    query = f"""
    SELECT
        {greatest_or_least}(A,B,C) as out_col
    FROM
        table1
    """
    df = bodosql_time_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "OUT_COL": df.apply(
                greatest_least_output_func, axis=1, args=(greatest_or_least,)
            )
        }
    )
    check_query(query, bodosql_time_types, None, expected_output=expected_output)
