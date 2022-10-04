# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of Snowflake TO_X functions in BodoSQL
"""

# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""


import bodosql
import pandas as pd
import pytest
from bodosql.tests.test_datetime_fns import dt_fn_dataframe  # noqa
from bodosql.tests.utils import check_query, make_tables_nullable

from bodo.tests.bodosql_array_kernel_tests.test_bodosql_snowflake_date_conversion_array_kernels import (  # pragma: no cover
    scalar_to_date_equiv_fn,
)


@pytest.fixture(
    params=[
        ("2020-12-01T13:56:03.172:00",),
        ("2342-312",),
        ("2020-13-01",),
        ("-20200-15-15",),
        ("2100-12-01-01-01-01-01-01-01-01-01-01-01-01-100",),
        (pd.Series(["2022-02-18", "2022-14-18"] * 10),),
    ]
)
def invalid_to_date_args(request):
    """set of arguments which cause NA in try_to_date, and throw an error for to_date"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("TRY_TO_DATE", id="try_to_date"),
        pytest.param("TO_DATE", id="to_date"),
    ]
)
def test_fn(request):
    return request.param


def to_date_gen_case_query(fn_call):
    return f"SELECT CASE WHEN {test_fn}(A) < DATE '2013-01-03' THEN {test_fn}(A) ELSE {test_fn}(A) END from table1"


def test_to_date_ints(spark_info, basic_df, test_fn, memory_leak_check):
    """tests to_date on integer values"""
    query = f"SELECT {test_fn}(A) from table1"

    basic_null_df = make_tables_nullable(basic_df)

    expected_output = pd.DataFrame(
        {
            "foo": basic_null_df["table1"]["A"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
            )
        }
    )
    check_query(
        query,
        basic_null_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_ints_case(spark_info, basic_df, test_fn, memory_leak_check):
    """tests to_date on integer values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(A) < DATE '2013-01-03' THEN {test_fn}(A) ELSE {test_fn}(A) END from table1"

    basic_null_df = make_tables_nullable(basic_df)

    expected_output = pd.DataFrame(
        {
            "foo": basic_null_df["table1"]["A"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
                if scalar_to_date_equiv_fn(val)
                < pd.Timestamp("2013-01-03").to_datetime64()
                else None
            )
        }
    )
    check_query(
        query,
        basic_null_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_valid_strings(spark_info, dt_fn_dataframe, test_fn, memory_leak_check):
    """tests to_date on valid string values"""
    query = f"SELECT {test_fn}(datetime_strings) from table1"
    spark_query = f"SELECT TO_DATE(datetime_strings) from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_strings_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid string values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(datetime_strings) < DATE '2013-01-03' THEN {test_fn}(datetime_strings) END from table1"
    spark_query = f"SELECT CASE WHEN TO_DATE(datetime_strings) < DATE '2013-01-03' THEN TO_DATE(datetime_strings) END from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_digit_strings(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid digit string values in a case statment"""
    query = f"SELECT {test_fn}(digit_strings) from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
    expected_output = pd.DataFrame(
        {
            "foo": dt_fn_dataframe_nullable["table1"]["digit_strings"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
            )
        }
    )

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_valid_digit_strings_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid digit string values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(digit_strings) < DATE '2013-01-03' THEN {test_fn}(digit_strings) END from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
    expected_output = pd.DataFrame(
        {
            "foo": dt_fn_dataframe_nullable["table1"]["digit_strings"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
                if not (scalar_to_date_equiv_fn(val) is None)
                and (
                    scalar_to_date_equiv_fn(val)
                    < pd.Timestamp("2013-01-03").to_datetime64()
                )
                else None
            )
        }
    )
    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_valid_datetime_types(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid datetime values"""
    query = f"SELECT {test_fn}(timestamps) from table1"
    spark_query = f"SELECT TO_DATE(timestamps) from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_datetime_types_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid datetime values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(timestamps) < DATE '2013-01-03' THEN {test_fn}(timestamps) END from table1"
    spark_query = f"SELECT CASE WHEN TO_DATE(timestamps) < DATE '2013-01-03' THEN TO_DATE(timestamps) END from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_invalid_to_date_args(spark_info, dt_fn_dataframe, test_fn, memory_leak_check):
    """tests arguments which cause NA in try_to_date, and throw an error for to_date"""

    query = f"SELECT {test_fn}(invalid_dt_strings) from table1"

    if test_fn == "TRY_TO_DATE":
        expected_output = pd.DataFrame(
            {"foo": pd.Series([None] * len(dt_fn_dataframe["table1"]))}
        )
        dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
        check_query(
            query,
            dt_fn_dataframe_nullable,
            spark_info,
            check_dtype=False,
            check_names=False,
            expected_output=expected_output,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(Exception, match=msg):
            bc = bodosql.BodoSQLContext(dt_fn_dataframe)
            bc.sql(query)


def test_to_date_format_string_err(dt_fn_dataframe, test_fn, memory_leak_check):
    """
    Tests that using a format string with TO_DATE or TRY_TO_DATE fails with a reasonable error
    see https://bodo.atlassian.net/browse/BE-3614
    """

    query = f"SELECT {test_fn}(datetime_strings, 'foo') from table1"
    msg = f"Error, format string for {test_fn} not yet supported"
    with pytest.raises(Exception, match=msg):
        bc = bodosql.BodoSQLContext(dt_fn_dataframe)
        bc.sql(query)
