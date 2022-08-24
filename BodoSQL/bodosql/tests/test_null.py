"""
Test that SQL is Null/Not Null and IS_TRUE builtins work with Columns, scalars, and NULL values.
Also tests that arithmatic and logical operators work with NULL values
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query
from pyspark.sql.functions import lit


# This is mostly copied from bodosql.tests.utils, just needed to
# explicitly set the column to a null in such a way that
# Spark knows the type
# TODO: should refactor this into utils eventually
def get_expected_output_with_null_col_b(s_info, ctx, query, typestr):
    for table_name, df in ctx.items():
        s_info.catalog.dropTempView(table_name)
        s_info.createDataFrame(df).withColumn(
            "B", lit(None).cast(typestr)
        ).createTempView(table_name)
    expected_output = s_info.sql(query).toPandas()
    return expected_output


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": ["hello", "world", "how"] * 4,
                    "B": pd.Series([None, None, None] * 4, dtype="string"),
                }
            )
        },
    ]
)
def bodosql_null_string_df(request):
    """fixture that returns a table with A being a non-null column of string, and B being an entirely null column of string"""
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [True, False, True] * 4,
                    "B": pd.Series([None, None, None] * 4, dtype="boolean"),
                }
            )
        },
    ]
)
def bodosql_null_bool_df(request):
    """fixture that returns a table with A being a non-null bool col, and B being an entirely Null bool column"""
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [
                        pd.Timestamp(2021, 5, 19),
                        pd.Timestamp(1999, 12, 31),
                        pd.Timestamp(2020, 10, 11),
                        pd.Timestamp(2025, 1, 1),
                    ]
                    * 3,
                    "B": pd.Series(
                        [
                            np.datetime64("nat"),
                            np.datetime64("nat"),
                            np.datetime64("nat"),
                        ]
                        * 4,
                    ),
                }
            )
        },
    ]
)
def bodosql_null_timestamp_df(request):
    """fixture that returns a table with A being a non-null timestamp col, and B being an entirely Null timestamp column"""
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [1, 2, 3] * 4,
                    "B": pd.Series([None, None, None] * 4, dtype="Int32"),
                }
            )
        },
    ]
)
def bodosql_null_integer_df(request):
    """fixture that returns a table with A being a non-null timestamp col, and B being an entirely Null column"""
    return request.param


@pytest.mark.slow
def test_is_null_str_cols(bodosql_null_string_df, spark_info, memory_leak_check):
    """tests is null on str columns"""
    query = "Select A is Null, B is Null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_string_df, query, "string"
    )
    check_query(
        query,
        bodosql_null_string_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_not_null_str_cols(bodosql_null_string_df, spark_info, memory_leak_check):
    """tests is not null on str columns"""
    query = "Select B is not Null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_string_df, query, "string"
    )
    check_query(
        query,
        bodosql_null_string_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_null_str_scalar(bodosql_null_string_df, spark_info, memory_leak_check):
    """tests is_null on str scalars"""
    query = "Select CASE WHEN B is NULL then TRUE ELSE FALSE END, CASE WHEN A is NULL then TRUE ELSE FALSE END from table1"

    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_string_df, query, "string"
    )
    check_query(
        query,
        bodosql_null_string_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
        optimize_calcite_plan=False,
        only_python=True,
    )


@pytest.mark.slow
def test_null_binary_scalar(bodosql_binary_types, spark_info, memory_leak_check):
    """tests is_null on binary scalars"""
    query = "Select CASE WHEN B is NULL then TRUE ELSE FALSE END, CASE WHEN A is not NULL then TRUE ELSE FALSE END from table1"

    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        optimize_calcite_plan=False,
        only_python=True,
    )


@pytest.mark.slow
def test_is_null_bool_cols(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is null on bool columns"""
    query = "Select A is Null, B is Null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_not_null_bool_cols(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is not null on bool columns"""
    query = "Select B is not Null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_null_bool_scalar(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is null on boolean scalars"""
    query = "Select (CASE WHEN A is Null THEN True ELSE false END), (CASE WHEN B is Null THEN True ELSE false END) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_not_null_bool_scalar(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is not null on boolean scalars"""
    query = "Select (CASE WHEN A is not Null THEN True ELSE false END) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("TODO: issue with Spark infering the type")
def test_is_null_timestamp_cols(
    bodosql_null_timestamp_df, spark_info, memory_leak_check
):
    """tests is_null on timestamp columns"""
    query = "Select B, B is NULL from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_timestamp_df, query, "TimestampType"
    )
    check_query(
        query,
        bodosql_null_timestamp_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("TODO: issue with Spark infering the type")
@pytest.mark.slow
def test_is_not_null_timestamp_cols(
    bodosql_null_timestamp_df, spark_info, memory_leak_check
):
    """tests is_not_null on timestamp columns"""
    query = "Select B, B is not null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_timestamp_df, query, "TimestampType"
    )
    check_query(
        query,
        bodosql_null_timestamp_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("TODO: issue with Spark infering the type")
def test_is_null_timestamp_scalar(
    bodosql_null_timestamp_df, spark_info, memory_leak_check
):
    """tests is_null on timestampean normal/nullscalars"""
    query = "Select (CASE WHEN B is Null THEN True ELSE false) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_timestamp_df, query, "timestamp"
    )
    check_query(
        query,
        bodosql_null_timestamp_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("TODO: issue with Spark infering the type")
@pytest.mark.slow
def test_is_not_null_timestamp_scalar(
    bodosql_null_timestamp_df, spark_info, memory_leak_check
):
    """tests is not null on timestamp normal/null scalars"""
    query = "Select (CASE WHEN A is not Null THEN True ELSE false) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_timestamp_df, query, "timestamp"
    )
    check_query(
        query,
        bodosql_null_timestamp_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-162] BodoSQL Null and's/or's not equivalent to Spark's'")
def test_boolean_null_comparisons_column(
    bodosql_null_bool_df, spark_info, memory_leak_check
):
    """tests logical operators work between two columns"""
    query = "Select A and B, A or B from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_boolean_null_comparisons_scalar(
    bodosql_null_bool_df, spark_info, memory_leak_check
):
    """tests logical operators work between a scalar and a column"""
    query = "Select CASE WHEN A and B THEN TRUE ELSE FALSE END, CASE WHEN A or B THEN TRUE ELSE FALSE END from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


def test_is_true_cols(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is_true works on nullable boolean columns"""
    query1 = "Select A from table1 where (A is True)"
    query2 = "Select B from table1 where (B is True)"
    expected1 = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query1, "boolean"
    )
    expected2 = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query2, "boolean"
    )
    check_query(
        query1,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected1,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        query2,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected2,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
@pytest.mark.skip("[BS-187] is true/false unsupported in the null literal case")
def test_is_true_scalar(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is_true works on nullable boolean scalars"""
    query = "Select (CASE WHEN A is TRUE THEN True ELSE false END), (CASE WHEN B is TRUE THEN True ELSE false END) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_is_false_cols(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is_true works on nullable boolean columns"""
    query = "Select B from table1 where B is False"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
@pytest.mark.skip("[BS-187] is true/false unsupported in the null literal case")
def test_is_false_scalar(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests is_true works on nullable boolean scalars"""
    query = "Select (CASE WHEN A is false THEN True ELSE false END), (CASE WHEN B is false THEN True ELSE false END) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


def test_is_null_integer_cols(bodosql_null_integer_df, spark_info, memory_leak_check):
    """tests is null on interger columns"""
    query = "Select A is null, B is null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query, "Int"
    )
    check_query(
        query,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
        only_python=True,
    )


@pytest.mark.slow
def test_is_not_null_integer_cols(
    bodosql_null_integer_df, spark_info, memory_leak_check
):
    """tests is not null on interger columns"""
    query = "Select A, A is not null from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query, "Int"
    )
    check_query(
        query,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
@pytest.mark.slow
def test_is_null_int_scalars(bodosql_null_integer_df, spark_info, memory_leak_check):
    """tests is null works with arithmatic scalars"""
    query = "Select (CASE WHEN A is Null THEN True ELSE false) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query, "Int"
    )
    check_query(
        query,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
@pytest.mark.slow
def test_is_not_null_int_scalars(
    bodosql_null_integer_df, spark_info, memory_leak_check
):
    """tests arithmatic is not null works with arithmatic scalars"""
    query = "Select (CASE WHEN A is not Null THEN True ELSE false) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query, "Int"
    )
    check_query(
        query,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
def test_numeric_null_arithmatic_cols(
    bodosql_null_integer_df, arith_ops, spark_info, memory_leak_check
):
    """tests arithmatic with Null columns works as intended"""
    query1 = f"Select A {arith_ops} B from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query1, "int"
    )
    check_query(
        query1,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
@pytest.mark.slow
def test_numeric_null_arithmatic_scalar(
    bodosql_null_integer_df, arith_ops, spark_info, memory_leak_check
):
    """tests arithmatic with Null scalars works as intended"""
    query = f"Select CASE WHEN A {arith_ops} B > 0 THEN 1 WHEN A {arith_ops} B <= 0 THEN 0 END From table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query, "int"
    )
    check_query(
        query,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
def test_numeric_null_comparison_cols(
    bodosql_null_integer_df, comparison_ops, spark_info, memory_leak_check
):
    """tests comparisons with Null columns works as intended"""
    query1 = f"Select A from table1 where A {comparison_ops} B"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query1, "int"
    )
    check_query(
        query1,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-149]")
@pytest.mark.slow
def test_numeric_null_comparisons_scalar(
    bodosql_null_integer_df, comparison_ops, spark_info, memory_leak_check
):
    """tests comparisons with Null scalars works as intended"""
    query1 = f"Select (CASE WHEN A {comparison_ops} B THEN True WHEN not (A {comparison_ops} B) THEN false END) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_integer_df, query1, "int"
    )
    check_query(
        query1,
        bodosql_null_integer_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_not_scalar(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests that 'not' works on boolean scalars"""

    query1 = "Select (Case WHEN not A then 1 when not not A then 0 end) from table1"

    expected1 = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query1, "boolean"
    )

    check_query(
        query1,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected1,
        check_names=False,
        check_dtype=False,
    )


def test_not_scalar_null(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests the behavior of not on scalar nulls"""

    query = "Select (Case WHEN not B then 1 when not not A then 0 end) from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
        only_python=True,
    )


@pytest.mark.slow
def test_not_null_columns(bodosql_null_bool_df, spark_info, memory_leak_check):
    """tests logical not works on possibly null columns"""
    query = "Select NOT A, NOT B from table1"
    expected = get_expected_output_with_null_col_b(
        spark_info, bodosql_null_bool_df, query, "boolean"
    )
    check_query(
        query,
        bodosql_null_bool_df,
        spark_info,
        expected_output=expected,
        check_names=False,
        check_dtype=False,
    )
