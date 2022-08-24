# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test file for tests related to the "with" operator
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


def test_with(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that WITH works in the simple case for tablenames
    """
    query = "WITH FOO as (SELECT A FROM table1) SELECT FOO.A from FOO"
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_with_multiple_tables(join_dataframes, spark_info, memory_leak_check):
    """
    Test that verifies that WITH works for multiple tablenames
    """
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A", "D"]
    else:
        convert_columns_bytearray = None
    query = "WITH FOO as (SELECT table1.A FROM table1), FOO2 as (SELECT table2.D FROM table2) SELECT FOO.A, FOO2.D from FOO, FOO2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_with_select_tables(join_dataframes, spark_info, memory_leak_check):
    """
    Test that verifies that WITH aliasing works on created tables as well
    """
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A", "Y", "Z"]
    else:
        convert_columns_bytearray = None
    query = """
    WITH table3 as (
        SELECT
            table1.A, table2.D as Y
        FROM
            table1, table2
    )
    SELECT
        table3.A, table3.Y, table1.A as Z
    FROM
        table3, table1
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_nested_with(join_dataframes, spark_info, memory_leak_check):
    """
    Test that verifies that WITH aliasing works in the nested case
    """
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A", "Y"]
    else:
        convert_columns_bytearray = None
    query = """
    WITH foo1 as (
        With foo1 as (
            SELECT
                A, B, C
            FROM
                table1
        ),
        foo2 as (
            SELECT
                D, B
            FROM
                table2
        )
        SELECT
            foo1.A, foo2.D
        FROM
            foo1, foo2
    )
    SELECT
        foo1.A, table1.A as Y
    FROM
        foo1, table1
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )
