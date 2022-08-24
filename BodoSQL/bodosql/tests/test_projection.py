# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL projection queries on BodoSQL
"""
import pandas as pd
import pytest
from bodosql.tests.utils import bodo_version_older, check_query


@pytest.mark.slow
def test_literal_project(basic_df, spark_info, memory_leak_check):
    """test that simply selecting literal values without any associated tables works as intended"""
    # Note we provide basic df because JIT code cannot handle an empty dict.
    query1 = "select 12"

    check_query(
        query1,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )

    query2 = "select .23 as mycol"
    # Spark keeps .23 as a decimal by default
    check_query(
        query2,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["mycol"],
    )

    query3 = "select 'hello'"
    check_query(
        query3,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )

    query4 = "select true"
    check_query(
        query4,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )

    query5 = "Select true, 1, false, 0, 'hello world', 13"
    check_query(
        query5,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )

    # TODO: "[BE-957] Support Bytes.fromhex"
    # query6 = "Select X'{b'1313'.hex()}'"
    # check_query(
    #     query6,
    #     basic_df,
    #     spark_info,
    #     check_dtype=False,
    #     check_names=False,
    # )


def test_project_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test projection queries"""
    query = "select B, C from table1"
    # TODO: Update check_dtype to be False only on differing numeric types
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


def test_select_multitable(join_dataframes, spark_info, memory_leak_check):
    """test selecting columns from multiple tables"""
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["C", "D"]
    else:
        convert_columns_bytearray = None
    check_query(
        "select C, D from table1, table2",
        join_dataframes,
        spark_info,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_select_multitable_order_by(join_dataframes, spark_info, memory_leak_check):
    """test selecting and sorting columns from multiple tables"""
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["C", "D"]
    else:
        convert_columns_bytearray = None
    check_query(
        "select C, D from table1, table2 order by D, C",
        join_dataframes,
        spark_info,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_select_all_numerics(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Simplest query possible on numeric types.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


@pytest.mark.skip("[BS-45]")
def test_select_all_large_numerics(
    bodosql_large_numeric_types, spark_info, memory_leak_check
):
    """
    Simplest query possible on large numeric types.
    This checks to ensure values aren't truncated.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_large_numeric_types, spark_info, check_dtype=False)


def test_select_all_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Simplest query possible on datetime/timedelta types.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_datetime_types, spark_info, check_dtype=False)


def test_select_all_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Simplest query possible on timedelta types.
    """
    query = "SELECT * FROM table1"
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["A", "B", "C"],
    )


def test_select_all_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Simplest query possible on boolean tables.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_boolean_types, spark_info, check_dtype=False)


def test_select_all_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Simplest query possible on string tables.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_string_types, spark_info)


@pytest.mark.skipif(
    bodo_version_older(2022, 6, 3),
    reason="Requires next mini-release to maintain non-nullable scalars",
)
def test_select_expand_literal(basic_df, spark_info, memory_leak_check):
    """
    Tests select with all literals is expanded to
    the size of the input table.
    """
    query = "SELECT 1, 2, 'A' FROM table1"
    check_query(query, basic_df, spark_info, check_names=False)


@pytest.mark.slow
def test_select_literal_no_table(basic_df, spark_info, memory_leak_check):
    """
    Tests that literals can be selected with no input table.
    """
    query = "SELECT 1, 2, 'A'"
    check_query(query, basic_df, spark_info, check_names=False)


def test_select_all_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """
    Simplest query possible on nullable numeric tables.
    """
    query = "SELECT * FROM table1"
    check_query(query, bodosql_nullable_numeric_types, spark_info, check_dtype=False)


def test_select_from_simple(join_dataframes, spark_info, memory_leak_check):
    """
    tests that the select and from operators are working correctly for simple cases
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
        convert_columns_bytearray1 = ["A", "B"]
        convert_columns_bytearray2 = ["A", "D"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
    query1 = """
        SELECT
            A, B
        FROM
            table1
    """
    query2 = """
        SELECT
            table1.A, table2.D
        FROM
            table1, table2
        """
    check_query(
        query1,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray1,
    )
    check_query(
        query2,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray2,
    )


@pytest.mark.slow
def test_nested_select_from(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that simple nested SQL queries using only select and from work as intended
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
        convert_columns_bytearray = ["A", "C"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
            A, C
        FROM
            (SELECT
                A,C
            FROM
                table1)
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_heavily_nested_select_from(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that heavily nested SQL queries using only select and from work as intended
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
        convert_columns_bytearray = ["A"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
            A
        FROM
            (SELECT
                A
            FROM
                (SELECT
                    A, C
                FROM
                    (SELECT
                        *
                    FROM
                        table1
                        )
                )
            )
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_simple_df_loc(basic_df, spark_info, memory_leak_check):
    """
    Checks that a simple projection that just selects columns
    from a data frame uses df.loc[:, colnames]
    """
    # Check subset of columns
    query1 = "SELECT A, B from table1"
    result = check_query(query1, basic_df, spark_info, return_codegen=True)
    pandas_code = result["pandas_code"]
    # Check for df.loc
    assert ".loc" in pandas_code

    # Check a single column
    query2 = "SELECT A from table1"
    result = check_query(query2, basic_df, spark_info, return_codegen=True)
    pandas_code = result["pandas_code"]
    # Check for df.loc
    assert ".loc" in pandas_code
