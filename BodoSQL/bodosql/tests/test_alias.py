# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test file for tests related to alias operations. Some tests
    looks at the plans that are generated and check correctness.
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_plan_length, check_query


@pytest.mark.slow
def test_trival_project_removed(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that trivial project nodes are removed
    """
    query1 = "select * from table1"
    query2 = "select A, B, C from table1"
    check_plan_length(query1, basic_df, 1)
    check_query(query1, basic_df, spark_info)
    check_plan_length(query2, basic_df, 1)
    check_query(query2, basic_df, spark_info)


@pytest.mark.slow
def test_trival_nested_removed(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that all trivial project nodes are removed
    """
    query = "select * from (select * from table1)"
    check_plan_length(query, basic_df, 1)
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_simplified_projection(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that projections are merged or removed.
    """
    query1 = "select * from (select A, B from table1)"
    query2 = "select A, B from (select A, B from table1)"
    check_plan_length(query1, basic_df, 2)
    check_query(query1, basic_df, spark_info)
    check_plan_length(query2, basic_df, 2)
    check_query(query2, basic_df, spark_info)


@pytest.mark.slow
def test_necessary_projection(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that necessary projections aren't removed.
    """
    query1 = "select A, C from table1"
    query2 = "select C, A, B from table1"
    query3 = "select A, B as D, C from table1"
    check_plan_length(query1, basic_df, 2)
    check_query(query1, basic_df, spark_info)
    check_plan_length(query2, basic_df, 2)
    check_query(query2, basic_df, spark_info)
    check_plan_length(query3, basic_df, 2)
    check_query(query3, basic_df, spark_info)


@pytest.mark.slow
def test_renamed_projection(basic_df, spark_info, memory_leak_check):
    """
    Test that verifies that simple projections that just rename
    are fused together.
    """
    query = "select A, B as C from (select A, B from table1)"
    check_plan_length(query, basic_df, 2)
    check_query(query, basic_df, spark_info)


def test_aliasing_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test aliasing in queries"""
    check_query(
        "select A as testCol, C from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select sum(B) as testCol from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "select sum(B) as testCol1, sum(C) as testCol2 from table1 group by A",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )
    check_query(
        "select A, sum(b) as testCol from table1 where (A > 1 and A < 3) group by A order by testCol desc limit 10",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_as_on_colnames(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that the as operator is working correctly for aliasing columns
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
        convert_columns_bytearray1 = ["X"]
        convert_columns_bytearray2 = ["X", "Y"]
        convert_columns_bytearray3 = ["A", "Y"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None
    query1 = """
        SELECT
            A as X
        FROM
            table1
        """
    query2 = """
        SELECT
            A as X, B as Y
        FROM
            table1
        """
    query3 = """
        SELECT
            table1.A, table2.D as Y
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
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_as_on_tablenames(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that the as operator is working correctly for aliasing tablenames
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
        convert_columns_bytearray1 = ["A"]
        convert_columns_bytearray2 = ["A", "B"]
        convert_columns_bytearray3 = ["A", "D"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None
    query1 = """
        SELECT
            X.A
        FROM
            table1 as X
        """
    query2 = """
        SELECT
            X.A, X.B
        FROM
            table1 as X
        """
    query3 = """
        SELECT
            X.A, Y.D
        FROM
            table1 as X, table2 as Y
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
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_cyclic_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that aliasing that could be interpreted as cyclic works as intended
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
        convert_columns_bytearray = ["B", "C", "A"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
            A as B, B as C, C as A
        FROM
            table1
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_col_aliased_to_tablename(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that bodosql works correctly when the column names are aliased to table names
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
        convert_columns_bytearray = ["table2", "table1"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
           table1.C as table2, table2.A as table1
        FROM
            table1, table2
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_table_aliased_to_colname(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that bodosql works correctly when the table names are aliased to column names
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
        convert_columns_bytearray = ["C", "A"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
           A.C, C.A
        FROM
            table1 as A, table2 as C
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_multitable_renamed_projection(join_dataframes, spark_info, memory_leak_check):
    """
    Test that verifies that aliased projections from two different tables
    behave as expected.
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
        convert_columns_bytearray1 = ["A", "Y"]
        convert_columns_bytearray2 = ["Y", "B"]
        convert_columns_bytearray3 = ["B", "Y", "D"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None

    query = "SELECT table1.A, table2.D as Y FROM table1, table2"
    query2 = "SELECT table1.A as Y, table2.B FROM table1, table2"
    query3 = "SELECT table1.B, table2.A as Y, table2.D FROM table1, table2"
    check_query(
        query,
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
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_implicite_table_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Test that aliasing tables with the implicit syntax works as intended
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
        convert_columns_bytearray1 = ["A"]
        convert_columns_bytearray2 = ["Y", "B"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
    query = "SELECT T1.A from (SELECT * from table1) T1"
    query2 = "SELECT T1.Y, T2.B from (SELECT table2.A as Y FROM table2) T1, (SELECT table1.B FROM table1) T2"
    check_query(
        query,
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
