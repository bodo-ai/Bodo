# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL comparison operations on BodoSQL
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param("BETWEEN", id="between"),
        pytest.param("NOT BETWEEN", marks=pytest.mark.slow, id="not_between"),
    ]
)
def between_clause(request):
    return request.param


def test_comparison_operators_numeric_within_table(
    bodosql_numeric_types,
    comparison_ops,
    spark_info,
    # seems to be leaking memory sporatically
    # memory_leak_check
):
    """
    Tests that the basic comparison operators work with numeric data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


def test_comparison_operators_binary_within_table(
    bodosql_binary_types, comparison_ops, spark_info, memory_leak_check
):
    """
    Tests that the basic comparison operators work with Binary data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        convert_columns_bytearray=["A", "C"],
    )


def test_comparison_operators_string_within_table(
    bodosql_string_types, comparison_ops, spark_info
):
    # TODO: Enable memory leak check
    """
    Tests that the basic comparison operators work with String data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_comparison_operators_datetime_within_table(
    bodosql_datetime_types, comparison_ops, spark_info, memory_leak_check
):
    """
    Tests that the basic comparison operators work with Timestamp data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    check_query(query, bodosql_datetime_types, spark_info, check_dtype=False)


def test_comparison_operators_interval_within_table(
    bodosql_interval_types, comparison_ops, spark_info, memory_leak_check
):
    """
    Tests that the basic comparison operators work with Timedelta data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["A", "C"],
    )


def test_comparison_operators_between_tables(
    bodosql_numeric_types, comparison_ops, spark_info, memory_leak_check
):
    """
    Tests that the basic comparison operators work when comparing data between two numeric tables of the same type
    """
    if comparison_ops != "=":
        # TODO: Add support for cross-join from general-join cond
        return
    query = f"""
        SELECT
            table1.A, table2.B
        FROM
            table1, table2
        WHERE
            table1.A {comparison_ops} table2.B
        """
    new_context = {
        "table1": bodosql_numeric_types["table1"],
        "table2": bodosql_numeric_types["table1"],
    }
    check_query(query, new_context, spark_info, check_dtype=False)


def test_where_and(join_dataframes, spark_info, memory_leak_check):
    """
    Tests an and expression within a where clause.
    """
    # For join dataframes, A and B must share a common type across both tables

    if isinstance(join_dataframes["table1"]["A"].values[0], bytes):
        pytest.skip(
            "No support for binary literals: https://bodo.atlassian.net/browse/BE-3304"
        )
    elif isinstance(join_dataframes["table1"]["A"].values[0], str):
        assert isinstance(join_dataframes["table2"]["A"].values[0], str)
        scalar_val1 = "'" + join_dataframes["table1"]["A"].values[0] + "'"
        scalar_val2 = "'" + join_dataframes["table2"]["A"].values[0] + "'"
    else:
        scalar_val1, scalar_val2 = (3, 4)

    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    query = f"""SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
               WHERE
                 (t1.A = {scalar_val1}
                  and t2.A = {scalar_val2})
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=check_dtype,
    )


def test_where_or(join_dataframes, spark_info, memory_leak_check):
    """
    Tests an or expression within a where clause.
    """
    # For join dataframes, A and B must share a common type across both tables
    if isinstance(join_dataframes["table1"]["A"].values[0], bytes):
        pytest.skip(
            "No support for binary literals: https://bodo.atlassian.net/browse/BE-3304"
        )
    elif isinstance(join_dataframes["table1"]["A"].values[0], str):
        assert isinstance(join_dataframes["table2"]["A"].values[0], str)
        scalar_val1 = "'" + join_dataframes["table1"]["A"].values[0] + "'"
        scalar_val2 = "'" + join_dataframes["table2"]["A"].values[-1] + "'"
    else:
        scalar_val1, scalar_val2 = (3, 4)

    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    query = f"""SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
               WHERE
                 (t1.A = {scalar_val1}
                  or t2.A = {scalar_val2})
    """
    check_query(
        query, join_dataframes, spark_info, check_names=False, check_dtype=check_dtype
    )


def test_between_date(spark_info, between_clause, memory_leak_check):
    query = f"""SELECT A {between_clause} DATE '1995-01-01'
                 AND DATE '1996-12-31' FROM table1"""
    dataframe_dict = {
        "table1": pd.DataFrame(
            {
                "A": [
                    np.datetime64("1996-12-31"),
                    np.datetime64("1995-01-01"),
                    np.datetime64("1996-01-01"),
                    np.datetime64("1997-01-01"),
                ]
                * 3,
            }
        )
    }
    if between_clause == "NOT BETWEEN":
        spark_query = (
            f"""SELECT (A < DATE '1995-01-01') OR (A > DATE '1996-12-31') FROM table1"""
        )
    else:
        spark_query = None
    check_query(
        query,
        dataframe_dict,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_between_interval(
    bodosql_interval_types, between_clause, spark_info, memory_leak_check
):
    """
    tests that between works for interval values
    """
    query = f"""
        SELECT
            A, B, C
        FROM
            table1
        WHERE
            table1.A {between_clause} Interval 1 SECOND AND Interval 1 DAY
    """

    # Using integer values for spark, since spark casts the input
    # pd timedeltas to integer ns values.
    if between_clause == "NOT BETWEEN":
        spark_query = f"""
        SELECT
            A, B, C
        FROM
            table1
        WHERE
            NOT (table1.A BETWEEN 1000000000 AND 86400000000000)
    """
    else:
        spark_query = f"""
        SELECT
            A, B, C
        FROM
            table1
        WHERE
            table1.A BETWEEN 1000000000 AND 86400000000000
    """

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        convert_columns_timedelta=["A", "B", "C"],
    )


def test_between_int(
    bodosql_numeric_types, between_clause, spark_info, memory_leak_check
):
    """
    tests that between works for integer values
    """
    query = f"""
        SELECT
            table1.A
        FROM
            table1
        WHERE
            table1.A {between_clause} 1 AND 3
    """

    if between_clause == "NOT_BETWEEN":
        spark_query = f"""
        SELECT
            table1.A
        FROM
            table1
        WHERE
            NOT (table1.A BETWEEN 1 AND 3)
    """
    else:
        spark_query = None
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_between_str(
    bodosql_string_types, between_clause, spark_info, memory_leak_check
):
    """
    tests that between works for string values
    """
    query = f"""
        SELECT
            *
        FROM
            table1
        WHERE
            table1.A {between_clause} 'a' AND 'z'
    """

    if between_clause == "NOT_BETWEEN":
        spark_query = f"""
        SELECT
            *
        FROM
            table1
        WHERE
            NOT (table1.A BETWEEN 'a' AND 'z')
    """
    else:
        spark_query = None

    check_query(query, bodosql_string_types, spark_info, check_dtype=False)
