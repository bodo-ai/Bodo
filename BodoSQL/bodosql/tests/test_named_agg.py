# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries that require Pandas NamedAgg syntax.
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_named_agg(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests Named Aggregation by performing a groupby with columns in a different
    order than the list syntax allows.
    """
    query = """
            SELECT SUM(A) as a1, SUM(C) as b1, AVG(A) as c1
            from table1
            GROUP BY B
            """
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


@pytest.mark.slow
def test_groupby_quansite(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Group by test aimed to match the bug reported in the quansite
    thread.
    """
    query = "select A, AVG(C) from table1 group by A"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_groupby_multicolumn(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Group by test with multiple output columns
    """
    query = "select A, AVG(C) from table1 group by A, B"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )
