# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries containing orderby on BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


def test_logical_or_columns(bodosql_boolean_types, spark_info, memory_leak_check):
    """insures that logical logical or works  correctly for columns of boolean values"""
    query1 = "SELECT A or B from table1"
    check_query(
        query1, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.skip(
    "Columnwise AND behavior on NULL values not yet fixed in our Calcite visitor"
)
def test_logical_and_columns(bodosql_boolean_types, spark_info, memory_leak_check):
    """insures that logical and works correctly for columns of boolean values"""
    query2 = "SELECT A and B from table1"
    check_query(
        query2, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
@pytest.mark.skip(
    "Scalar AND behavior on NULL values not yet fixed in our Calcite visitor"
)
def test_logical_and_scalars(bodosql_boolean_types, spark_info, memory_leak_check):
    """insures that logical and works correctly for scalar boolean values"""
    query1 = "SELECT (CASE WHEN A and B THEN 0 ELSE 1 end) from table1"
    check_query(
        query1, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
@pytest.mark.skip("Scalar OR behavior on NULLs not yet fixed in our Calcite visitor")
def test_logical_or_scalars(bodosql_boolean_types, spark_info, memory_leak_check):
    """insures that logical or works correctly for scalar boolean values"""
    query2 = "SELECT (CASE WHEN A or B THEN 0 ELSE 1 end) from table1"
    check_query(
        query2, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.fixture(params=["TRUE", "FALSE", "UNKNOWN"])
def is_clause_literal_values(request):
    return request.param


def test_logical_is_columns(
    bodosql_boolean_types, spark_info, is_clause_literal_values, memory_leak_check
):
    """insures that logical IS NOT works correctly for columns of boolean values"""
    query = f"SELECT A IS {is_clause_literal_values} from table1"
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_logical_is_scalars(
    bodosql_boolean_types, spark_info, is_clause_literal_values, memory_leak_check
):
    """insures that logical is works correctly for scalar boolean values"""
    query = f"SELECT (CASE WHEN A is {is_clause_literal_values} THEN 0 ELSE 1 end) from table1"
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_is_logical_not_columns(
    bodosql_boolean_types, spark_info, is_clause_literal_values, memory_leak_check
):
    """insures that logical IS works correctly for columns of boolean values"""
    query = f"SELECT A IS NOT {is_clause_literal_values} from table1"
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
@pytest.mark.skip("Is not True/False currently failing due to null issues, see BS-485")
def test_logical_is_not_scalars(
    bodosql_boolean_types, spark_info, is_clause_literal_values, memory_leak_check
):
    """insures that logical is works correctly for scalar boolean values"""
    query = f"SELECT (CASE WHEN A is not {is_clause_literal_values} THEN 0 ELSE 1 end) from table1"
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )
