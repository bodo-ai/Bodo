# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests correctness of the In and Not In operations in BodoSQL
"""

import pytest
from bodosql.tests.utils import check_query


def test_in_columns(basic_df, spark_info, memory_leak_check):
    "tests the in operation on column values"
    query = "SELECT 1 in (A,B,C) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_in_scalars(basic_df, spark_info, memory_leak_check):
    "tests the in operation on scalar values"
    query = "SELECT CASE WHEN 1 in (A,B,C) THEN -1 else 100 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


def test_not_in_columns(basic_df, spark_info, memory_leak_check):
    "tests the not in operation on column values"
    query = "SELECT 1 not in (A,B,C) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_not_in_scalars(basic_df, spark_info, memory_leak_check):
    "tests the not in operation on scalar values"
    query = "SELECT CASE WHEN 1 not in (A,B,C) THEN -1 else 100 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


def test_in_scalar_literals(basic_df, spark_info):
    """tests the in operation when comparing a column against a list of literals"""
    query = "SELECT A in (1, 3, 5) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_not_in_scalar_literals(basic_df, spark_info):
    """tests the not in operation when comparing a column against a list of literals"""
    query = "SELECT A not in (1, 3, 5) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_in_list_one_literal(basic_df, spark_info):
    """tests the in operation when comparing a column against a list of a single literal"""
    query = "SELECT A in (1) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_in_scalar_with_scalar_literals(basic_df, spark_info):
    """tests the in operation when comparing a scalar against a list of literals"""
    query = "SELECT CASE WHEN A in (1, 3, 5) THEN 1 ELSE 2 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_not_in_scalar_with_scalar_literals(basic_df, spark_info):
    """tests the not in operation when comparing a scalar against a list of literals"""
    query = "SELECT CASE WHEN A not in (1, 3, 5) THEN 1 ELSE 2 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)
