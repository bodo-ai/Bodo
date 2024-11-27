"""
Tests correctness of the In and Not In operations in BodoSQL
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query


def check_codegen_uses_optimized_is_in(codegen):
    return "bodosql.kernels.is_in" in codegen


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


def test_in_scalar_literals(basic_df, spark_info, memory_leak_check):
    """tests the in operation when comparing a column against a list of literals"""
    query = "SELECT A in (1, 3, 5) from table1"
    output = check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        return_codegen=True,
    )
    assert check_codegen_uses_optimized_is_in(output["pandas_code"])


def test_string_in_scalar_literals(spark_info):
    """tests the in operation when using string literals, possibly with quotes"""
    query = "SELECT A in ('a', '\"happy\"', 'smi\"le') from table1"
    df = pd.DataFrame(
        {"A": ["a", "A", None, "happy", '"happy"', "smile", '"smile"', 'smi"le'] * 3}
    )
    ctx = {"TABLE1": df}
    output = check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        return_codegen=True,
    )
    assert check_codegen_uses_optimized_is_in(output["pandas_code"])


@pytest.mark.slow
def test_not_in_scalar_literals(basic_df, spark_info, memory_leak_check):
    """tests the not in operation when comparing a column against a list of literals"""
    query = "SELECT A not in (1, 3, 5) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_in_list_one_literal(basic_df, spark_info, memory_leak_check):
    """tests the in operation when comparing a column against a list of a single literal"""
    query = "SELECT A in (1) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_in_scalar_with_scalar_literals(basic_df, spark_info, memory_leak_check):
    """tests the in operation when comparing a scalar against a list of literals"""
    query = "SELECT CASE WHEN A in (1, 3, 5) THEN 1 ELSE 2 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_not_in_scalar_with_scalar_literals(basic_df, spark_info, memory_leak_check):
    """tests the not in operation when comparing a scalar against a list of literals"""
    query = "SELECT CASE WHEN A not in (1, 3, 5) THEN 1 ELSE 2 END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)
