"""
Test that Named Parameters are appropriately handled in situations where we have to generate
dataframe applies.
"""
import re

# Copyright (C) 2022 Bodo Inc. All rights reserved.
import pandas as pd
import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.fixture
def many_params_fixture():
    return {
        "a": True,
        "b": False,
        "c": 12,
        "d": "hello",
        "e": "hello2",
        "f": pd.Timestamp("2021-10-14 15:32:28.400163"),
        "g": pd.Timestamp("2019-03-04 11:12:28"),
    }


@pytest.mark.slow
def test_case(basic_df, spark_info, many_params_fixture, memory_leak_check):
    """tests that the params are properly passed as arguments when generate an apply due to a case stmt"""
    query = "select CASE WHEN @a THEN @a WHEN @b THEN @b WHEN @c > 12 THEN FALSE WHEN @d ='hello' THEN FALSE WHEN @e = 'hello' THEN FALSE WHEN @f > TIMESTAMP '2021-10-14' THEN FALSE WHEN @g > TIMESTAMP '2021-10-14' THEN FALSE ELSE A > 1 END from table1"
    pandas_code = check_query(
        query,
        basic_df,
        spark_info,
        named_params=many_params_fixture,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # Check pandas code has appropriate applies
    regexp = re.compile(
        r".*bodosql_case_placeholder.*_PARAM_g\s*=\s*_PARAM_g, _PARAM_f\s*=\s*_PARAM_f, _PARAM_e\s*=\s*_PARAM_e, _PARAM_d\s*=\s*_PARAM_d, _PARAM_c\s*=\s*_PARAM_c, _PARAM_b\s*=\s*_PARAM_b, _PARAM_a\s*=\s*_PARAM_a.*"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_repeated_param_usage(
    basic_df, spark_info, many_params_fixture, memory_leak_check
):
    """tests that the params are properly passed as arguments when a single param is used repeatedly in an apply"""
    query = "select CASE WHEN A > 10 THEN @c + 1 WHEN A > 100 THEN @c + 2 WHEN A > 1000 THEN @c + 3 ELSE @c + 4 END from table1"

    pandas_code = check_query(
        query,
        basic_df,
        spark_info,
        named_params=many_params_fixture,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # check pandas code has appropriate applies
    regexp = re.compile(r".*bodosql_case_placeholder.*_PARAM_c\s*=\s*_PARAM_c")
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_nested_and_or(basic_df, spark_info, many_params_fixture, memory_leak_check):
    """tests that the params are properly passed as arguments when converting ands/or's inside applies"""
    query = "select CASE WHEN @a THEN (CASE WHEN A >= 0 AND @c > 0 THEN A ELSE 0 END) ELSE 0 END from table1"

    pandas_code = check_query(
        query,
        basic_df,
        spark_info,
        named_params=many_params_fixture,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # check pandas code has appropriate applies
    regexp = re.compile(
        r".*bodosql_case_placeholder.*_PARAM_c\s*=\s*_PARAM_c, _PARAM_a\s*=\s*_PARAM_a"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_nested_case(basic_df, spark_info, many_params_fixture, memory_leak_check):
    """tests that the params are properly passed as arguments when converting ands/or's inside applies"""
    query = "select CASE WHEN @a THEN (CASE WHEN A >= 0 AND @c > 0 THEN LPAD(@d, A, @e) ELSE 'case2' END) WHEN @b THEN (CASE WHEN @d = 'hello' then 'case2' WHEN @e = 'hello2' THEN 'case3' END) ELSE 'case4' END from table1"

    pandas_code = check_query(
        query,
        basic_df,
        spark_info,
        named_params=many_params_fixture,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # check pandas code has appropriate applies
    regexp = re.compile(
        r".*bodosql_case_placeholder.*_PARAM_e\s*=\s*_PARAM_e, _PARAM_d\s*=\s*_PARAM_d, _PARAM_c\s*=\s*_PARAM_c, _PARAM_b\s*=\s*_PARAM_b, _PARAM_a\s*=\s*_PARAM_a"
    )
    assert bool(regexp.search(pandas_code))
