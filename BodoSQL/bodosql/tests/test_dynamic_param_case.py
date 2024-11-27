"""
Test that Dynamic Parameters are appropriately handled in case statements.
"""

import re

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
def test_named_param_case(basic_df, spark_info, many_params_fixture, memory_leak_check):
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
        r".*bodosql_case_placeholder.*_NAMED_PARAM_g\s*=\s*_NAMED_PARAM_g, _NAMED_PARAM_e\s*=\s*_NAMED_PARAM_e, _NAMED_PARAM_f\s*=\s*_NAMED_PARAM_f, _NAMED_PARAM_c\s*=\s*_NAMED_PARAM_c, _NAMED_PARAM_d\s*=\s*_NAMED_PARAM_d, _NAMED_PARAM_a\s*=\s*_NAMED_PARAM_a, _NAMED_PARAM_b\s*=\s*_NAMED_PARAM_b.*"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_dynamic_param_case(basic_df, many_params_fixture, memory_leak_check):
    """tests that the params are properly passed as arguments when generate an apply due to a case stmt"""
    query = "select CASE WHEN ? THEN TRUE WHEN ? THEN FALSE WHEN ? > 12 THEN FALSE WHEN ? ='hello' THEN FALSE WHEN ? = 'hello' THEN FALSE WHEN ? > TIMESTAMP '2021-10-14' THEN FALSE WHEN ? > TIMESTAMP '2021-10-14' THEN FALSE ELSE A > 1 END as OUTPUT from table1"
    bind_variables = (
        many_params_fixture["a"],
        many_params_fixture["b"],
        many_params_fixture["c"],
        many_params_fixture["d"],
        many_params_fixture["e"],
        many_params_fixture["f"],
        many_params_fixture["g"],
    )
    expected_output = pd.DataFrame(
        {
            "OUTPUT": [True] * len(basic_df["TABLE1"]),
        }
    )

    pandas_code = check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        return_codegen=True,
        expected_output=expected_output,
    )["pandas_code"]

    # Check pandas code has appropriate applies
    regexp = re.compile(
        r".*bodosql_case_placeholder.*_DYNAMIC_PARAM_0\s*=\s*_DYNAMIC_PARAM_0, _DYNAMIC_PARAM_6\s*=\s*_DYNAMIC_PARAM_6, _DYNAMIC_PARAM_5\s*=\s*_DYNAMIC_PARAM_5, _DYNAMIC_PARAM_4\s*=\s*_DYNAMIC_PARAM_4, _DYNAMIC_PARAM_3\s*=\s*_DYNAMIC_PARAM_3, _DYNAMIC_PARAM_2\s*=\s*_DYNAMIC_PARAM_2, _DYNAMIC_PARAM_1\s*=\s*_DYNAMIC_PARAM_1.*"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_named_param_repeated_param_usage(
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
    regexp = re.compile(
        r".*bodosql_case_placeholder.*_NAMED_PARAM_c\s*=\s*_NAMED_PARAM_c"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_named_param_nested_and_or(
    basic_df, spark_info, many_params_fixture, memory_leak_check
):
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
        r".*bodosql_case_placeholder.*_NAMED_PARAM_c\s*=\s*_NAMED_PARAM_c, _NAMED_PARAM_a\s*=\s*_NAMED_PARAM_a"
    )
    assert bool(regexp.search(pandas_code))


@pytest.mark.slow
def test_named_param_nested_case(
    basic_df, spark_info, many_params_fixture, memory_leak_check
):
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
        r".*bodosql_case_placeholder.*_NAMED_PARAM_e\s*=\s*_NAMED_PARAM_e, _NAMED_PARAM_c\s*=\s*_NAMED_PARAM_c, _NAMED_PARAM_d\s*=\s*_NAMED_PARAM_d, _NAMED_PARAM_a\s*=\s*_NAMED_PARAM_a, _NAMED_PARAM_b\s*=\s*_NAMED_PARAM_b"
    )
    assert bool(regexp.search(pandas_code))
