# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test that Dynamic Parameters generate runtime casts when
the actual type doesn't match the inferred type.
"""

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.utils.typing import BodoError
from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_named_param_string_limit(basic_df, memory_leak_check):
    """
    Check that limit enforces the integer requirement.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 limit @a", {"a": a})

    with pytest.raises(
        BodoError, match=r"Failure in compiling or validating SQL Query"
    ):
        impl(basic_df["TABLE1"], "1")


@pytest.mark.slow
def test_bind_variable_string_limit(basic_df, memory_leak_check):
    """
    Test that bind variables with a string limit casts the
    string to an integer.
    """
    query = "Select A from table1 limit ?"
    expected_output = pd.DataFrame({"A": basic_df["TABLE1"]["A"].head(1)})
    check_query(
        query,
        basic_df,
        None,
        bind_variables=("1",),
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_string_offset(basic_df, memory_leak_check):
    """
    Check that offset enforces the integer requirement.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 limit @a, 4", {"a": a})

    with pytest.raises(
        BodoError, match=r"Failure in compiling or validating SQL Query"
    ):
        impl(basic_df["TABLE1"], "1")


@pytest.mark.slow
def test_bind_variable_string_offset(basic_df, memory_leak_check):
    """
    Test that bind variables with a string offset casts the
    string to an integer.
    """
    query = "Select A from table1 limit ?, 4"
    expected_output = pd.DataFrame({"A": basic_df["TABLE1"]["A"].iloc[1:5]})
    check_query(
        query,
        basic_df,
        None,
        bind_variables=("1",),
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_string_function_cast(
    bodosql_string_types, spark_info, int_named_params, memory_leak_check
):
    """
    Check that limit a string function will cast a non-string
    named Parameter.
    """
    query = "select CONCAT(A, @a) as OUTPUT from table1"
    expected_output = pd.DataFrame(
        {"OUTPUT": bodosql_string_types["TABLE1"]["A"] + str(int_named_params["a"])}
    )
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        named_params=int_named_params,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_bind_variable_string_function_cast(
    bodosql_string_types, int_named_params, memory_leak_check
):
    """
    Test that a string function will cast a non-string
    named Parameter.
    """
    query = "select CONCAT(A, ?) as OUTPUT from table1"
    bind_variables = (int_named_params["a"], int_named_params["b"])
    expected_output = pd.DataFrame(
        {"OUTPUT": bodosql_string_types["TABLE1"]["A"] + str(bind_variables[0])}
    )
    check_query(
        query,
        bodosql_string_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )
