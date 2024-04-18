# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test that Dynamic Parameters can be used for where expressions.
"""

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


def generate_comparison_filter(comparison_op, A, B):
    if comparison_op == "=":
        return A == B
    elif comparison_op == "<>" or comparison_op == "!=":
        return A != B
    elif comparison_op == "<":
        return A < B
    elif comparison_op == "<=":
        return A <= B
    elif comparison_op == ">":
        return A > B
    elif comparison_op == ">=":
        return A >= B
    elif comparison_op == "<=>":
        # Note: We should check for null but we know here at least
        # A is never null.
        return A == B
    else:
        raise ValueError(f"Unknown comparison operator: {comparison_op}")


@pytest.mark.slow
def test_named_param_int_compare(
    bodosql_nullable_numeric_types,
    spark_info,
    comparison_ops,
    int_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with integer data and integer named parameters
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @a {comparison_ops} C
        """
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        named_params=int_named_params,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variable_int_compare(
    bodosql_nullable_numeric_types,
    comparison_ops,
    int_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with integer data and integer bind variables
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            ? {comparison_ops} C
        """
    bind_variables = (int_named_params["a"], int_named_params["b"])
    table = bodosql_nullable_numeric_types["TABLE1"]
    filter = generate_comparison_filter(comparison_ops, bind_variables[0], table["C"])
    expected_output = pd.DataFrame({"A": table["A"][filter]})
    check_query(
        query,
        bodosql_nullable_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_float_compare(
    bodosql_numeric_types,
    spark_info,
    comparison_ops,
    float_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with numeric data and float named parameters
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @a {comparison_ops} C
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        named_params=float_named_params,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variable_float_compare(
    bodosql_numeric_types,
    comparison_ops,
    float_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with numeric data and float bind variables
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            ? {comparison_ops} C
        """
    bind_variables = (float_named_params["a"], float_named_params["b"])
    table = bodosql_numeric_types["TABLE1"]
    filter = generate_comparison_filter(comparison_ops, bind_variables[0], table["C"])
    expected_output = pd.DataFrame({"A": table["A"][filter]})
    check_query(
        query,
        bodosql_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_named_param_string_compare(
    bodosql_string_types,
    spark_info,
    comparison_ops,
    string_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with string data and string named parameters
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @a {comparison_ops} C
        """
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        named_params=string_named_params,
        check_dtype=False,
    )


def test_bind_variable_string_compare(
    bodosql_string_types,
    comparison_ops,
    string_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with string data and string bind variables
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            ? {comparison_ops} C
        """
    bind_variables = (string_named_params["a"], string_named_params["b"])
    table = bodosql_string_types["TABLE1"]
    filter = generate_comparison_filter(comparison_ops, bind_variables[0], table["C"])
    expected_output = pd.DataFrame({"A": table["A"][filter]})
    check_query(
        query,
        bodosql_string_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_datetime_compare(
    bodosql_datetime_types,
    spark_info,
    comparison_ops,
    timestamp_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with datetime data and timestamp named parameters
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @a {comparison_ops} C
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=timestamp_named_params,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variable_datetime_compare(
    bodosql_datetime_types,
    comparison_ops,
    timestamp_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with datetime data and timestamp bind variables
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            ? {comparison_ops} C
        """
    bind_variables = (timestamp_named_params["a"], timestamp_named_params["b"])
    table = bodosql_datetime_types["TABLE1"]
    filter = generate_comparison_filter(comparison_ops, bind_variables[0], table["C"])
    expected_output = pd.DataFrame({"A": table["A"][filter]})
    check_query(
        query,
        bodosql_datetime_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_named_param_tzaware_timestamp_compare(
    bodosql_datetime_types,
    spark_info,
    tzaware_timestamp_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with datetime data and tz aware timestamp named parameters.
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @b = C
        """
    params, tz = tzaware_timestamp_named_params
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=params,
        check_dtype=False,
        session_tz=tz,
    )


def test_bind_variable_tz_aware_compare(
    bodosql_datetime_types,
    tzaware_timestamp_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with datetime data and tz aware timestamp bind variables.
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            ? = C
        """
    params, tz = tzaware_timestamp_named_params
    bind_variables = (params["a"], params["b"])
    table = bodosql_datetime_types["TABLE1"]
    filter = generate_comparison_filter("=", bind_variables[0], table["C"])
    expected_output = pd.DataFrame({"A": table["A"][filter]})
    check_query(
        query,
        bodosql_datetime_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
        session_tz=tz,
    )


@pytest.mark.slow
def test_named_param_interval_compare(
    bodosql_interval_types,
    comparison_ops,
    timedelta_named_params,
    memory_leak_check,
):
    """
    Tests that comparison operators work with interval data and Timedelta named parameters
    """
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            @a {comparison_ops} C
        """
    # NOTE: this assumes that the input data doesn't require comparing two nulls in <=>
    pd_op = (
        "=="
        if comparison_ops in ("=", "<=>")
        else "!="
        if comparison_ops == "<>"
        else comparison_ops
    )
    a = timedelta_named_params["a"]
    check_query(
        query,
        bodosql_interval_types,
        None,
        named_params=timedelta_named_params,
        check_dtype=False,
        expected_output=bodosql_interval_types["TABLE1"].query(f"@a {pd_op} C")[["A"]],
    )
