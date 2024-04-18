# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test that Dynamic Parameters can be used in arithmetic operations expressions.
"""

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_int_arith_named(
    bodosql_nullable_numeric_types,
    spark_info,
    arith_ops,
    int_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with integer data and integer named parameters
    """
    query = f"""
        SELECT @a {arith_ops} A from table1
        """

    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        named_params=int_named_params,
        check_dtype=False,
        check_names=False,
    )


def compute_output_column(arith_op, a, b):
    if arith_op == "+":
        return a + b
    elif arith_op == "-":
        return a - b
    elif arith_op == "*":
        return a * b
    elif arith_op == "/":
        return a / b
    else:
        assert False, "Unsupported arithmetic operator"


@pytest.mark.slow
def test_int_arith_bind(
    bodosql_nullable_numeric_types,
    arith_ops,
    int_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with integer data and integer bind variables
    """
    query = f"SELECT ? {arith_ops} A as OUTPUT from table1"
    bind_variables = (int_named_params["a"], int_named_params["b"])
    expected_output = pd.DataFrame(
        {
            "OUTPUT": compute_output_column(
                arith_ops,
                np.int64(bind_variables[0]),
                bodosql_nullable_numeric_types["TABLE1"]["A"].astype("Int64"),
            )
        }
    )
    check_query(
        query,
        bodosql_nullable_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_float_arith_named(
    bodosql_numeric_types, spark_info, arith_ops, float_named_params, memory_leak_check
):
    """
    Tests that arithmetic operators work with numeric data and float named parameters
    """
    query = f"""
        SELECT cast(@a as DOUBLE) {arith_ops} A as COL from table1
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        named_params=float_named_params,
        check_dtype=False,
        convert_columns_decimal=["COL"],
    )


@pytest.mark.slow
def test_float_arith_bind(
    bodosql_numeric_types,
    arith_ops,
    float_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with numeric data and float bind variables
    """
    query = f"SELECT ?::DOUBLE {arith_ops} A as OUTPUT from table1"
    bind_variables = (float_named_params["a"], float_named_params["b"])
    expected_output = pd.DataFrame(
        {
            "OUTPUT": compute_output_column(
                arith_ops,
                np.float64(bind_variables[0]),
                bodosql_numeric_types["TABLE1"]["A"].astype("Float64"),
            )
        }
    )
    check_query(
        query,
        bodosql_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_string_binary_named(
    bodosql_string_types, spark_info, string_named_params, memory_leak_check
):
    """
    Tests that binary operators work with string data and string named parameters
    """
    query = f"""
        SELECT @a || A from table1
        """
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        named_params=string_named_params,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_string_binary_bind(
    bodosql_string_types,
    string_named_params,
    memory_leak_check,
):
    """
    Tests that binary operators work with string data and string bind variables
    """
    query = f"SELECT ? || A as OUTPUT from table1"
    bind_variables = (string_named_params["a"], string_named_params["b"])
    expected_output = pd.DataFrame(
        {
            "OUTPUT": compute_output_column(
                "+", bind_variables[0], bodosql_string_types["TABLE1"]["A"]
            )
        }
    )
    check_query(
        query,
        bodosql_string_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )
