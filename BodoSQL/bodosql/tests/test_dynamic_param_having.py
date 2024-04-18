# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test that Dynamic Parameters can be used in having expressions.
"""

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_named_param_int_having(
    bodosql_nullable_numeric_types, spark_info, int_named_params, memory_leak_check
):
    """
    Tests that having works with integer data and integer named parameters
    """
    query = f"""
        SELECT
            COUNT(A)
        FROM
            table1
        GROUP BY
            B
        HAVING
            @a > B
        """
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        named_params=int_named_params,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_bind_variables_int_having(
    bodosql_nullable_numeric_types, int_named_params, memory_leak_check
):
    """
    Tests that having works with integer data and integer bind variables
    """
    query = f"""
        SELECT
            COUNT(A) as OUTPUT
        FROM
            table1
        GROUP BY
            B
        HAVING
            ? > B
        """
    bind_variables = (int_named_params["a"], int_named_params["b"])
    expected_output = (
        bodosql_nullable_numeric_types["TABLE1"].groupby("B", as_index=False).count()
    )
    expected_output = expected_output[expected_output.B < bind_variables[0]]
    expected_output = pd.DataFrame({"OUTPUT": expected_output["A"]})
    check_query(
        query,
        bodosql_nullable_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_float_having(
    bodosql_numeric_types, spark_info, float_named_params, memory_leak_check
):
    """
    Tests that having works with numeric data and float named parameters
    """
    query = f"""
        SELECT
            COUNT(A)
        FROM
            table1
        GROUP BY
            B
        HAVING
            @a > B
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        named_params=float_named_params,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_bind_variables_float_having(
    bodosql_numeric_types, float_named_params, memory_leak_check
):
    """
    Tests that having works with numeric data and float bind variables
    """
    query = f"""
        SELECT
            COUNT(A) as OUTPUT
        FROM
            table1
        GROUP BY
            B
        HAVING
            ? > B
        """
    bind_variables = (float_named_params["a"], float_named_params["b"])
    expected_output = (
        bodosql_numeric_types["TABLE1"].groupby("B", as_index=False).count()
    )
    expected_output = expected_output[expected_output.B < bind_variables[0]]
    expected_output = pd.DataFrame({"OUTPUT": expected_output["A"]})
    check_query(
        query,
        bodosql_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_string_having(
    bodosql_string_types, spark_info, string_named_params, memory_leak_check
):
    """
    Tests that having works with string data and string named parameters
    """
    query = f"""
        SELECT
            COUNT(A)
        FROM
            table1
        GROUP BY
            B
        HAVING
            @a > B
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
def test_bind_variables_string_having(
    bodosql_string_types, string_named_params, memory_leak_check
):
    """
    Tests that having works with string data and string bind variables
    """
    query = f"""
        SELECT
            COUNT(A) as OUTPUT
        FROM
            table1
        GROUP BY
            B
        HAVING
            ? > B
        """
    bind_variables = (string_named_params["a"], string_named_params["b"])
    expected_output = (
        bodosql_string_types["TABLE1"].groupby("B", as_index=False).count()
    )
    expected_output = expected_output[expected_output.B < bind_variables[0]]
    expected_output = pd.DataFrame({"OUTPUT": expected_output["A"]})
    check_query(
        query,
        bodosql_string_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_named_param_datetime_having(
    bodosql_datetime_types, spark_info, timestamp_named_params, memory_leak_check
):
    """
    Tests that having works with datetime data and timestamp named parameters
    """
    query = f"""
        SELECT
            COUNT(A)
        FROM
            table1
        GROUP BY
            B
        HAVING
            @a > B
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=timestamp_named_params,
        check_dtype=False,
        check_names=False,
    )


def test_bind_variables_datetime_having(
    bodosql_datetime_types, timestamp_named_params, memory_leak_check
):
    """
    Tests that having works with Timestamp data and Timestamp bind variables
    """
    query = f"""
        SELECT
            COUNT(A) as OUTPUT
        FROM
            table1
        GROUP BY
            B
        HAVING
            ? > B
        """
    bind_variables = (timestamp_named_params["a"], timestamp_named_params["b"])
    expected_output = (
        bodosql_datetime_types["TABLE1"].groupby("B", as_index=False).count()
    )
    expected_output = expected_output[expected_output.B < bind_variables[0]]
    expected_output = pd.DataFrame({"OUTPUT": expected_output["A"]})
    check_query(
        query,
        bodosql_datetime_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )
