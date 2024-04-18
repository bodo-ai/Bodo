# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test that Named Parameters can be used in various functions.
"""

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_named_param_int_func(
    bodosql_nullable_numeric_types, spark_info, int_named_params, memory_leak_check
):
    """
    Checks that named params can be used in an integer function
    """
    query = "select A + sin(@a) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        named_params=int_named_params,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variables_int_func(
    bodosql_nullable_numeric_types, int_named_params, memory_leak_check
):
    """
    Checks that bind variables can be used in an integer function
    """
    query = "select A + sin(?) as OUTPUT from table1"
    bind_variables = (int_named_params["a"], int_named_params["b"])
    expected_output = pd.DataFrame(
        {
            "OUTPUT": bodosql_nullable_numeric_types["TABLE1"]["A"]
            + np.sin(np.int64(bind_variables[0]))
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
def test_named_param_float_func(
    bodosql_numeric_types, spark_info, float_named_params, memory_leak_check
):
    """
    Checks that named params can be used in a float function
    """
    query = "select A + sqrt(@a) from table1"
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        named_params=float_named_params,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variables_float_func(
    bodosql_numeric_types, float_named_params, memory_leak_check
):
    """
    Checks that bind variables can be used in a float function
    """
    query = "select A + sqrt(?) as OUTPUT from table1"
    bind_variables = (float_named_params["a"], float_named_params["b"])
    expected_output = pd.DataFrame(
        {"OUTPUT": bodosql_numeric_types["TABLE1"]["A"] + np.sqrt(bind_variables[0])}
    )
    check_query(
        query,
        bodosql_numeric_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_named_param_string_func(
    bodosql_string_types, spark_info, string_named_params, memory_leak_check
):
    """
    Checks that named params can be used in a string function
    """
    query = "select A || UPPER(@a) from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        named_params=string_named_params,
        check_names=False,
    )


def test_bind_variables_string_func(
    bodosql_string_types, string_named_params, memory_leak_check
):
    """
    Checks that bind variables can be used in a string function
    """
    query = "select A || UPPER(?) as OUTPUT from table1"
    bind_variables = (string_named_params["a"], string_named_params["b"])
    expected_output = pd.DataFrame(
        {"OUTPUT": bodosql_string_types["TABLE1"]["A"] + bind_variables[0].upper()}
    )
    check_query(
        query,
        bodosql_string_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_datetime_func(
    bodosql_datetime_types, spark_info, timestamp_named_params, memory_leak_check
):
    """
    Checks that named params can be used in a timestamp function
    """
    query = "select DATEDIFF(A, @a) from table1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=timestamp_named_params,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_bind_variables_datetime_func(
    bodosql_datetime_types, timestamp_named_params, memory_leak_check
):
    """
    Checks that bind variables can be used in a string function
    """
    query = "select DATEDIFF(A, ?) as OUTPUT from table1"
    bind_variables = (timestamp_named_params["a"], timestamp_named_params["b"])
    expected_output = pd.DataFrame(
        {
            "OUTPUT": (bodosql_datetime_types["TABLE1"]["A"] - bind_variables[0])
            .dt.round("d")
            .dt.days
        }
    )
    check_query(
        query,
        bodosql_datetime_types,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_named_param_interval_func(
    bodosql_datetime_types, spark_info, timedelta_named_params, memory_leak_check
):
    """
    Checks that named params can be used in a timestamp function
    """
    query = "select date_add(A, @a) from table1"
    spark_query = "select A + @a from table1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=timedelta_named_params,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
        named_params_timedelta_interval=True,
    )
