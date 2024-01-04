"""
Test that Named Parameters can be used in having expressions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_int_arith(
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


@pytest.mark.slow
def test_float_arith(
    bodosql_numeric_types, spark_info, arith_ops, float_named_params, memory_leak_check
):
    """
    Tests that arithmetic operators work with numeric data and float named parameters
    """
    query = f"""
        SELECT @a {arith_ops} A as COL from table1
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
def test_string_arith(
    bodosql_string_types, spark_info, string_named_params, memory_leak_check
):
    """
    Tests that arithmetic operators work with string data and string named parameters
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
def test_datetime_timedelta_arith(
    bodosql_datetime_types,
    spark_info,
    datetime_arith_ops,
    timedelta_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with datetime data and timedelta named parameters
    """
    query = f"""
        SELECT A {datetime_arith_ops} @a from table1
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=timedelta_named_params,
        check_dtype=False,
        check_names=False,
        named_params_timedelta_interval=True,
    )


def test_datetime_dateoffset_arith(
    bodosql_datetime_types,
    spark_info,
    datetime_arith_ops,
    dateoffset_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with datetime data and dateoffset named parameters
    """
    query = f"""
        SELECT A {datetime_arith_ops} @a from table1
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        named_params=dateoffset_named_params,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_interval_arith(
    bodosql_interval_types,
    datetime_arith_ops,
    timedelta_named_params,
    memory_leak_check,
):
    """
    Tests that arithmetic operators work with interval data and Timedelta named parameters
    """
    query = f"""
        SELECT @a {datetime_arith_ops} A as colname from table1
        """
    A = bodosql_interval_types["TABLE1"]["A"]
    a = timedelta_named_params["a"]
    expected_output = (a + A) if datetime_arith_ops == "+" else (a - A)
    expected_output = expected_output.to_frame()
    check_query(
        query,
        bodosql_interval_types,
        None,
        named_params=timedelta_named_params,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )
