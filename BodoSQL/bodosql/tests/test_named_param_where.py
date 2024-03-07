"""
Test that Named Parameters can be used for where expressions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pytest

from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


def test_int_compare(
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


def test_float_compare(
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
def test_string_compare(
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


def test_datetime_compare(
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


def test_tzaware_timestamp_compare(
    bodosql_datetime_types, spark_info, tzaware_timestamp_named_params
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


@pytest.mark.slow
def test_interval_compare(
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
