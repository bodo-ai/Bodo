"""
Test that Named Parameters can be used for where expressions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pytest
from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import bodo_version_older, check_query


def test_int_compare(
    bodosql_nullable_numeric_types, spark_info, comparison_ops, int_named_params
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
    bodosql_numeric_types, spark_info, comparison_ops, float_named_params
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
    bodosql_string_types, spark_info, comparison_ops, string_named_params
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


@pytest.mark.skipif(
    bodo_version_older(2021, 7, 3),
    reason="[BE-1174] Requires next mini-release for Timestamp bug fix",
)
def test_datetime_compare(
    bodosql_datetime_types, spark_info, comparison_ops, timestamp_named_params
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
def test_interval_compare(
    bodosql_interval_types, spark_info, comparison_ops, timedelta_named_params
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
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        named_params=timedelta_named_params,
        check_dtype=False,
        convert_columns_timedelta=["A"],
    )
