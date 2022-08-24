"""
Test that Named Parameters can be used in having expressions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pytest
from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import bodo_version_older, check_query


@pytest.mark.slow
def test_int_having(bodosql_nullable_numeric_types, spark_info, int_named_params):
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
def test_float_having(bodosql_numeric_types, spark_info, float_named_params):
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
def test_string_having(bodosql_string_types, spark_info, string_named_params):
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


@pytest.mark.skipif(
    bodo_version_older(2021, 7, 3),
    reason="[BE-1174] Requires next mini-release for Timestamp bug fix",
)
def test_datetime_having(bodosql_datetime_types, spark_info, timestamp_named_params):
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


def test_interval_having(bodosql_interval_types, spark_info, timedelta_named_params):
    """
    Tests that having works with interval data and Timedelta named parameters
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
        bodosql_interval_types,
        spark_info,
        named_params=timedelta_named_params,
        check_dtype=False,
        check_names=False,
    )
