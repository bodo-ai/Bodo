"""
Test that Named Parameters can be used in various functions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pytest
from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_int_func(bodosql_nullable_numeric_types, spark_info, int_named_params):
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
def test_float_func(bodosql_numeric_types, spark_info, float_named_params):
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


def test_string_func(bodosql_string_types, spark_info, string_named_params):
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


@pytest.mark.slow
def test_datetime_func(bodosql_datetime_types, spark_info, timestamp_named_params):
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
def test_interval_func(bodosql_datetime_types, spark_info, timedelta_named_params):
    """
    Checks that named params can be used in a timestamp function
    """
    query = "select date_add(A, @a) from table1"
    # Spark cannot use interval scalars in any function both BodoSQL and SparkSQL
    # support.
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
