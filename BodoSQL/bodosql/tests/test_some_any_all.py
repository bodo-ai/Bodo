# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL SOME and ALL clauses.
"""
import pytest
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param("SOME", marks=pytest.mark.skip),
        pytest.param("ANY", marks=pytest.mark.skip),
        pytest.param("ALL", marks=pytest.mark.skip),
    ]
)
def some_any_all(request):
    """Returns a string that is either SOME ANY or ALL"""
    return request.param


def test_some_any_all_numeric_non_null_tuples(
    bodosql_numeric_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that SOME and ALL work with numeric value tuples"""

    query = f"SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (1, 2)"

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = (
            f"SELECT A FROM table1 WHERE A {comparison_ops} 1 OR A {comparison_ops} 2 "
        )
    else:
        spark_query = (
            f"SELECT A FROM table1 WHERE A {comparison_ops} 1 AND A {comparison_ops} 2 "
        )

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_some_any_all_string_non_null_tuples(
    bodosql_string_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that some, any, and all work on non null String tuples"""
    query = f"""SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} ('A', 'B', 'C')"""

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 'A' OR A {comparison_ops} 'B' OR A {comparison_ops} 'C' "
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 'A' AND A {comparison_ops} 'B' AND A {comparison_ops} 'C' "

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_some_any_all_datetime_non_null_tuples(
    bodosql_datetime_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that some, any, and all work on Timestamp tuples"""
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            A {comparison_ops} {some_any_all} (TIMESTAMP '2011-01-01', TIMESTAMP '1998-08-10', TIMESTAMP '2008-12-13')
        """

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} TIMESTAMP '2011-01-01' OR A {comparison_ops} TIMESTAMP '1998-08-10' OR A {comparison_ops} TIMESTAMP '2008-12-13'"
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} TIMESTAMP '2011-01-01' AND A {comparison_ops} TIMESTAMP '1998-08-10' AND A {comparison_ops} TIMESTAMP '2008-12-13'"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_some_any_all_interval_non_null_tuples(
    bodosql_interval_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that the basic comparison operators work with Timedelta data within the same table"""
    query = f""" SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (INTERVAL 1 HOUR, INTERVAL 2 SECOND, INTERVAL -3 DAY)"""

    # Spark casts interval types to bigint, doing comparisons to bigint equivalents of the above intervals to avoid typing errors.
    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 3600000000000 OR A {comparison_ops} 2000000000 OR A {comparison_ops} -259200000000000"
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 3600000000000 AND A {comparison_ops} 2000000000 AND A {comparison_ops} -259200000000000"

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["A"],
        equivalent_spark_query=spark_query,
    )


def test_some_any_all_like_not_like_non_null_tuples(
    bodosql_string_types, like_expression, some_any_all, spark_info, memory_leak_check
):
    """Tests that the basic comparison operators work with Timedelta data within the same table"""
    query = f""" SELECT A FROM table1 WHERE A {like_expression} {some_any_all} ('h%o', '%', '%el%')"""

    # Spark casts interval types to bigint, doing comparisons to bigint equivalents of the above inttervals to avoid typing errors.
    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {like_expression} 'h%o' OR A {like_expression} '%' OR A {like_expression} '%el%'"
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {like_expression} 'h%o' AND A {like_expression} '%' AND A {like_expression} '%el%'"

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_some_any_all_null_tuples(
    bodosql_nullable_numeric_types,
    some_any_all,
    comparison_ops,
    spark_info,
    memory_leak_check,
):
    """Tests that SOME and ALL work with numeric value tuples"""

    query = f"SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (1, NULL)"

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 1 OR A {comparison_ops} NULL "
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 1 AND A {comparison_ops} NULL "

    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )
