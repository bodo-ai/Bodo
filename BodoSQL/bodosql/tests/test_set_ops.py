# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL set like operations. Namley, Union, Intersect, and Exclude
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def null_set_df():
    int_data = {
        "A": [1, 2, 3, None] * 3,
        "B": [4, None, 2, 3] * 3,
    }
    return {"table1": pd.DataFrame(data=int_data, dtype="Int64")}


def test_union_cols(basic_df, spark_info, memory_leak_check):
    """tests that union works for columns"""
    query = "(Select A from table1) union (Select B from table1)"
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_union_null_cols(null_set_df, spark_info, memory_leak_check):
    """tests that union works for columns"""
    query = "(Select A from table1) union (Select B from table1)"
    check_query(query, null_set_df, spark_info, convert_float_nan=True)


def test_union_string_cols(bodosql_string_types, spark_info, memory_leak_check):
    """tests that union works for string columns"""
    query = "(Select A from table1) union (Select B from table1)"
    check_query(query, bodosql_string_types, spark_info, convert_float_nan=True)


def test_union_all_cols(basic_df, spark_info, memory_leak_check):
    """tests that union all works for columns"""
    query = "(Select A from table1) union ALL (Select A from table1)"
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_union_all_null_cols(null_set_df, spark_info, memory_leak_check):
    """tests that union all works for columns"""
    query = "(Select A from table1) union ALL (Select B from table1)"
    check_query(query, null_set_df, spark_info, convert_float_nan=True)


def test_intersect_cols(basic_df, spark_info, memory_leak_check):
    """tests that intersect works for columns"""
    query = "(Select A from table1) intersect (Select B from table1)"
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_intersect_null_cols(null_set_df, spark_info, memory_leak_check):
    """tests that intersect works for columns"""
    query = "(Select A from table1) intersect (Select B from table1)"
    check_query(query, null_set_df, spark_info, convert_float_nan=True)


def test_intersect_string_cols(bodosql_string_types, spark_info, memory_leak_check):
    """tests that union works for columns"""
    query = "(Select A from table1) intersect (Select B from table1)"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        convert_float_nan=True,
    )


@pytest.mark.skip("[BS-379] Except not supported")
def test_except_cols(basic_df, spark_info, memory_leak_check):
    """tests that except works for columns"""
    query = "(Select A, B from table1) except (Select A, C from table1)"
    check_query(query, basic_df, spark_info, only_python=True)


@pytest.mark.skip("[BS-379] Except not supported")
def test_except_string_cols(bodosql_string_types, spark_info, memory_leak_check):
    """tests that except works for string columns"""
    query = "(Select A, B from table1) except (Select A, C from table1)"
    check_query(query, bodosql_string_types, spark_info, only_python=True)


@pytest.mark.slow
@pytest.mark.skip("[BS-379] Except not supported")
def test_except_scalars(spark_info, memory_leak_check):
    """tests that except works for Scalars"""
    query = "(SELECT 1, 2) except (SELECT 2, 3)"
    # the above query is not valid for spark
    expected = pd.DataFrame(
        {
            "unkown1": [1],
            "unkown2": [2],
        }
    )

    query2 = "(SELECT 1, 2) except (SELECT 1, 2)"
    # the above query is not valid for spark
    expected2 = pd.DataFrame(
        {
            "unkown1": [],
            "unkown2": [],
        }
    )

    check_query(
        query,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected,
        check_dtype=False,
        only_python=True,
    )
    check_query(
        query2,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected2,
        check_dtype=False,
        only_python=True,
    )


# the following literal tests are done using only python to avoid [BS-381]


@pytest.mark.slow
def test_union_scalars(spark_info, memory_leak_check):
    """tests that union works for Scalars"""
    query1 = "SELECT 1,2 UNION SELECT 1,2"
    # the above query is not valid for spark
    expected1 = pd.DataFrame(
        {
            "unkown": [1],
            "unkown2": [2],
        }
    )
    query2 = "SELECT 1,2 UNION SELECT 1,3"
    # the above query is not valid for spark
    expected2 = pd.DataFrame(
        {
            "unkown": [1, 1],
            "unkown2": [2, 3],
        }
    )

    check_query(
        query1,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected1,
        check_dtype=False,
    )
    check_query(
        query2,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected2,
        check_dtype=False,
    )


@pytest.mark.slow
def test_union_all_scalars(spark_info, memory_leak_check):
    """tests that union all works for Scalars"""
    query = "SELECT 1,2 UNION ALL SELECT 1,2"
    # the above query is not valid for spark
    expected = pd.DataFrame(
        {
            "unkown": [1, 1],
            "unkown2": [2, 2],
        }
    )
    check_query(
        query,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected,
        check_dtype=False,
    )


@pytest.mark.slow
def test_intersect_scalars(spark_info, memory_leak_check):
    """tests that intersect works for Scalars"""
    query1 = "SELECT 1, 2 intersect SELECT 2, 3"
    query2 = "SELECT 1, 2 intersect SELECT 1, 2"
    # the above query is not valid for spark

    expected1 = pd.DataFrame(
        {
            "unkown1": [],
            "unkown2": [],
        }
    )
    expected2 = pd.DataFrame(
        {
            "unkown1": [1],
            "unkown2": [2],
        }
    )
    check_query(
        query1,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected1,
        check_dtype=False,
    )
    check_query(
        query2,
        dict(),
        spark_info,
        check_names=False,
        expected_output=expected2,
        check_dtype=False,
    )
