# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL set like operations. Namley, Union, Intersect, and Exclude
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.timezone_common import representative_tz  # noqa


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


def test_union_tz_aware_cols(representative_tz, memory_leak_check):
    """tests that union works for tz_aware columns"""
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="4D7H", periods=30, tz=representative_tz
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022", freq="12D21H", periods=20, tz=representative_tz
                )
            ),
        }
    )
    py_output = pd.DataFrame({"A": pd.concat((df["A"], df["B"]))}).drop_duplicates()
    ctx = {"table1": df}
    query = "(Select A from table1) union (Select B from table1)"
    check_query(query, ctx, None, expected_output=py_output)


def test_union_all_cols(basic_df, spark_info, memory_leak_check):
    """tests that union all works for columns"""
    query = "(Select A from table1) union ALL (Select A from table1)"
    check_query(query, basic_df, spark_info)


@pytest.mark.slow
def test_union_all_null_cols(null_set_df, spark_info, memory_leak_check):
    """tests that union all works for columns"""
    query = "(Select A from table1) union ALL (Select B from table1)"
    check_query(query, null_set_df, spark_info, convert_float_nan=True)


def test_union_all_tz_aware_cols(representative_tz, memory_leak_check):
    """tests that union all works for tz_aware columns"""
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="4D7H", periods=30, tz=representative_tz
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022", freq="12D21H", periods=20, tz=representative_tz
                )
            ),
        }
    )
    py_output = pd.DataFrame({"A": pd.concat((df["A"], df["B"]))})
    ctx = {"table1": df}
    query = "(Select A from table1) union all (Select B from table1)"
    check_query(query, ctx, None, expected_output=py_output)


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


def test_intersect_tz_aware_cols(representative_tz, memory_leak_check):
    """tests that intersect works for tz_aware columns"""
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="4D7H", periods=30, tz=representative_tz
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022", freq="12D21H", periods=20, tz=representative_tz
                )
            ),
        }
    )
    py_output = df[["A"]].merge(df[["B"]].rename(columns={"B": "A"}), on="A")
    # If there is 1 NA in the output it should exist.
    # TODO: Double check intersect null behavior with Snowflake.
    na_entries = py_output["A"].isna()
    # If there is 1 NA in the output it should exist.
    append_na = py_output["A"].isna().any()
    py_output = py_output[~na_entries]
    if append_na:
        py_output = pd.DataFrame({"A": list(py_output["A"]) + [None]})
    ctx = {"table1": df}
    query = "(Select A from table1) intersect (Select B from table1)"
    check_query(query, ctx, None, expected_output=py_output)


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


@pytest.mark.skip("[BS-379] Except not supported")
def test_except_tz_aware_cols(representative_tz, memory_leak_check):
    """tests that except works for string columns"""
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="4D7H", periods=30, tz=representative_tz
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022", freq="12D21H", periods=20, tz=representative_tz
                )
            ),
        }
    )
    # TODO: Fix the expected output.
    py_output = pd.DataFrame({"A": pd.concat((df["A"], df["B"]))})
    ctx = {"table1": df}
    query = "(Select A from table1) union all (Select B from table1)"
    check_query(query, ctx, None, expected_output=py_output)


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
