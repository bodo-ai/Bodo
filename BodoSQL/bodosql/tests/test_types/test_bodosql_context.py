# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for user facing BodoSQL APIs.
"""
import numpy as np
import pandas as pd
import pytest
from bodosql import BodoSQLContext, TablePath

from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_add_or_replace_view(memory_leak_check):
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    df3 = pd.DataFrame({"B": [1, 2, 3]})
    bc_orig = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    bc = bc_orig
    pd.testing.assert_frame_equal(bc.sql("select * from t1"), df1)
    pd.testing.assert_frame_equal(bc.sql("select * from t2"), df2)
    bc = bc_orig.add_or_replace_view("t3", df3)
    pd.testing.assert_frame_equal(bc.sql("select * from t1"), df1)
    pd.testing.assert_frame_equal(bc.sql("select * from t2"), df2)
    pd.testing.assert_frame_equal(bc.sql("select * from t3"), df3)
    bc = bc.add_or_replace_view("t1", df3)
    pd.testing.assert_frame_equal(bc.sql("select * from t1"), df3)
    pd.testing.assert_frame_equal(bc.sql("select * from t2"), df2)
    pd.testing.assert_frame_equal(bc.sql("select * from t3"), df3)

    # The original context should be unchanged.
    pd.testing.assert_frame_equal(bc_orig.sql("select * from t1"), df1)
    pd.testing.assert_frame_equal(bc_orig.sql("select * from t2"), df2)
    with pytest.raises(BodoError):
        bc_orig.sql("select * from t3")


def test_remove_view(memory_leak_check):
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"B": [1, 2, 3]})
    bc_orig = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    bc = bc_orig
    pd.testing.assert_frame_equal(bc.sql("select * from t1"), df1)
    pd.testing.assert_frame_equal(bc.sql("select * from t2"), df2)
    bc = bc.remove_view("t2")
    pd.testing.assert_frame_equal(bc.sql("select * from t1"), df1)
    with pytest.raises(BodoError):
        bc.sql("select * from t2")
    with pytest.raises(BodoError, match="'name' must refer to a registered view"):
        bc.remove_view("t2")
    bc = bc.remove_view("t1")
    with pytest.raises(BodoError):
        bc.sql("select * from t1")
    with pytest.raises(BodoError):
        bc.sql("select * from t2")

    # The original context should be unchanged.
    pd.testing.assert_frame_equal(bc_orig.sql("select * from t1"), df1)
    pd.testing.assert_frame_equal(bc_orig.sql("select * from t2"), df2)


def test_bodosql_context_boxing(memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context.
    """

    def impl(bc):
        return bc

    bc = BodoSQLContext(
        {
            "t1": pd.DataFrame(
                {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
            ),
            "t2": pd.DataFrame({"C": [b"345253"] * 100}),
            "t3": TablePath(
                "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
            ),
        }
    )
    check_func(impl, (bc,))


def test_bodosql_context_boxed_sql(memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context.
    """

    def impl(bc):
        return bc.sql("select * from t1")

    py_output = pd.DataFrame(
        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
    )
    bc = BodoSQLContext(
        {
            "t1": py_output,
            "t2": pd.DataFrame({"C": [b"345253"] * 100}),
        }
    )
    check_func(impl, (bc,), py_output=py_output)


def test_bodosql_context_boxed_sql_table_path(memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context where 1
    table uses the table path API.
    """

    def impl(bc):
        return bc.sql("select * from t1")

    py_output = pd.DataFrame(
        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
    )
    bc = BodoSQLContext(
        {
            "t1": py_output,
            "t2": pd.DataFrame({"C": [b"345253"] * 100}),
            "t3": TablePath(
                "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
            ),
        }
    )
    check_func(impl, (bc,), py_output=py_output)


def test_add_or_replace_view_jit(memory_leak_check):
    def impl1(bc, df):
        bc = bc.add_or_replace_view("t1", df)
        return bc.sql("select * from t1")

    def impl2(bc):
        return bc.sql("select * from t1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    df3 = pd.DataFrame({"B": [1, 2, 3]})
    bc = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    # check adding a table
    check_func(impl1, (bc, df3), py_output=df3)
    # Check that the original isn't updated
    check_func(impl2, (bc,), py_output=df1)


def test_remove_view_jit(memory_leak_check):
    def impl1(bc):
        bc = bc.remove_view("t1")
        return bc.sql("select * from t1")

    def impl2(bc):
        return bc.sql("select * from t1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    bc = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    # check removing a table
    with pytest.raises(BodoError):
        impl1(bc)
    # Check that the original isn't updated
    check_func(impl2, (bc,), py_output=df1)


def test_add_or_replace_view_table_path(memory_leak_check):
    """Tests add_or_replace_view distributed analysis code
    works when introducing a table path value.
    """

    def impl1(bc, path):
        bc = bc.add_or_replace_view("t3", path)
        return bc.sql("select * from t1")

    def impl2(bc):
        return bc.sql("select * from t1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    path = TablePath("bodosql/tests/data/sample-parquet-data/partitioned", "parquet")
    bc = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
            "t4": path,
        }
    )
    # check adding a table
    check_func(impl1, (bc, path), py_output=df1)
    check_func(impl2, (bc,), py_output=df1)


def test_remove_view_table_path(memory_leak_check):
    """Tests add_or_replace_view distributed analysis code
    works when removing or keeping a TablePath.
    """

    def impl1(bc):
        bc = bc.remove_view("t4")
        return bc.sql("select * from t1")

    def impl2(bc):
        bc = bc.remove_view("t2")
        return bc.sql("select * from t1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    path = TablePath("bodosql/tests/data/sample-parquet-data/partitioned", "parquet")
    bc = BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
            "t4": path,
        }
    )
    # check adding a table
    check_func(impl1, (bc,), py_output=df1)
    check_func(impl2, (bc,), py_output=df1)


def test_bodosql_context_global_import(memory_leak_check):
    """Tests that BodoSQLContext works as a global relative import in JIT"""

    def impl(df):
        bc = BodoSQLContext({"t1": df})
        return bc.sql("select * from t1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    check_func(impl, (df,), py_output=df)


def test_bodosql_context_closure_import(memory_leak_check):
    """Tests that BodoSQLContext works as a global relative import in JIT"""
    from bodosql import BodoSQLContext

    def impl(df):
        bc = BodoSQLContext({"t1": df})
        return bc.sql("select * from t1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    check_func(impl, (df,), py_output=df)
