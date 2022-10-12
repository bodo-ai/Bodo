# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for user facing BodoSQL APIs.
"""

import os

import numpy as np
import pandas as pd
import pytest
from bodosql import BodoSQLContext, SnowflakeCatalog, TablePath

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        pytest.param(
            (
                SnowflakeCatalog(
                    os.environ.get("SF_USER", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "TEST_DB",
                    connection_params={"schema": "PUBLIC"},
                ),
                SnowflakeCatalog(
                    os.environ.get("SF_USER", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "SNOWFLAKE_SAMPLE_DATA",
                    connection_params={"schema": "PUBLIC"},
                ),
            ),
            marks=pytest.mark.skipif(
                "AGENT_NAME" not in os.environ, reason="requires Azure Pipelines"
            ),
        )
    ]
)
def dummy_snowflake_catalogs(request):
    """Return a tuple with 2 dummy snowflake catalogs to use
    for testing.
    """
    return request.param


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
    pd.testing.assert_frame_equal(
        bc.sql("select * from t1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t2"), df2, check_column_type=False
    )
    bc = bc_orig.add_or_replace_view("t3", df3)
    pd.testing.assert_frame_equal(
        bc.sql("select * from t1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t2"), df2, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t3"), df3, check_column_type=False
    )
    bc = bc.add_or_replace_view("t1", df3)
    pd.testing.assert_frame_equal(
        bc.sql("select * from t1"), df3, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t2"), df2, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t3"), df3, check_column_type=False
    )

    # The original context should be unchanged.
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from t1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from t2"), df2, check_column_type=False
    )
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
    pd.testing.assert_frame_equal(
        bc.sql("select * from t1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from t2"), df2, check_column_type=False
    )
    bc = bc.remove_view("t2")
    pd.testing.assert_frame_equal(
        bc.sql("select * from t1"), df1, check_column_type=False
    )
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
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from t1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from t2"), df2, check_column_type=False
    )


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(
            BodoSQLContext(
                {
                    "t1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "t2": pd.DataFrame({"C": [b"345253"] * 100}),
                    "t3": TablePath(
                        "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
                    ),
                },
            ),
            id="no-catalog",
        ),
        pytest.param(
            BodoSQLContext(
                {
                    "t1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "t2": pd.DataFrame({"C": [b"345253"] * 100}),
                    "t3": TablePath(
                        "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
                    ),
                },
                SnowflakeCatalog(
                    os.environ.get("SF_USER", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "TEST_DB",
                    connection_params={"schema": "PUBLIC"},
                ),
            ),
            id="snowflake-catalog",
            marks=pytest.mark.skipif(
                "AGENT_NAME" not in os.environ, reason="requires Azure Pipelines"
            ),
        ),
    ],
)
def test_bodosql_context_boxing(bc, memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context.
    """

    def impl(bc):
        return bc

    check_func(impl, (bc,))


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(
            BodoSQLContext(
                {
                    "t1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "t2": pd.DataFrame({"C": [b"345253"] * 100}),
                },
            ),
            id="no-catalog",
        ),
        pytest.param(
            BodoSQLContext(
                {
                    "t1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "t2": pd.DataFrame({"C": [b"345253"] * 100}),
                },
                SnowflakeCatalog(
                    os.environ.get("SF_USER", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "TEST_DB",
                    connection_params={"schema": "PUBLIC"},
                ),
            ),
            id="snowflake-catalog",
            marks=pytest.mark.skipif(
                "AGENT_NAME" not in os.environ, reason="requires Azure Pipelines"
            ),
        ),
    ],
)
def test_bodosql_context_boxed_sql(bc, memory_leak_check):
    """
    Tests unboxing with a BodoSQL context and executing a query.
    """

    def impl(bc):
        return bc.sql("select * from t1")

    py_output = bc.tables["t1"]
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
        bodo.jit(impl1)(bc)
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


def test_add_or_replace_catalog(dummy_snowflake_catalogs, memory_leak_check):
    """
    Verify add_or_replace_catalog properly updates a BodoSQLContext
    with the correct catalog. This is tested by comparing a local table
    and two different Snowflake account and relying on the resolution
    searching the catalog first and then the local table.
    """

    def impl(bc):
        return bc.sql("select * from catalog_table")

    local_df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    # TODO: Update with real catalogs
    catalog1, catalog2 = dummy_snowflake_catalogs

    bc = BodoSQLContext({"catalog_table": local_df})
    check_func(impl, (bc,), py_output=local_df)
    bc2 = bc.add_or_replace_catalog(catalog1)
    # TODO: Update the expected output
    check_func(impl, (bc2,), py_output=local_df)
    # Verify the old BodoSQLContext is unchanged
    check_func(impl, (bc,), py_output=local_df)
    bc3 = bc2.add_or_replace_catalog(catalog2)
    # TODO: Update the expected output
    check_func(impl, (bc3,), py_output=local_df)
    # Verify the old BodoSQLContext is unchanged
    # TODO: Update the expected output
    check_func(impl, (bc2,), py_output=local_df)


def test_remove_catalog(dummy_snowflake_catalogs, memory_leak_check):
    """
    Verify remove_catalog properly updates a BodoSQLContext
    output. This is tested by comparing a local table
    and a Snowflake account and relying on the resolution
    searching the catalog first and then the local table.
    """

    def impl(bc):
        return bc.sql("select * from catalog_table")

    local_df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    # TODO: Update with a real catalog
    catalog = dummy_snowflake_catalogs[0]
    bc = BodoSQLContext({"catalog_table": local_df}, catalog)
    # TODO: Update the expected output
    check_func(impl, (bc,), py_output=local_df)
    bc2 = bc.remove_catalog()
    check_func(impl, (bc2,), py_output=local_df)
    # Verify the old BodoSQLContext is unchanged
    # TODO: Update the expected output
    check_func(impl, (bc,), py_output=local_df)
    with pytest.raises(
        BodoError, match="BodoSQLContext must have an existing catalog registered"
    ):
        bc2.remove_catalog()


def test_add_or_replace_catalog_jit(dummy_snowflake_catalogs, memory_leak_check):
    """
    Verify add_or_replace_catalog properly updates a BodoSQLContext
    with the correct catalog inside JIT. This is tested by comparing
    a local table and two different Snowflake account and relying on
    the resolution searching the catalog first and then the local table.
    """

    def impl(bc, catalog):
        # Load with a new context.
        bc2 = bc.add_or_replace_catalog(catalog)
        df2 = bc2.sql("select * from catalog_table")
        # Reload from the original context
        df1 = bc.sql("select * from catalog_table")
        return (df2, df1)

    local_df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    # Unused table path for checking typing/distribution info
    table_path = TablePath(
        "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
    )
    # TODO: Update with real catalogs
    catalog1, catalog2 = dummy_snowflake_catalogs

    bc = BodoSQLContext({"catalog_table": local_df, "t2": table_path})
    check_func(impl, (bc, catalog1))
    bc = bc.add_or_replace_catalog(catalog1)
    check_func(impl, (bc, catalog2))


def test_remove_catalog_jit(dummy_snowflake_catalogs, memory_leak_check):
    """
    Verify remove_catalog properly updates a BodoSQLContext
    with the correct catalog inside JIT. This is tested by comparing
    a local table and a Snowflake account and relying on
    the resolution searching the catalog first and then the local table.
    """

    def impl(bc):
        # Load with a new context.
        bc2 = bc.remove_catalog()
        df2 = bc2.sql("select * from catalog_table")
        # Reload from the original context
        df1 = bc.sql("select * from catalog_table")
        return (df2, df1)

    local_df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    # Unused table path for checking typing/distribution info
    table_path = TablePath(
        "bodosql/tests/data/sample-parquet-data/partitioned", "parquet"
    )
    # TODO: Update with a real catalog
    catalog = dummy_snowflake_catalogs[0]

    bc = BodoSQLContext({"catalog_table": local_df, "t2": table_path}, catalog)
    check_func(impl, (bc,))
    bc = bc.remove_catalog()
    with pytest.raises(
        BodoError, match="BodoSQLContext must have an existing catalog registered"
    ):
        bodo.jit(impl)(bc)
