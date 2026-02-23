"""Tests for user facing BodoSQL APIs."""

import datetime
import os
import re
import time

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.conftest import (  # noqa: F401
    iceberg_database,
    iceberg_table_conn,
)
from bodo.tests.utils import (
    _test_equal_guard,
    check_func,
    count_array_OneD_Vars,
    count_array_OneDs,
    count_array_REPs,
    pytest_mark_one_rank,
    pytest_snowflake,
)
from bodo.utils.typing import BodoError
from bodosql import BodoSQLContext, SnowflakeCatalog, TablePath


@pytest.fixture(
    params=[
        pytest.param(
            (
                SnowflakeCatalog(
                    os.environ.get("SF_USERNAME", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "TEST_DB",
                    connection_params={"schema": "PUBLIC"},
                ),
                SnowflakeCatalog(
                    os.environ.get("SF_USERNAME", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "SNOWFLAKE_SAMPLE_DATA",
                    connection_params={"schema": "PUBLIC"},
                ),
            ),
            marks=pytest_snowflake,
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
            "TABLE1": df1,
            "TABLE2": df2,
        }
    )
    bc = bc_orig
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE2"), df2, check_column_type=False
    )
    bc = bc_orig.add_or_replace_view("TABLE3", df3)
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE2"), df2, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE3"), df3, check_column_type=False
    )
    bc = bc.add_or_replace_view("TABLE1", df3)
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE1"), df3, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE2"), df2, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE3"), df3, check_column_type=False
    )

    # The original context should be unchanged.
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from TABLE1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from TABLE2"), df2, check_column_type=False
    )
    with pytest.raises(BodoError):
        bc_orig.sql("select * from TABLE3")


def test_remove_view(memory_leak_check):
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"B": [1, 2, 3]})
    bc_orig = BodoSQLContext(
        {
            "TABLE1": df1,
            "TABLE2": df2,
        }
    )
    bc = bc_orig
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE2"), df2, check_column_type=False
    )
    bc = bc.remove_view("TABLE2")
    pd.testing.assert_frame_equal(
        bc.sql("select * from TABLE1"), df1, check_column_type=False
    )
    with pytest.raises(BodoError):
        bc.sql("select * from TABLE2")
    with pytest.raises(ValueError, match="'name' must refer to a registered view"):
        bc.remove_view("TABLE2")
    bc = bc.remove_view("TABLE1")
    with pytest.raises(BodoError):
        bc.sql("select * from TABLE1")
    with pytest.raises(BodoError):
        bc.sql("select * from TABLE2")

    # The original context should be unchanged.
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from TABLE1"), df1, check_column_type=False
    )
    pd.testing.assert_frame_equal(
        bc_orig.sql("select * from TABLE2"), df2, check_column_type=False
    )


def test_bodosql_context_boxing_no_catalog(datapath, memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context with no catalog
    """

    bc = BodoSQLContext(
        {
            "TABLE1": pd.DataFrame(
                {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
            ),
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
            "TABLE3": TablePath(
                datapath("sample-parquet-data/partitioned"),
                "parquet",
            ),
        },
    )

    def impl(bc):
        return bc

    check_func(impl, (bc,))


def test_bodosql_context_boxing_with_catalog(datapath, memory_leak_check):
    """
    Tests boxing and unboxing with a BodoSQL context with a catalog
    """
    bc = BodoSQLContext(
        {
            "TABLE1": pd.DataFrame(
                {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
            ),
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
            "TABLE3": TablePath(
                datapath("sample-parquet-data/partitioned"),
                "parquet",
            ),
        },
        SnowflakeCatalog(
            os.environ.get("SF_USERNAME", ""),
            os.environ.get("SF_PASSWORD", ""),
            "bodopartner.us-east-1",
            "DEMO_WH",
            "TEST_DB",
            connection_params={"schema": "PUBLIC"},
        ),
    )

    def impl(bc):
        return bc

    check_func(impl, (bc,))


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(
            BodoSQLContext(
                {
                    "TABLE1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
                },
            ),
            id="no-catalog",
        ),
        pytest.param(
            BodoSQLContext(
                {
                    "TABLE1": pd.DataFrame(
                        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
                    ),
                    "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
                },
                SnowflakeCatalog(
                    os.environ.get("SF_USERNAME", ""),
                    os.environ.get("SF_PASSWORD", ""),
                    "bodopartner.us-east-1",
                    "DEMO_WH",
                    "TEST_DB",
                    connection_params={"schema": "PUBLIC"},
                ),
            ),
            id="snowflake-catalog",
            marks=pytest_snowflake,
        ),
    ],
)
def test_bodosql_context_boxed_sql(bc, memory_leak_check):
    """
    Tests unboxing with a BodoSQL context and executing a query.
    """

    def impl(bc):
        return bc.sql("select * from __bodolocal__.TABLE1")

    py_output = bc.tables["TABLE1"]
    check_func(impl, (bc,), py_output=py_output)


def test_bodosql_context_boxed_sql_table_path(
    memory_leak_check, datapath, iceberg_database, iceberg_table_conn
):
    """
    Tests boxing and unboxing with a BodoSQL context where 1
    table uses the table path API.
    """

    def impl(bc):
        return bc.sql("select * from TABLE1")

    py_out = pd.DataFrame(
        {"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25}
    )

    bc = BodoSQLContext(
        {
            "TABLE1": py_out,
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
            "TABLE3": TablePath(datapath("sample-parquet-data/partitioned"), "parquet"),
        }
    )
    check_func(impl, (bc,), py_output=py_out)


def test_add_or_replace_view_jit(memory_leak_check):
    def impl1(bc, df):
        bc = bc.add_or_replace_view("TABLE1", df)
        return bc.sql("select * from TABLE1")

    def impl2(bc):
        return bc.sql("select * from TABLE1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    df3 = pd.DataFrame({"B": [1, 2, 3]})
    bc = BodoSQLContext(
        {
            "TABLE1": df1,
            "TABLE2": df2,
        }
    )
    # check adding a table
    check_func(impl1, (bc, df3), py_output=df3)
    # Check that the original isn't updated
    check_func(impl2, (bc,), py_output=df1)


def test_remove_view_jit(memory_leak_check):
    def impl1(bc):
        bc = bc.remove_view("TABLE1")
        return bc.sql("select * from TABLE1")

    def impl2(bc):
        return bc.sql("select * from TABLE1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    bc = BodoSQLContext(
        {
            "TABLE1": df1,
            "TABLE2": df2,
        }
    )
    # check removing a table
    with pytest.raises(BodoError):
        bodo.jit(impl1)(bc)
    # Check that the original isn't updated
    check_func(impl2, (bc,), py_output=df1)


def test_add_or_replace_view_table_path(datapath, memory_leak_check):
    """Tests add_or_replace_view distributed analysis code
    works when introducing a table path value.
    """

    def impl1(bc, path):
        bc = bc.add_or_replace_view("TABLE3", path)
        return bc.sql("select * from TABLE1")

    def impl2(bc):
        return bc.sql("select * from TABLE1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    path = TablePath(datapath("sample-parquet-data/partitioned"), "parquet")
    bc = BodoSQLContext(
        {
            "TABLE1": df1,
            "TABLE2": df2,
            "TABLE4": path,
        }
    )
    # check adding a table
    check_func(impl1, (bc, path), py_output=df1)
    check_func(impl2, (bc,), py_output=df1)


def test_remove_view_table_path(datapath, memory_leak_check):
    """Tests add_or_replace_view distributed analysis code
    works when removing or keeping a TablePath.
    """

    def impl1(bc):
        bc = bc.remove_view("TABLE4")
        return bc.sql("select * from TABLE1")

    def impl2(bc):
        bc = bc.remove_view("TABLE2")
        return bc.sql("select * from TABLE1")

    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"C": [1, 2, 3]})
    path = TablePath(datapath("sample-parquet-data/partitioned"), "parquet")
    bc = BodoSQLContext(
        {
            "TABLE1": df1,
            "TABLE2": df2,
            "TABLE4": path,
        }
    )
    # check adding a table
    check_func(impl1, (bc,), py_output=df1)
    check_func(impl2, (bc,), py_output=df1)


def test_bodosql_context_global_import(memory_leak_check):
    """Tests that BodoSQLContext works as a global relative import in JIT"""

    def impl(df):
        bc = BodoSQLContext({"TABLE1": df})
        return bc.sql("select * from TABLE1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    check_func(impl, (df,), py_output=df)


def test_bodosql_context_closure_import(memory_leak_check):
    """Tests that BodoSQLContext works as a global relative import in JIT"""
    from bodosql import BodoSQLContext

    def impl(df):
        bc = BodoSQLContext({"TABLE1": df})
        return bc.sql("select * from TABLE1")

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

    bc = BodoSQLContext({"CATALOG_TABLE": local_df})
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
    bc = BodoSQLContext({"CATALOG_TABLE": local_df}, catalog)
    # TODO: Update the expected output
    check_func(impl, (bc,), py_output=local_df)
    bc2 = bc.remove_catalog()
    check_func(impl, (bc2,), py_output=local_df)
    # Verify the old BodoSQLContext is unchanged
    # TODO: Update the expected output
    check_func(impl, (bc,), py_output=local_df)
    with pytest.raises(
        ValueError, match="BodoSQLContext must have an existing catalog registered"
    ):
        bc2.remove_catalog()


def test_add_or_replace_catalog_jit(
    datapath, dummy_snowflake_catalogs, memory_leak_check
):
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
    table_path = TablePath(datapath("sample-parquet-data/partitioned"), "parquet")
    # TODO: Update with real catalogs
    catalog1, catalog2 = dummy_snowflake_catalogs

    bc = BodoSQLContext({"CATALOG_TABLE": local_df, "TABLE2": table_path})
    check_func(impl, (bc, catalog1))
    bc = bc.add_or_replace_catalog(catalog1)
    check_func(impl, (bc, catalog2))


def test_remove_catalog_jit(datapath, dummy_snowflake_catalogs, memory_leak_check):
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
    table_path = TablePath(datapath("sample-parquet-data/partitioned"), "parquet")
    # TODO: Update with a real catalog
    catalog = dummy_snowflake_catalogs[0]

    bc = BodoSQLContext({"CATALOG_TABLE": local_df, "TABLE2": table_path}, catalog)
    check_func(impl, (bc,))
    bc = bc.remove_catalog()
    with pytest.raises(
        BodoError, match="BodoSQLContext must have an existing catalog registered"
    ):
        bodo.jit(impl)(bc)


def test_bodosql_context_arg_dist(memory_leak_check):
    """make sure BodoSQLContext.dataframes is assigned proper distributions if
    BodoSQLContext is passed as an argument [BE-3968]"""

    @bodo.jit(distributed=["bc"])
    def run_query2(bc, q):
        out = bc.sql(q)
        return out

    outermost_QUERY = """select
            ac.num_impressions_1w as num_impressions_1w,
            case when ac.num_clicks_1w >= 100 then ac.num_clicks_1w end as num_clicks_1w
    from _action_counts ac
    """

    rank = bodo.get_rank()
    # it is necessary for ranks to have different data lengths since we are checking
    # 1D vs. 1D_Var issues
    n = 25949 if rank == 0 else 26315
    action_df = pd.DataFrame(
        {"NUM_IMPRESSIONS_1W": np.zeros(n), "NUM_CLICKS_1W": np.zeros(n)}
    )
    bc = BodoSQLContext({"_ACTION_COUNTS": action_df})
    run_query2(bc, outermost_QUERY)

    # all arrays should be 1D_Var, not 1D
    assert count_array_REPs() == 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() > 0


def test_bodosql_context_loop_unrolling(memory_leak_check):
    """
    Make sure loop unrolling for constant inference works after BodoSQL call
    """

    def impl(bc, query):
        output = bc.sql(query)
        output.columns = [str(col).upper() for col in output.columns]
        return output

    query = "select * from TABLE1"
    df = pd.DataFrame({"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25})

    bc = BodoSQLContext(
        {
            "TABLE1": df,
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
        }
    )
    check_func(impl, (bc, query), py_output=df)


@pytest_mark_one_rank
def test_bodosql_context_generate_plan(memory_leak_check):
    """Tests that BodoSQLContext.generate_plan works as expected"""

    from bodosql import BodoSQLContext

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    bc = BodoSQLContext({"TABLE1": df})
    query = "select * from TABLE1"

    # Don't perform a detailed check on the output,
    # since it's a string that can change.
    # the full correctness checking for the generated string
    # should be done in the maven unit tests.
    # This test is just to confirm that we can generate the plan at all in python.
    plan1 = bc.generate_plan(query)
    plan2 = bc.generate_plan(query, show_cost=True)
    assert "cost:" not in plan1
    assert "cost:" in plan2


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("select * from TABLE1", id="valid-query"),
        pytest.param('select * from "table1"', id="invalid-query"),
    ],
)
def test_validate_query(query, request, memory_leak_check):
    """
    Make sure validate query fails on a bad query.
    """
    df = pd.DataFrame({"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25})
    bc = BodoSQLContext(
        {
            "TABLE1": df,
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
        }
    )

    if "invalid" not in request.node.name:
        assert bc.validate_query(query)
    else:
        assert not bc.validate_query(query)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("select * from TABLE1", id="valid-query"),
        pytest.param('select * from "table1"', id="invalid-query"),
    ],
)
def test_fail_validate_also_fails_compile(query, request, memory_leak_check):
    """
    All queries that don't validate also shouldn't compile.
    """
    df = pd.DataFrame({"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25})
    bc = BodoSQLContext(
        {
            "TABLE1": df,
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
        }
    )

    (compiles_flag, _compile_time, _error_message) = bc.validate_query_compiles(query)
    if "invalid" not in request.node.name:
        assert compiles_flag
    else:
        assert not compiles_flag


@pytest.mark.parametrize(
    "query_text",
    [
        """
def impl(TABLE1, TABLE2):
    return 'hello' + True
""",
        """
def impl(TABLE1, TABLE2):
    return TABLE1[TABLE2]
""",
    ],
)
def test_fails_compile(query_text):
    """
    Tests queries that pass validation, but fail during compile.
    Calls a helper function of the sql context, so we don't have to rely on
    generating a query that fails to compile.
    """

    df = pd.DataFrame({"A": np.arange(100), "B": ["r32r", "R32", "Rew", "r32r"] * 25})
    bc = BodoSQLContext(
        {
            "TABLE1": df,
            "TABLE2": pd.DataFrame({"C": [b"345253"] * 100}),
        }
    )

    glbls = {
        "np": np,
        "pd": pd,
        "bodosql": bodosql,
        "re": re,
        "bodo": bodo,
        "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
        "MetaType": bodo.utils.typing.MetaType,
        "numba": numba,
        "time": time,
        "datetime": datetime,
    }

    try:
        bc._functext_compile(query_text, {}, glbls)
    except Exception:
        return

    raise Exception("Should have failed to compile")


def test_sql_jit_options():
    """Test passing JIT options to bc.sql() when called outside JIT"""

    n = 100
    df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
    df_dist = bodo.scatterv(df)

    # Test args_maybe_distributed=True and returns_maybe_distributed=True defaults
    bc = bodosql.BodoSQLContext({"T1": df_dist})
    bc.sql("select sum(B) from T1 group by A")
    assert count_array_REPs() == 0

    # Test distributed flag
    bc = bodosql.BodoSQLContext({"T1": df})
    bc.sql("select sum(B) from T1 group by A", distributed=["T1"])
    assert count_array_REPs() == 0

    # Test replicated data
    bc = bodosql.BodoSQLContext({"T1": df})
    bc.sql("select sum(B) from T1 group by A")
    assert count_array_REPs() > 0

    # Using JIT options inside JIT should raise error
    @bodo.jit
    def f(bc):
        bc.sql("select 1", distributed=False)

    with pytest.raises(
        BodoError,
        match=re.escape(
            r"Argument 'distributed' is not supported for BodoSQLContextType.sql() inside JIT functions"
        ),
    ):
        f(bodosql.BodoSQLContext())


@pytest.mark.bodosql_cpp
def test_plan_execs_cpp_backend(datapath, memory_leak_check):
    """Makes sure C++ backend doesn't execute any plan unnecessarily"""
    if not bodosql.use_cpp_backend:
        return

    import bodo.pandas as bd
    from bodo.pandas.plan import assert_executed_plan_count

    with assert_executed_plan_count(1):
        df1 = bd.read_parquet(datapath("sample-parquet-data/partitioned"))
        bc = BodoSQLContext({"TABLE1": df1})
        out = bc.sql("select * from TABLE1")

    assert isinstance(out, bd.BodoDataFrame)
    pd_out = pd.read_parquet(
        datapath("sample-parquet-data/partitioned"), dtype_backend="pyarrow"
    )
    _test_equal_guard(
        out,
        pd_out,
        sort_output=True,
        check_names=False,
        check_dtype=False,
        reset_index=True,
        check_categorical=False,
        check_pandas_types=False,
    )
