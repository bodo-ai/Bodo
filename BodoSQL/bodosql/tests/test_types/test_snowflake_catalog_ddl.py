"""
Tests DDL operations on a BodoSQL catalog.
"""

import os
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest
import snowflake.connector

import bodo
import bodosql
from bodo.tests.utils import (
    _test_equal_guard,
    check_func_seq,
    create_snowflake_table_from_select_query,
    get_snowflake_connection_string,
    pytest_snowflake,
)
from bodo.tests.utils_jit import reduce_sum
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (
    test_db_snowflake_catalog,  # noqa
)
from bodosql.tests.utils import assert_equal_par, gen_unique_id

pytestmark = pytest_snowflake


def check_schema_exists(conn_str, schema_name) -> bool:
    tables = pd.read_sql(
        f"SHOW SCHEMAS LIKE '{schema_name}' STARTS WITH '{schema_name}'",
        conn_str,
    )
    return len(tables) == 1


@contextmanager
def schema_helper(conn_str, schema_name, create=True):
    if create:
        if bodo.get_rank() == 0:
            pd.read_sql(f"CREATE SCHEMA {schema_name}", conn_str)
            assert check_schema_exists(conn_str, schema_name)
        bodo.barrier()

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP SCHEMA IF EXISTS {schema_name}", conn_str)
            assert not check_schema_exists(conn_str, schema_name)


def check_view_exists(conn_str, view_name) -> bool:
    tables = pd.read_sql(
        f"SHOW VIEWS LIKE '{view_name}' STARTS WITH '{view_name}'",
        conn_str,
    )
    return len(tables) == 1


@contextmanager
def view_helper(conn_str, view_name, create=True):
    """A helper function to drop the view at end of context.
    Setting create = True will create a view with given name.
    """
    if create:
        if bodo.get_rank() == 0:
            pd.read_sql(f"CREATE VIEW {view_name} AS SELECT 0 AS A", conn_str)
            assert check_view_exists(conn_str, view_name)
        bodo.barrier()

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn_str)
            assert not check_view_exists(conn_str, view_name)


def check_table_exists(conn_str, table_name) -> bool:
    tables = pd.read_sql(
        f"SHOW TABLES LIKE '{table_name}' STARTS WITH '{table_name}'",
        conn_str,
    )
    return len(tables) == 1


@contextmanager
def table_helper(conn_str, table_name, create=True):
    """A helper function to drop the table at end of context.
    Setting create = True will create a table with given name.
    """
    if create:
        if bodo.get_rank() == 0:
            pd.read_sql(f"CREATE TABLE {table_name} AS SELECT 0 AS A", conn_str)
        bodo.barrier()
        assert check_table_exists(conn_str, table_name)

    try:
        yield
    finally:
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP TABLE IF EXISTS {table_name}", conn_str)
        bodo.barrier()
        assert not check_table_exists(conn_str, table_name)


#####################
#   CREATE tests    #
#####################


@pytest.mark.parametrize("if_not_exists", [True, False])
def test_create_schema(if_not_exists, test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo can create a schema in Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    if_not_exists_str = "IF NOT EXISTS" if if_not_exists else ""
    query = f"CREATE SCHEMA {if_not_exists_str} {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' successfully created."]}
    )

    # execute_ddl Version
    with schema_helper(conn, schema_name, create=False):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        assert check_schema_exists(conn, schema_name)

    # Python Version
    with schema_helper(conn, schema_name, create=False):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        assert check_schema_exists(conn, schema_name)

    # Jit Version
    # Intentionally returns replicated output
    with schema_helper(conn, schema_name, create=False):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=False,
        )
        assert check_schema_exists(conn, schema_name)


def test_create_schema_already_exists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that Bodo errors when creating a schema
    that already exists in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    query = f"CREATE SCHEMA {schema_name}"
    py_output = pd.DataFrame({"STATUS": []})

    with schema_helper(conn, schema_name):
        with pytest.raises(ValueError, match=f"Schema '{schema_name}' already exists."):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query),
                py_output=py_output,
                test_str_literal=True,
            )
            assert check_schema_exists(conn, schema_name)


def test_create_schema_if_not_exists_already_exists(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that Bodo doesn't error when creating a schema
    that already exists in Snowflake, when specifying IF NOT EXISTS.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"'{schema_name}' already exists, statement succeeded."]}
    )

    with schema_helper(conn, schema_name):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=True,
        )
        assert check_schema_exists(conn, schema_name)


def test_create_view(test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo can create a view in Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully created."]})

    def verify_view_created():
        assert check_view_exists(conn, view_name)
        x = pd.read_sql(f"SELECT * FROM {view_name}", conn)
        assert "a" in x
        assert x["a"].shape == (1,)
        assert x["a"][0] == "testview"

    # execute_ddl Version
    with view_helper(conn, view_name, create=False):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Python Version
    with view_helper(conn, view_name, create=False):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Jit Version
    # Intentionally returns replicated output
    with view_helper(conn, view_name, create=False):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=False,
        )
        verify_view_created()


def test_create_view_validates(test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo validates view definitions before submitting to Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    schema_1 = gen_unique_id("SCEHMA1").upper()
    schema_2 = gen_unique_id("SCEHMA2").upper()

    # this test creates two schemas - schema_1 and schema_2, and creates a table
    # at SCHEMA_1.TABLE1. We then ensure that view validation will fail if we
    # create a view referencing TABLE1 in SCHEMA_2, and that the column of
    # TABLE1 are validated when we create a view in SCHEMA_1.

    catalog = deepcopy(test_db_snowflake_catalog)
    with schema_helper(conn, schema_1, create=True):
        catalog.connection_params["schema"] = schema_1
        bc = bodosql.BodoSQLContext(catalog=catalog)
        table_query = f"CREATE OR REPLACE TABLE {schema_1}.TABLE1 AS SELECT 0 as A"
        bc.sql(table_query)
        with schema_helper(conn, schema_2, create=True):
            with pytest.raises(ValueError, match="Object 'TABLE1' not found"):
                query = f"CREATE OR REPLACE VIEW {schema_2}.VIEW2 AS SELECT A + 1 as A from TABLE1"
                bc.execute_ddl(query)

        # Test that the view validates if ran in the correct schema
        py_output = pd.DataFrame({"STATUS": ["View 'VIEW2' successfully created."]})
        query = (
            f"CREATE OR REPLACE VIEW {schema_1}.VIEW2 AS SELECT A + 1 as A from TABLE1"
        )
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)

        # Column B does not exist - validation should fail
        with pytest.raises(ValueError, match="Column 'B' not found"):
            query = f"CREATE OR REPLACE VIEW {schema_1}.VIEW3 AS SELECT B + 1 as B from TABLE1"
            bc.execute_ddl(query)


###################
#   DROP tests    #
###################


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_schema(if_exists, test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo can drop a schema in Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query = f"DROP SCHEMA {if_exists_str} {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' successfully dropped."]}
    )

    # execute_ddl Version
    with schema_helper(conn, schema_name):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        assert not check_schema_exists(conn, schema_name)

    # Python Version
    with schema_helper(conn, schema_name):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        assert not check_schema_exists(conn, schema_name)

    # Jit Version
    # Intentionally returns replicated output
    with schema_helper(conn, schema_name):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=False,
        )
        assert not check_schema_exists(conn, schema_name)


def test_drop_schema_doesnt_exists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that Bodo errors when dropping a schema
    that doesn't exist in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    query = f"DROP SCHEMA {schema_name}"
    py_output = pd.DataFrame({"STATUS": []})

    with pytest.raises(
        ValueError,
        match=f"Schema '{schema_name}' does not exist or drop cannot be performed.",
    ):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=True,
        )
        assert not check_schema_exists(conn, schema_name)


def test_drop_schema_if_exists_doesnt_exists(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that Bodo doesn't error when dropping a schema
    that doesn't exists in Snowflake, when specifying IF EXISTS.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA").upper()
    query = f"DROP SCHEMA IF EXISTS {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' already dropped, statement succeeded."]}
    )

    check_func_seq(
        lambda bc, query: bc.sql(query),
        (bc, query),
        py_output=py_output,
        test_str_literal=True,
    )
    assert not check_schema_exists(conn, schema_name)


@pytest.mark.parametrize("purge", [True, False])
def test_drop_table(purge, test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    purge_str = "PURGE" if purge else ""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a table
    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "drop_table_test", db, schema
    ) as table_name:
        case_insenstive_table_name = table_name.upper()
        # Drop Execute a drop table query.
        query = f"DROP TABLE {table_name} {purge_str}"
        py_output = pd.DataFrame(
            {"STATUS": [f"{case_insenstive_table_name} successfully dropped."]}
        )
        bodo_output = impl(bc, query)
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output=True,
        )
        # count how many pes passed the test, since throwing exceptions directly
        # can lead to inconsistency across pes and hangs
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Sequential test failed"
        # Verify we can't find the table.
        tables = pd.read_sql(
            f"SHOW TABLES LIKE '{case_insenstive_table_name}' STARTS WITH '{case_insenstive_table_name}'",
            conn,
        )
        assert len(tables) == 0, "Table was not dropped"


@pytest.mark.parametrize("purge", [True, False])
def test_drop_table_execute_ddl(purge, test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake using bc.execute_ddl
    from Python"""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    purge_str = "PURGE" if purge else ""

    # Create a table
    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "drop_table_test", db, schema
    ) as table_name:
        case_insenstive_table_name = table_name.upper()
        # Drop Execute a drop table query.
        query = f"DROP TABLE {table_name} {purge_str}"
        py_output = pd.DataFrame(
            {"STATUS": [f"{case_insenstive_table_name} successfully dropped."]}
        )
        bodo_output = bc.execute_ddl(query)
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output=True,
        )
        # count how many pes passed the test, since throwing exceptions directly
        # can lead to inconsistency across pes and hangs
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Sequential test failed"
        # Verify we can't find the table.
        tables = pd.read_sql(
            f"SHOW TABLES LIKE '{case_insenstive_table_name}' STARTS WITH '{case_insenstive_table_name}'",
            conn,
        )
        assert len(tables) == 0, "Table was not dropped"


@pytest.mark.parametrize("purge", [True, False])
def test_drop_table_case_sensitive(purge, test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake that
    is case sensitive."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    purge_str = "PURGE" if purge else ""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a table
    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "drop_table_test", db, schema, case_sensitive=True
    ) as table_name:
        # Unwrap the quotes for non-identifier queries
        unwrapped_table_name = table_name[1:-1]
        # Drop Execute a drop table query.
        query = f"DROP TABLE {table_name} {purge_str}"
        py_output = pd.DataFrame(
            {"STATUS": [f"{unwrapped_table_name} successfully dropped."]}
        )
        bodo_output = impl(bc, query)
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output=True,
        )
        # count how many pes passed the test, since throwing exceptions directly
        # can lead to inconsistency across pes and hangs
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Sequential test failed"
        # Verify we can't find the table.
        tables = pd.read_sql(
            f"SHOW TABLES LIKE '{unwrapped_table_name}' STARTS WITH '{unwrapped_table_name}'",
            conn,
        )
        assert len(tables) == 0, "Table was not dropped"


@pytest.mark.parametrize("purge", [True, False])
def test_drop_table_not_found(purge, test_db_snowflake_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Snowflake raises an error."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    purge_str = "PURGE" if purge else ""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    tables = pd.read_sql(
        f"SHOW TABLES LIKE '{table_name}' STARTS WITH '{table_name}'",
        conn,
    )
    assert len(tables) == 0, "Table exists. Need to test with a non-existent table."
    with pytest.raises(ValueError, match=""):
        query = f"DROP TABLE {table_name} {purge_str}"
        impl(bc, query)


@pytest.mark.parametrize("purge", [True, False])
def test_drop_table_not_found_if_exists(
    purge, test_db_snowflake_catalog, memory_leak_check
):
    """Tests a table that doesn't exist in Snowflake doesn't raise an error
    with IF EXISTS."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    purge_str = "PURGE" if purge else ""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    tables = pd.read_sql(
        f"SHOW TABLES LIKE '{table_name}' STARTS WITH '{table_name}'",
        conn,
    )
    assert len(tables) == 0, "Table exists. Please choose a different table name."
    query = f"DROP TABLE IF EXISTS {table_name} {purge_str}"
    py_output = pd.DataFrame(
        {
            "STATUS": [
                f"Drop statement executed successfully ({table_name} already dropped)."
            ]
        }
    )
    bodo_output = impl(bc, query)
    passed = _test_equal_guard(
        bodo_output,
        py_output,
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_view(if_exists, test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo can drop a view in Snowflake if the view does exist."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP VIEW {if_exists_str} {view_name}"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully dropped."]})

    # execute_ddl Version
    with view_helper(conn, view_name, create=True):
        bodo_output = bc.execute_ddl(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
        assert not check_view_exists(conn, view_name)

    # Python Version
    with view_helper(conn, view_name, create=True):
        bodo_output = bc.sql(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
        assert not check_view_exists(conn, view_name)

    # Jit Version
    with view_helper(conn, view_name, create=True):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_drop_view),
            py_output=py_output,
            test_str_literal=False,
        )
        assert not check_view_exists(conn, view_name)


@pytest.mark.parametrize("if_exists", [True, False])
@pytest.mark.parametrize("tableview", ["Table", "View"])
def test_drop_viewortable_bug(
    test_db_snowflake_catalog, if_exists, tableview, memory_leak_check
):
    """Test for fix for both drop view and drop table"""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    tableview_name = "TEST_DB.TPCH_SF1.UNEXIST_TABLE"
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP {tableview} {if_exists_str} {tableview_name}"
    py_success_output = pd.DataFrame(
        {
            "STATUS": [
                f"Drop statement executed successfully ({tableview_name} already dropped)."
            ]
        }
    )

    # execute_ddl Version
    if if_exists:
        bodo_output = bc.execute_ddl(query_drop_view)
        _test_equal_guard(bodo_output, py_success_output)
    else:
        with pytest.raises(
            ValueError,
            match=f"{tableview} '{tableview_name}' does not exist or not authorized to drop.",
        ):
            bodo_output = bc.execute_ddl(query_drop_view)

    # Jit Version
    if if_exists:
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_drop_view),
            py_output=py_success_output,
            test_str_literal=False,
        )
    else:
        with pytest.raises(
            ValueError,
            match=f"{tableview} '{tableview_name}' does not exist or not authorized to drop.",
        ):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_drop_view),
                py_output="",
                test_str_literal=False,
            )


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_view_error_does_not_exist(
    if_exists, test_db_snowflake_catalog, memory_leak_check
):
    """Tests that Bodo can drop a view in Snowflake if the view does not exist with if_exists.
    Tests that Bodo signals error in Snowflake if the view does not exist without if_exists.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP VIEW {if_exists_str} {view_name}"
    py_output = pd.DataFrame(
        {
            "STATUS": [
                f"Drop statement executed successfully ({view_name} already dropped)."
            ]
        }
    )

    with view_helper(conn, view_name, create=True):
        pass
    # execute_ddl Version
    if if_exists:
        bodo_output = bc.execute_ddl(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
    else:
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to drop.",
        ):
            bc.execute_ddl(query_drop_view)

    # Python Version
    if if_exists:
        bodo_output = bc.sql(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
    else:
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to drop.",
        ):
            bc.sql(query_drop_view)

    # Jit Version
    if if_exists:
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_drop_view),
            py_output=py_output,
            test_str_literal=False,
        )
    else:
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to drop.",
        ):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_drop_view),
                py_output=py_output,
                test_str_literal=False,
            )


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_view_error_non_view(
    if_exists, test_db_snowflake_catalog, memory_leak_check
):
    """Tests that Bodo signals error in Snowflake if the path is not a view."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    view_name = gen_unique_id("TEST_VIEW").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP VIEW {if_exists_str} {view_name}"
    with table_helper(conn, view_name, create=True):
        # execute_ddl Version
        with pytest.raises(
            ValueError, match="Unable to drop Snowflake view from query"
        ):
            bc.execute_ddl(query_drop_view)

        # Python Version
        with pytest.raises(
            ValueError, match="Unable to drop Snowflake view from query"
        ):
            bc.sql(query_drop_view)

        # Jit Version
        with pytest.raises(
            ValueError, match="Unable to drop Snowflake view from query"
        ):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_drop_view),
                py_output="",
                test_str_literal=False,
            )


#####################
# ALTER TABLE Tests #
#####################


def test_alter_table_rename(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a table using ALTER TABLE in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "alter_table_rename_test", db, schema
    ) as table_name:
        try:
            pd.read_sql(
                f"DROP TABLE IF EXISTS {table_name}_renamed", conn
            )  # Clean up if previously existed
            # Execute ALTER TABLE query.
            query = f"ALTER TABLE {table_name} RENAME TO {table_name}_renamed"
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = impl(bc, query)
            assert_equal_par(
                bodo_output,
                py_output,
            )
            # Verify there exists a table called renamedTable.
            tables = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}_renamed'",
                conn,
            )
            assert len(tables) == 1, "Renamed table was not found"
        finally:
            # Clean up after
            bc.sql(f"DROP TABLE IF EXISTS {table_name}")
            bc.sql(f"DROP TABLE IF EXISTS {table_name}_renamed")


def test_alter_table_rename_compound(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a table using ALTER TABLE in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "alter_table_rename_test", db, schema
    ) as table_name:
        try:
            pd.read_sql(
                f"DROP TABLE IF EXISTS {table_name}_renamed", conn
            )  # Clean up if previously existed
            # Execute ALTER TABLE query.
            query = f"ALTER TABLE {schema}.{table_name} RENAME TO {schema}.{table_name}_renamed"
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = impl(bc, query)
            assert_equal_par(
                bodo_output,
                py_output,
            )
            # Verify there exists a table called renamedTable.
            tables = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}_renamed'",
                conn,
            )
            assert len(tables) == 1, "Renamed table was not found"
        finally:
            # Clean up after
            bc.sql(f"DROP TABLE IF EXISTS {table_name}")
            bc.sql(f"DROP TABLE IF EXISTS {table_name}_renamed")


def test_alter_table_rename_diffschema(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a table using ALTER TABLE in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "alter_table_rename_test", db, schema
    ) as table_name:
        try:
            pd.read_sql(
                f"DROP TABLE IF EXISTS {table_name}_renamed", conn
            )  # Clean up if previously existed

            # Rename into non-existent schema
            with pytest.raises(ValueError, match="does not exist or not authorized."):
                query = f"ALTER TABLE {schema}.{table_name} RENAME TO NONEXISTENT_SCHEMA.{table_name}_renamed"
                bodo_output = impl(bc, query)

            # Execute ALTER TABLE query.
            query = f"ALTER TABLE {schema}.{table_name} RENAME TO TEST_SCHEMA.{table_name}_renamed"
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = impl(bc, query)
            assert_equal_par(
                bodo_output,
                py_output,
            )
            # Verify there exists a table called renamedTable.
            tables = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}_renamed' IN SCHEMA TEST_SCHEMA",
                conn,
            )
            assert len(tables) == 1, "Renamed table was not found"
        finally:
            # Clean up after
            bc.sql(f"DROP TABLE IF EXISTS {table_name}")
            bc.sql(f"DROP TABLE IF EXISTS {table_name}_renamed")
            bc.sql(f"DROP TABLE IF EXISTS TEST_SCHEMA.{table_name}_renamed")


def test_alter_table_rename_ifexists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a table using ALTER TABLE in Snowflake, with the IF EXISTS option.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "alter_table_rename_ifexists_test", db, schema
    ) as table_name:
        try:
            pd.read_sql(
                f"DROP TABLE IF EXISTS {table_name}_renamed", conn
            )  # Clean up if previously existed
            # Execute ALTER TABLE query with IF EXISTS option.
            query = f"ALTER TABLE IF EXISTS {table_name} RENAME TO {table_name}_renamed"
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = impl(bc, query)
            assert_equal_par(
                bodo_output,
                py_output,
            )
            # Verify there exists a table called renamedTable.
            tables = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}_renamed'",
                conn,
            )
            assert len(tables) == 1, "Renamed table was not found"
        finally:
            # Clean up after
            bc.sql(f"DROP TABLE IF EXISTS {table_name}")
            bc.sql("DROP TABLE IF EXISTS renamedTable")


def test_alter_table_rename_not_found(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that attempting to ALTER a non-existent table without
    the IF EXISTS option raises an error.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    tables = pd.read_sql(
        f"SHOW TABLES LIKE '{table_name}' STARTS WITH '{table_name}'",
        conn,
    )
    assert len(tables) == 0, "Table exists. Need to test with a non-existent table."

    # This should now throw an error.
    with pytest.raises(ValueError, match="does not exist or not authorized"):
        query = f"ALTER TABLE {table_name} RENAME TO {table_name}_renamed"
        impl(bc, query)


def test_alter_table_rename_ifexists_not_found(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that attempting to ALTER a non-existent table with
    the IF EXISTS option does not raise an error.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    tables = pd.read_sql(
        f"SHOW TABLES LIKE '{table_name}' STARTS WITH '{table_name}'",
        conn,
    )
    assert len(tables) == 0, "Table exists. Need to test with a non-existent table."

    # This query should NOT throw an error, and instead gracefully return a status message.
    query = f"ALTER TABLE IF EXISTS {table_name} RENAME TO {table_name}_renamed"
    py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
    bodo_output = impl(bc, query)
    assert_equal_par(bodo_output, py_output)


def test_alter_table_rename_alreadyexists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests if renaming a table to an already existing table
    using ALTER TABLE in Snowflake will throw an error.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "alter_table_rename_test", db, schema
    ) as table_name:
        try:
            pd.read_sql(
                f"DROP TABLE IF EXISTS {table_name}_renamed", conn
            )  # Clean up if previously existed
            # Create another table
            bc.sql(f"CREATE OR REPLACE TABLE {table_name}_renamed AS SELECT 1 as A")
            # Execute ALTER TABLE query.
            query = f"ALTER TABLE {table_name} RENAME TO {table_name}_renamed"
            # This should now throw an error.
            with pytest.raises(ValueError, match="already exists"):
                impl(bc, query)

        finally:
            # Clean up after
            bc.sql(f"DROP TABLE IF EXISTS {table_name}")
            bc.sql(f"DROP TABLE IF EXISTS {table_name}_renamed")


# Unsupported operations tests
def test_alter_table_not_supported(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that attempt to use not-yet supported ALTER TABLE operations.
    """

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # These should all throw an error.
    # NOTE: These should all preferrably throw a nice error such as
    #       "This DDL operation is not supported yet", but this requires
    #       changing more of the parser, so I think this is lower priority for now.
    with pytest.raises(ValueError, match="This DDL operation is currently unsupported"):
        query = "ALTER TABLE test1 SWAP WITH renamedTable"
        impl(bc, query)

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        query = "ALTER TABLE test1 CLUSTER BY test2"
        impl(bc, query)

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        query = "ALTER TABLE test1 SET test2"
        impl(bc, query)


#####################
# ALTER VIEW Tests  #
#####################


def test_alter_view_rename(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a view using ALTER VIEW in Snowflake.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    view_name = gen_unique_id("TEST_VIEW").upper()
    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully created."]})

    # Create view
    bodo_output = pd.read_sql(query, conn)

    try:
        pd.read_sql(
            "DROP VIEW IF EXISTS renamedView", conn
        )  # Clean up if previously existed

        # Execute ALTER TABLE query.
        query = f"ALTER VIEW {view_name} RENAME TO {view_name}_renamed"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = impl(bc, query)
        assert_equal_par(
            bodo_output,
            py_output,
        )
        # Verify there exists a table called renamedTable.
        tables = pd.read_sql(
            f"SHOW VIEWS LIKE '{view_name}_renamed'",
            conn,
        )
        assert len(tables) == 1, "Renamed view was not found"
    finally:
        # Clean up after
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn)
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}_renamed", conn)


def test_alter_view_rename_ifexists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that we can rename a view using ALTER VIEW in Snowflake, with the IF EXISTS option.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    view_name = gen_unique_id("TEST_VIEW").upper()
    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A"
    py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})

    # Create view with a call to Snowflake
    bodo_output = pd.read_sql(query, conn)

    try:
        # Execute ALTER TABLE query with IF EXISTS option.
        query = f"ALTER VIEW IF EXISTS {view_name} RENAME TO {view_name}_renamed"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = impl(bc, query)
        assert_equal_par(
            bodo_output,
            py_output,
        )
        # Verify there exists a table called renamedTable.
        tables = pd.read_sql(
            f"SHOW VIEWS LIKE '{view_name}_renamed'",
            conn,
        )
        assert len(tables) == 1, "Renamed view was not found"
    finally:
        # Clean up afterwards!
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn)
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}_renamed", conn)


def test_alter_view_rename_not_found(test_db_snowflake_catalog, memory_leak_check):
    """
     Tests that attempting to ALTER a non-existent view without
    the IF EXISTS option raises an error.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage table name.
    view_name = "FEJWIOPFE13_9029J03C32"
    views = pd.read_sql(
        f"SHOW VIEWS LIKE '{view_name}' STARTS WITH '{view_name}'",
        conn,
    )
    assert len(views) == 0, "View exists. Need to test with a non-existent view."

    # This should now throw an error.
    with pytest.raises(ValueError, match="does not exist or not authorized"):
        query = f"ALTER VIEW {view_name} RENAME TO {view_name}_renamed"
        impl(bc, query)


def test_alter_view_rename_ifexists_not_found(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that attempting to ALTER a non-existent view with
    the IF EXISTS option does NOT raise an error.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # Create a garbage view name.
    view_name = "FEJWIOPFE13_9029J03C32"
    views = pd.read_sql(
        f"SHOW VIEWS LIKE '{view_name}' STARTS WITH '{view_name}'",
        conn,
    )
    assert len(views) == 0, "View exists. Need to test with a non-existent view."

    # This query should NOT throw an error, and instead gracefully return a status message.
    query = f"ALTER VIEW IF EXISTS {view_name} RENAME TO {view_name}_renamed"
    py_output = py_output = pd.DataFrame(
        {"STATUS": ["Statement executed successfully."]}
    )
    bodo_output = impl(bc, query)
    assert_equal_par(
        bodo_output,
        py_output,
    )


def test_alter_view_rename_alreadyexists(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests if renaming a view to an already existing view
    using ALTER VIEW in Snowflake will throw an error.
    """

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    view_name = gen_unique_id("TEST_VIEW").upper()

    # Create views
    pd.read_sql(f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A", conn)
    pd.read_sql(
        f"CREATE OR REPLACE VIEW {view_name}_renamed AS SELECT 'testview' as A", conn
    )

    try:
        # Execute ALTER TABLE query.
        query = f"ALTER VIEW {view_name} RENAME TO {view_name}_renamed"
        # This should now throw an error.
        with pytest.raises(ValueError, match="already exists"):
            impl(bc, query)
    finally:
        # Clean up after
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn)
        pd.read_sql(f"DROP VIEW IF EXISTS {view_name}_renamed", conn)


# Unsupported operations tests
def test_alter_view_not_supported(test_db_snowflake_catalog, memory_leak_check):
    """Tests that attempt to use not-yet supported ALTER VIEW operations."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    # These should all throw an error.
    # NOTE: These should all preferrably throw a nice error such as
    #       "This DDL operation is not supported yet", but this requires
    #       changing more of the parser, so I think this is lower priority for now.
    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        query = "ALTER VIEW test1 SET COMMENT = 'test'"
        impl(bc, query)

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        query = "ALTER VIEW test1 SET SECURE"
        impl(bc, query)

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        query = "ALTER VIEW test1 SET SECURE"
        impl(bc, query)


###################
# DESCRIBE tests  #
###################


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_table(describe_keyword, test_db_snowflake_catalog, memory_leak_check):
    """Tests that describe table works on a Snowflake table."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl(
        f"{describe_keyword} TABLE SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM"
    )
    expected_output = pd.DataFrame(
        {
            "NAME": [
                "L_ORDERKEY",
                "L_PARTKEY",
                "L_SUPPKEY",
                "L_LINENUMBER",
                "L_QUANTITY",
                "L_EXTENDEDPRICE",
                "L_DISCOUNT",
                "L_TAX",
                "L_RETURNFLAG",
                "L_LINESTATUS",
                "L_SHIPDATE",
                "L_COMMITDATE",
                "L_RECEIPTDATE",
                "L_SHIPINSTRUCT",
                "L_SHIPMODE",
                "L_COMMENT",
            ],
            "TYPE": [
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "VARCHAR(1)",
                "VARCHAR(1)",
                "DATE",
                "DATE",
                "DATE",
                "VARCHAR(25)",
                "VARCHAR(10)",
                "VARCHAR(44)",
            ],
            "KIND": ["COLUMN"] * 16,
            "NULL?": ["N"] * 16,
            "DEFAULT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIMARY_KEY": ["N"] * 16,
            "UNIQUE_KEY": ["N"] * 16,
            "CHECK": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "EXPRESSION": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "COMMENT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "POLICY NAME": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIVACY DOMAIN": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
        }
    )
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Describe table test failed"


def test_describe_table_compiles_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that describe table compiles in JIT."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "DESCRIBE TABLE SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM"
    bc.validate_query_compiles(query)


###################
#   SHOW tests    #
###################


def _show_tables_snowflake_sample_data_tpch_sf1_output(terse=True):
    if terse:
        return pd.DataFrame(
            {
                "CREATED_ON": [
                    "2021-11-11 13:17:32.899 -0800",
                    "2021-11-11 13:17:33.269 -0800",
                    "2021-11-11 13:17:33.308 -0800",
                    "2021-11-11 13:17:33.242 -0800",
                    "2021-11-11 13:17:33.225 -0800",
                    "2021-11-11 13:17:33.199 -0800",
                    "2021-11-11 13:17:34.582 -0800",
                    "2021-11-11 13:17:34.593 -0800",
                ],
                "NAME": [
                    "CUSTOMER",
                    "LINEITEM",
                    "NATION",
                    "ORDERS",
                    "PART",
                    "PARTSUPP",
                    "REGION",
                    "SUPPLIER",
                ],
                "SCHEMA_NAME": ["SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"] * 8,
                "KIND": ["TABLE"] * 8,
            }
        )
    else:
        return pd.DataFrame(
            {
                "CREATED_ON": [
                    "2021-11-11 13:17:32.899 -0800",
                    "2021-11-11 13:17:33.269 -0800",
                    "2021-11-11 13:17:33.308 -0800",
                    "2021-11-11 13:17:33.242 -0800",
                    "2021-11-11 13:17:33.225 -0800",
                    "2021-11-11 13:17:33.199 -0800",
                    "2021-11-11 13:17:34.582 -0800",
                    "2021-11-11 13:17:34.593 -0800",
                ],
                "NAME": [
                    "CUSTOMER",
                    "LINEITEM",
                    "NATION",
                    "ORDERS",
                    "PART",
                    "PARTSUPP",
                    "REGION",
                    "SUPPLIER",
                ],
                "SCHEMA_NAME": ["SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"] * 8,
                "KIND": ["TABLE"] * 8,
                "COMMENT": [
                    "Customer data as defined by TPC-H",
                    "Lineitem data as defined by TPC-H",
                    "Nation data as defined by TPC-H",
                    "Orders data as defined by TPC-H",
                    "Part data as defined by TPC-H",
                    "Partsupp data as defined by TPC-H",
                    "Region data as defined by TPC-H",
                    "Supplier data as defined by TPC-H",
                ],
                "CLUSTER_BY": [""] * 8,
                "ROWS": [
                    Decimal("150000"),
                    Decimal("6001215"),
                    Decimal("25"),
                    Decimal("1500000"),
                    Decimal("200000"),
                    Decimal("800000"),
                    Decimal("5"),
                    Decimal("10000"),
                ],
                "BYTES": [
                    Decimal("10747904"),
                    Decimal("165228544"),
                    Decimal("4096"),
                    Decimal("42303488"),
                    Decimal("5214208"),
                    Decimal("36589568"),
                    Decimal("4096"),
                    Decimal("692224"),
                ],
                "OWNER": [""] * 8,
                "RETENTION_TIME": ["1"] * 8,
                "AUTOMATIC_CLUSTERING": ["OFF"] * 8,
                "CHANGE_TRACKING": ["OFF"] * 8,
                "IS_EXTERNAL": ["N"] * 8,
                "ENABLE_SCHEMA_EVOLUTION": ["N"] * 8,
                "OWNER_ROLE_TYPE": [""] * 8,
                "IS_EVENT": ["N"] * 8,
                "IS_HYBRID": ["N"] * 8,
                "IS_ICEBERG": ["N"] * 8,
                "IS_IMMUTABLE": ["N"] * 8,
            }
        )


def _show_objects_snowflake_sample_data_tpch_sf1_output():
    return pd.DataFrame(
        {
            "CREATED_ON": [
                "2021-11-11 13:17:32.899 -0800",
                "2021-11-11 13:17:33.269 -0800",
                "2021-11-11 13:17:33.308 -0800",
                "2021-11-11 13:17:33.242 -0800",
                "2021-11-11 13:17:33.225 -0800",
                "2021-11-11 13:17:33.199 -0800",
                "2021-11-11 13:17:34.582 -0800",
                "2021-11-11 13:17:34.593 -0800",
            ],
            "NAME": [
                "CUSTOMER",
                "LINEITEM",
                "NATION",
                "ORDERS",
                "PART",
                "PARTSUPP",
                "REGION",
                "SUPPLIER",
            ],
            "SCHEMA_NAME": ["SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"] * 8,
            "KIND": ["TABLE"] * 8,
            "COMMENT": [
                "Customer data as defined by TPC-H",
                "Lineitem data as defined by TPC-H",
                "Nation data as defined by TPC-H",
                "Orders data as defined by TPC-H",
                "Part data as defined by TPC-H",
                "Partsupp data as defined by TPC-H",
                "Region data as defined by TPC-H",
                "Supplier data as defined by TPC-H",
            ],
            "CLUSTER_BY": [""] * 8,
            "ROWS": [
                Decimal("150000"),
                Decimal("6001215"),
                Decimal("25"),
                Decimal("1500000"),
                Decimal("200000"),
                Decimal("800000"),
                Decimal("5"),
                Decimal("10000"),
            ],
            "BYTES": [
                Decimal("10747904"),
                Decimal("165228544"),
                Decimal("4096"),
                Decimal("42303488"),
                Decimal("5214208"),
                Decimal("36589568"),
                Decimal("4096"),
                Decimal("692224"),
            ],
            "OWNER": [""] * 8,
            "RETENTION_TIME": ["1"] * 8,
            "OWNER_ROLE_TYPE": [""] * 8,
        }
    )


def test_show_objects_terse(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show objects works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW TERSE OBJECTS in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1")

    expected_output = _show_tables_snowflake_sample_data_tpch_sf1_output()
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Objects test failed"


def test_show_objects_terse_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that show objects works in JIT. This is needed because we need
    to ensure the type information is correct."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "SHOW TERSE OBJECTS IN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    expected_output = _show_tables_snowflake_sample_data_tpch_sf1_output()
    bodo_output = impl(bc, query)
    passed = _test_equal_guard(
        bodo_output,
        expected_output,
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"


def test_show_objects(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show objects works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW OBJECTS in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1")

    expected_output = _show_objects_snowflake_sample_data_tpch_sf1_output()
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Objects test failed"


def _show_schemas_snowflake_sample_data_output(terse=True):
    if terse:
        return pd.DataFrame(
            {
                "CREATED_ON": [
                    "2024-05-14 20:12:27.165 -0700",
                    "2021-11-11 13:17:31.657 -0800",
                    "2023-09-05 18:34:28.256 -0700",
                    "2021-11-11 13:17:31.816 -0800",
                    "2021-11-11 13:17:31.807 -0800",
                    "2021-11-11 13:17:31.656 -0800",
                    "2021-11-11 13:17:31.793 -0800",
                ],
                "NAME": [
                    "INFORMATION_SCHEMA",
                    "TPCDS_SF100TCL",
                    "TPCDS_SF10TCL",
                    "TPCH_SF1",
                    "TPCH_SF10",
                    "TPCH_SF100",
                    "TPCH_SF1000",
                ],
                "SCHEMA_NAME": [
                    "SNOWFLAKE_SAMPLE_DATA",
                ]
                * 7,
                "KIND": pd.array([None] * 7, dtype=pd.ArrowDtype(pa.string())),
            }
        )
    else:
        return pd.DataFrame(
            {
                "CREATED_ON": [
                    "2024-06-24 12:36:45.585 -0700",
                    "2021-11-11 13:17:31.657 -0800",
                    "2023-09-05 18:34:28.256 -0700",
                    "2021-11-11 13:17:31.816 -0800",
                    "2021-11-11 13:17:31.807 -0800",
                    "2021-11-11 13:17:31.656 -0800",
                    "2021-11-11 13:17:31.793 -0800",
                ],
                "NAME": [
                    "INFORMATION_SCHEMA",
                    "TPCDS_SF100TCL",
                    "TPCDS_SF10TCL",
                    "TPCH_SF1",
                    "TPCH_SF10",
                    "TPCH_SF100",
                    "TPCH_SF1000",
                ],
                "IS_DEFAULT": ["N"] * 7,
                "IS_CURRENT": ["N"] * 7,
                "DATABASE_NAME": ["SNOWFLAKE_SAMPLE_DATA"] * 7,
                "OWNER": [""] * 7,
                "COMMENT": [
                    "Views describing the contents of schemas in this database",
                    "",
                    "",
                    "TPC-H scaling factor 1",
                    "TPC-H scaling factor 10",
                    "TPC-H scaling factor 100",
                    "TPC-H scaling factor 1000",
                ],
                "OPTIONS": [""] * 7,
                "RETENTION_TIME": ["1"] * 7,
                "OWNER_ROLE_TYPE": [""] * 7,
            }
        )


def test_show_schemas_terse(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show schemas terse works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW TERSE SCHEMAS in SNOWFLAKE_SAMPLE_DATA")
    expected_output = _show_schemas_snowflake_sample_data_output()
    # excluding INFORMATION_SCHEMA
    # INFORMATION_SCHEMA: created_on changes with each run
    passed = _test_equal_guard(bodo_output.drop([0]), expected_output.drop([0]))
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show schemas test failed"


def test_show_schemas_terse_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that show schemas terse works in JIT. This is needed because we need
    to ensure the type information is correct."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "SHOW TERSE SCHEMAS IN SNOWFLAKE_SAMPLE_DATA"

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    expected_output = _show_schemas_snowflake_sample_data_output()
    bodo_output = impl(bc, query)
    # excluding INFORMATION_SCHEMA
    # INFORMATION_SCHEMA: created_on changes with each run
    passed = _test_equal_guard(
        bodo_output.drop(0),
        expected_output.drop(0),
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"


def test_show_schemas(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show schemas works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW SCHEMAS in SNOWFLAKE_SAMPLE_DATA")
    expected_output = _show_schemas_snowflake_sample_data_output(terse=False)
    # excluding INFORMATION_SCHEMA
    # INFORMATION_SCHEMA: created_on changes with each run
    passed = _test_equal_guard(bodo_output.drop([0]), expected_output.drop([0]))
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show schemas test failed"


def check_view_exists(conn_str, view_name) -> bool:
    tables = pd.read_sql(
        f"SHOW VIEWS LIKE '{view_name}' STARTS WITH '{view_name}'",
        conn_str,
    )
    return len(tables) == 1


@contextmanager
def view_helper(conn_str, view_name, create=True):
    if create:
        if bodo.get_rank() == 0:
            pd.read_sql(f"CREATE VIEW {view_name} AS SELECT 0 AS A", conn_str)
            assert check_view_exists(conn_str, view_name)
        bodo.barrier()

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn_str)
            assert not check_view_exists(conn_str, view_name)


def test_show_tables_terse(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show tables works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW TERSE TABLES in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1")

    expected_output = _show_tables_snowflake_sample_data_tpch_sf1_output()
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Tables test failed"


def test_show_tables_terse_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that show tables works in JIT. This is needed because we need
    to ensure the type information is correct."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "SHOW TERSE TABLES IN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    expected_output = _show_tables_snowflake_sample_data_tpch_sf1_output()
    bodo_output = impl(bc, query)
    passed = _test_equal_guard(
        bodo_output,
        expected_output,
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"


def test_show_tables(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show tables works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("SHOW TABLES in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1")
    expected_output = _show_tables_snowflake_sample_data_tpch_sf1_output(terse=False)
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Tables test failed"


def _show_views_snowflake_sample_data_output(terse: bool = True) -> pd.DataFrame:
    """Fetch the SHOW VIEWS output directly from Snowflake sample DB."""
    maybe_terse = " TERSE " if terse else ""

    conn = snowflake.connector.connect(
        user=os.environ["SF_USERNAME"],
        password=os.environ["SF_PASSWORD"],
        account="bodopartner.us-east-1",
        warehouse="DEMO_WH",
    )
    try:
        cur = conn.cursor()
        cur.execute(
            f"SHOW {maybe_terse} VIEWS in SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA"
        )
        rows = cur.fetchall()
        colnames = [desc[0].upper() for desc in cur.description]
        df = pd.DataFrame(rows, columns=colnames)

        # Fix expected output to match format of bodo
        df = df.drop(columns=["DATABASE_NAME"])
        df["CREATED_ON"] = ["1969-12-31 16:00:00.000 -0800"] * len(df)
        df["SCHEMA_NAME"] = ["SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA"] * len(df)

        return df
    finally:
        cur.close()
        conn.close()


def test_show_views_terse(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show views works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    # Using Snowflake sample data to test to avoid overhead of creating a view
    bodo_output = bc.execute_ddl(
        "SHOW TERSE VIEWS in SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA"
    )
    expected_output = _show_views_snowflake_sample_data_output()
    # Columns might be in a different order:
    expected_output = expected_output[list(bodo_output.columns)]
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show views test failed"


def test_show_views_terse_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that show views works in JIT. This is needed because we need
    to ensure the type information is correct."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "SHOW TERSE VIEWS IN SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA"

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    bodo_output = impl(bc, query)
    expected_output = _show_views_snowflake_sample_data_output()
    # Columns might be in a different order:
    expected_output = expected_output[list(bodo_output.columns)]
    passed = _test_equal_guard(
        bodo_output,
        expected_output,
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"


def test_show_views(test_db_snowflake_catalog, memory_leak_check):
    """Tests that show views works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    # Using Snowflake sample data to test to avoid overhead of creating a view
    bodo_output = bc.execute_ddl(
        "SHOW VIEWS in SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA"
    )

    expected_output = _show_views_snowflake_sample_data_output(terse=False)
    # Columns might be in a different order:
    expected_output = expected_output[list(bodo_output.columns)]
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show views test failed"


def test_create_view(test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo can create a view in Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully created."]})

    def verify_view_created():
        assert check_view_exists(conn, view_name)
        x = pd.read_sql(f"SELECT * FROM {view_name}", conn)
        assert "a" in x
        assert x["a"].shape == (1,)
        assert x["a"][0] == "testview"

    # execute_ddl Version
    with view_helper(conn, view_name, create=False):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Python Version
    with view_helper(conn, view_name, create=False):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Jit Version
    # Intentionally returns replicated output
    with view_helper(conn, view_name, create=False):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=False,
        )
        verify_view_created()


def test_create_view_validates(test_db_snowflake_catalog, memory_leak_check):
    """Tests that Bodo validates view definitions before submitting to Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)

    schema_1 = gen_unique_id("SCEHMA1").upper()
    schema_2 = gen_unique_id("SCEHMA2").upper()

    # this test creates two schemas - schema_1 and schema_2, and creates a table
    # at SCHEMA_1.TABLE1. We then ensure that view validation will fail if we
    # create a view referencing TABLE1 in SCHEMA_2, and that the column of
    # TABLE1 are validated when we create a view in SCHEMA_1.

    catalog = deepcopy(test_db_snowflake_catalog)
    with schema_helper(conn, schema_1, create=True):
        catalog.connection_params["schema"] = schema_1
        bc = bodosql.BodoSQLContext(catalog=catalog)
        table_query = f"CREATE OR REPLACE TABLE {schema_1}.TABLE1 AS SELECT 0 as A"
        bc.sql(table_query)
        with schema_helper(conn, schema_2, create=True):
            with pytest.raises(ValueError, match="Object 'TABLE1' not found"):
                query = f"CREATE OR REPLACE VIEW {schema_2}.VIEW2 AS SELECT A + 1 as A from TABLE1"
                bc.execute_ddl(query)

        # Test that the view validates if ran in the correct schema
        py_output = pd.DataFrame({"STATUS": ["View 'VIEW2' successfully created."]})
        query = (
            f"CREATE OR REPLACE VIEW {schema_1}.VIEW2 AS SELECT A + 1 as A from TABLE1"
        )
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)

        # Column B does not exist - validation should fail
        with pytest.raises(ValueError, match="Column 'B' not found"):
            query = f"CREATE OR REPLACE VIEW {schema_1}.VIEW3 AS SELECT B + 1 as B from TABLE1"
            bc.execute_ddl(query)


@contextmanager
def view_helper_nontrivialview(conn_str, view_name, create=True):
    if create:
        if bodo.get_rank() == 0:
            pd.read_sql(
                f"CREATE VIEW {view_name} AS (SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM)",
                conn_str,
            )
            assert check_view_exists(conn_str, view_name)
        bodo.barrier()

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn_str)
            assert not check_view_exists(conn_str, view_name)


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_view(describe_keyword, test_db_snowflake_catalog, memory_leak_check):
    """Tests that describe view works on a proper Snowflake view."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query_describe_view = f"{describe_keyword} VIEW {view_name}"
    expected_output = pd.DataFrame(
        {
            "NAME": [
                "L_ORDERKEY",
                "L_PARTKEY",
                "L_SUPPKEY",
                "L_LINENUMBER",
                "L_QUANTITY",
                "L_EXTENDEDPRICE",
                "L_DISCOUNT",
                "L_TAX",
                "L_RETURNFLAG",
                "L_LINESTATUS",
                "L_SHIPDATE",
                "L_COMMITDATE",
                "L_RECEIPTDATE",
                "L_SHIPINSTRUCT",
                "L_SHIPMODE",
                "L_COMMENT",
            ],
            "TYPE": [
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "VARCHAR(1)",
                "VARCHAR(1)",
                "DATE",
                "DATE",
                "DATE",
                "VARCHAR(25)",
                "VARCHAR(10)",
                "VARCHAR(44)",
            ],
            "KIND": ["COLUMN"] * 16,
            "NULL?": ["N"] * 16,
            "DEFAULT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIMARY_KEY": ["N"] * 16,
            "UNIQUE_KEY": ["N"] * 16,
            "CHECK": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "EXPRESSION": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "COMMENT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "POLICY NAME": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIVACY DOMAIN": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
        }
    )
    with view_helper_nontrivialview(conn, view_name, create=True):
        # Python Version
        bodo_output = bc.sql(query_describe_view)
        passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Describe table test failed"

        # execute_ddl Version
        bodo_output = bc.execute_ddl(query_describe_view)
        passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Describe table test failed"

        # Jit Version
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_describe_view),
            py_output=expected_output,
            test_str_literal=False,
        )


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_view_on_table(
    describe_keyword, test_db_snowflake_catalog, memory_leak_check
):
    """Tests that describe view also works on a Snowflake table."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query_describe_view = (
        f"{describe_keyword} VIEW SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM"
    )
    expected_output = pd.DataFrame(
        {
            "NAME": [
                "L_ORDERKEY",
                "L_PARTKEY",
                "L_SUPPKEY",
                "L_LINENUMBER",
                "L_QUANTITY",
                "L_EXTENDEDPRICE",
                "L_DISCOUNT",
                "L_TAX",
                "L_RETURNFLAG",
                "L_LINESTATUS",
                "L_SHIPDATE",
                "L_COMMITDATE",
                "L_RECEIPTDATE",
                "L_SHIPINSTRUCT",
                "L_SHIPMODE",
                "L_COMMENT",
            ],
            "TYPE": [
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "DOUBLE",
                "VARCHAR(1)",
                "VARCHAR(1)",
                "DATE",
                "DATE",
                "DATE",
                "VARCHAR(25)",
                "VARCHAR(10)",
                "VARCHAR(44)",
            ],
            "KIND": ["COLUMN"] * 16,
            "NULL?": ["N"] * 16,
            "DEFAULT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIMARY_KEY": ["N"] * 16,
            "UNIQUE_KEY": ["N"] * 16,
            "CHECK": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "EXPRESSION": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "COMMENT": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "POLICY NAME": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
            "PRIVACY DOMAIN": pd.array([None] * 16, dtype=pd.ArrowDtype(pa.string())),
        }
    )

    # Python Version
    bodo_output = bc.sql(query_describe_view)
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Describe table test failed"

    # execute_ddl Version
    bodo_output = bc.execute_ddl(query_describe_view)
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Describe table test failed"

    # Jit Version
    check_func_seq(
        lambda bc, query: bc.sql(query),
        (bc, query_describe_view),
        py_output=expected_output,
        test_str_literal=False,
    )


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_view_error_does_not_exist(
    describe_keyword, test_db_snowflake_catalog, memory_leak_check
):
    """Tests that describe view raise error if the view does not exist."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query_describe_view = f"{describe_keyword} VIEW {view_name}"
    with view_helper_nontrivialview(conn, view_name, create=False):
        # execute_ddl Version
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to describe.",
        ):
            bc.execute_ddl(query_describe_view)

        # Python Version
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to describe.",
        ):
            bc.sql(query_describe_view)

        # Jit Version
        # Intentionally returns replicated output
        with pytest.raises(
            ValueError,
            match=f"View '{view_name}' does not exist or not authorized to describe.",
        ):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_describe_view),
                py_output="",
                test_str_literal=False,
            )


def test_describe_schema(test_db_snowflake_catalog, memory_leak_check):
    """Tests that describe schema works on Snowflake."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bodo_output = bc.execute_ddl("DESCRIBE SCHEMA TEST_DB.DDL_READ_TEST")
    expected_output = pd.DataFrame(
        {
            "CREATED_ON": [
                "2024-06-26 09:17:47.175 -0700",
                "2024-06-26 09:18:23.413 -0700",
            ],
            "NAME": ["TEST_TABLE_1", "TEST_VIEW_1"],
            "KIND": ["TABLE", "VIEW"],
        }
    )
    passed = _test_equal_guard(bodo_output, expected_output, sort_output=True)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "DESCRIBE SCHEMA test failed"
