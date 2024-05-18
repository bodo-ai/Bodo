# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests DDL operations on a BodoSQL catalog.
"""

from contextlib import contextmanager
from copy import deepcopy

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import (
    _test_equal_guard,
    check_func_seq,
    create_snowflake_table_from_select_query,
    get_snowflake_connection_string,
    pytest_snowflake,
    reduce_sum,
)
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (
    test_db_snowflake_catalog,  # noqa
)
from bodosql.tests.utils import gen_unique_id, test_equal_par

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
        test_equal_par(bodo_output, py_output)
        assert check_schema_exists(conn, schema_name)

    # Python Version
    with schema_helper(conn, schema_name, create=False):
        bodo_output = bc.sql(query)
        test_equal_par(bodo_output, py_output)
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
        with pytest.raises(BodoError, match=f"Schema '{schema_name}' already exists."):
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
        test_equal_par(bodo_output, py_output)
        assert not check_schema_exists(conn, schema_name)

    # Python Version
    with schema_helper(conn, schema_name):
        bodo_output = bc.sql(query)
        test_equal_par(bodo_output, py_output)
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
        BodoError,
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


def test_drop_table(test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

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
        query = f"DROP TABLE {table_name}"
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


def test_drop_table_python(test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake using regular bc.sql
    from Python"""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Create a table
    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "drop_table_test", db, schema
    ) as table_name:
        case_insenstive_table_name = table_name.upper()
        # Drop Execute a drop table query.
        query = f"DROP TABLE {table_name}"
        py_output = pd.DataFrame(
            {"STATUS": [f"{case_insenstive_table_name} successfully dropped."]}
        )
        bodo_output = bc.sql(query)
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


def test_drop_table_execute_ddl(test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake using bc.execute_ddl
    from Python"""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Create a table
    table_query = "SELECT 1 as A"
    with create_snowflake_table_from_select_query(
        table_query, "drop_table_test", db, schema
    ) as table_name:
        case_insenstive_table_name = table_name.upper()
        # Drop Execute a drop table query.
        query = f"DROP TABLE {table_name}"
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


def test_drop_table_case_sensitive(test_db_snowflake_catalog, memory_leak_check):
    """Tests that we can drop a table from Snowflake that
    is case sensitive."""
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn = get_snowflake_connection_string(db, schema)
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

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
        query = f"DROP TABLE {table_name}"
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


def test_drop_table_not_found(test_db_snowflake_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Snowflake raises an error."""
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
    assert len(tables) == 0, "Table exists. Please choose a different table name."
    with pytest.raises(BodoError, match=""):
        query = f"DROP TABLE {table_name}"
        impl(bc, query)


def test_drop_table_not_found_if_exists(test_db_snowflake_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Snowflake doesn't raise an error
    with IF EXISTS."""
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
    assert len(tables) == 0, "Table exists. Please choose a different table name."
    query = f"DROP TABLE IF EXISTS {table_name}"
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
            "DEFAULT": [None] * 16,
            "PRIMARY_KEY": ["N"] * 16,
            "UNIQUE_KEY": ["N"] * 16,
        }
    )
    passed = _test_equal_guard(bodo_output, expected_output)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Describe table test failed"


def test_describe_table_compiles_jit(test_db_snowflake_catalog, memory_leak_check):
    """Verify that describe table compiles in JIT."""
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query = "DESCRIBE TABLE SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM"
    bc.validate_query_compiles(query)


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
            pd.read_sql(f"CREATE VIEW {view_name} AS SELECT 0", conn_str)
            assert check_view_exists(conn_str, view_name)
        bodo.barrier()

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            pd.read_sql(f"DROP VIEW IF EXISTS {view_name}", conn_str)
            assert not check_view_exists(conn_str, view_name)


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
        test_equal_par(bodo_output, py_output)
        verify_view_created()

    # Python Version
    with view_helper(conn, view_name, create=False):
        bodo_output = bc.sql(query)
        test_equal_par(bodo_output, py_output)
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
            with pytest.raises(BodoError, match=f"Object 'TABLE1' not found"):
                query = f"CREATE OR REPLACE VIEW {schema_2}.VIEW2 AS SELECT A + 1 as A from TABLE1"
                bc.execute_ddl(query)

        # Test that the view validates if ran in the correct schema
        py_output = pd.DataFrame({"STATUS": [f"View 'VIEW2' successfully created."]})
        query = (
            f"CREATE OR REPLACE VIEW {schema_1}.VIEW2 AS SELECT A + 1 as A from TABLE1"
        )
        bodo_output = bc.execute_ddl(query)
        test_equal_par(bodo_output, py_output)

        # Column B does not exist - validation should fail
        with pytest.raises(BodoError, match=f"Column 'B' not found"):
            query = f"CREATE OR REPLACE VIEW {schema_1}.VIEW3 AS SELECT B + 1 as B from TABLE1"
            bc.execute_ddl(query)
