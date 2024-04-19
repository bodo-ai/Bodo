# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests DDL operations on a BodoSQL catalog.
"""


import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import (
    _test_equal_guard,
    create_snowflake_table_from_select_query,
    get_snowflake_connection_string,
    pytest_snowflake,
    reduce_sum,
)
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (
    test_db_snowflake_catalog,  # noqa
)

pytestmark = pytest_snowflake


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
