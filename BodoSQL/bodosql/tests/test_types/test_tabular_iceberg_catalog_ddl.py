# Note:
# This file is deprecated, and all tests in this file
# are in the process of being ported over to the
# test_iceberg_ddl.py file, which uses the new DDLTestHarness system.
# New tests should be added to test_iceberg_ddl.py instead of this file,
# in order to be consistent with the new harness system.


from contextlib import contextmanager

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.iceberg_database_helpers.utils import get_spark_tabular
from bodo.tests.utils import (
    _test_equal_guard,
    check_func_seq,
    pytest_mark_one_rank,
    pytest_tabular,
    reduce_sum,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import run_rank0
from bodosql.tests.utils import assert_equal_par, gen_unique_id

pytestmark = pytest_tabular


def check_view_exists(tabular_connection, view_name) -> bool:
    # These should only be run from rank 0, but we can't mark them as
    # run_rank0 since they are used in other run_rank0 functions
    assert bodo.get_rank() == 0
    spark = get_spark_tabular(tabular_connection)
    tables = spark.sql(f"SHOW VIEWS LIKE '{view_name}'").toPandas()
    return len(tables) == 1


def check_table_exists(tabular_connection, table_name) -> bool:
    # These should only be run from rank 0, but we can't mark them as
    # run_rank0 since they are used in other run_rank0 functions
    assert bodo.get_rank() == 0
    spark = get_spark_tabular(tabular_connection)
    tables = spark.sql(f"SHOW TABLES LIKE '{table_name}'").toPandas()
    return len(tables) == 1


@contextmanager
def view_helper(tabular_connection, view_name, create=True):
    if create:

        @run_rank0
        def create_view():
            get_spark_tabular(tabular_connection).sql(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT 0 AS A"
            )
            assert check_view_exists(tabular_connection, view_name)

        create_view()
    try:
        yield
    finally:
        bodo.barrier()

        @run_rank0
        def destroy_view():
            get_spark_tabular(tabular_connection).sql(
                f"DROP VIEW IF EXISTS {view_name}"
            )
            assert not check_view_exists(tabular_connection, view_name)

        destroy_view()


def test_create_view(tabular_catalog, tabular_connection, memory_leak_check):
    """Tests that Bodo can create a view using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully created."]})

    def verify_view_created():
        x = bc.sql(f"SELECT * FROM {view_name}")
        x = bodo.gatherv(x)
        if bodo.get_rank() == 0:
            assert "A" in x
            assert len(x["A"]) == 1
            assert x["A"][0] == "testview"

    # execute_ddl Version
    with view_helper(tabular_connection, view_name, create=False):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Python Version
    with view_helper(tabular_connection, view_name, create=False):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        verify_view_created()

    # Jit Version
    # Intentionally returns replicated output
    with view_helper(tabular_connection, view_name, create=False):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=False,
        )
        verify_view_created()


def check_schema_exists(tabular_connection, schema_name) -> bool:
    tables = (
        get_spark_tabular(tabular_connection)
        .sql(f"SHOW SCHEMAS LIKE '{schema_name}'")
        .toPandas()
    )
    return len(tables) == 1


@contextmanager
def schema_helper(tabular_connection, schema_name, create=True):
    if create:

        @run_rank0
        def create_schema():
            get_spark_tabular(tabular_connection).sql(f"CREATE SCHEMA {schema_name}")
            assert check_schema_exists(tabular_connection, schema_name)

        create_schema()

    try:
        yield
    finally:
        bodo.barrier()

        @run_rank0
        def destroy_schema():
            get_spark_tabular(tabular_connection).sql(
                f"DROP SCHEMA IF EXISTS {schema_name}"
            )
            assert not check_schema_exists(tabular_connection, schema_name)

        destroy_schema()


# This test is marked one rank because it relies on executing a CREATE TABLE
# query which currently doesn't work on multiple ranks.
@pytest_mark_one_rank
def test_create_view_validates(tabular_catalog, tabular_connection, memory_leak_check):
    """Tests that Bodo validates view definitions before submitting to Snowflake."""
    schema_1 = gen_unique_id("SCHEMA1").upper()
    schema_2 = gen_unique_id("SCHEMA2").upper()

    # this test creates two schemas - schema_1 and schema_2, and creates a table
    # at SCHEMA_1.TABLE1. We then ensure that view validation will fail if we
    # create a view referencing TABLE1 in SCHEMA_2, and that the column of
    # TABLE1 are validated when we create a view in SCHEMA_1.

    with schema_helper(tabular_connection, schema_1, create=True):
        try:
            # catalog.connection_params["schema"] = schema_1
            bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
            table_query = f"CREATE OR REPLACE TABLE {schema_1}.TABLE1 AS SELECT 0 as A"
            bc.sql(table_query)
            with schema_helper(tabular_connection, schema_2, create=True):
                with pytest.raises(BodoError, match=f"Object 'TABLE1' not found"):
                    query = f"CREATE OR REPLACE VIEW {schema_2}.VIEW2 AS SELECT A + 1 as A from TABLE1"
                    bc.execute_ddl(query)

            # Test that the view validates if ran in the correct schema
            py_output = pd.DataFrame(
                {"STATUS": [f"View 'VIEW2' successfully created."]}
            )
            query = f"CREATE OR REPLACE VIEW {schema_1}.VIEW2 AS SELECT A + 1 as A from TABLE1"
            bodo_output = bc.execute_ddl(query)
            assert_equal_par(bodo_output, py_output)

            # Column B does not exist - validation should fail
            with pytest.raises(BodoError, match=f"Column 'B' not found"):
                query = f"CREATE OR REPLACE VIEW {schema_1}.VIEW3 AS SELECT B + 1 as B from TABLE1"
                bc.execute_ddl(query)
        finally:
            # drop created table and view so that the schema can be dropped
            @run_rank0
            def cleanup():
                spark = get_spark_tabular(tabular_connection)
                spark.sql(f"DROP TABLE IF EXISTS {schema_1}.TABLE1")
                spark.sql(f"DROP VIEW IF EXISTS {schema_1}.VIEW2")
                spark.sql(f"DROP VIEW IF EXISTS {schema_1}.VIEW3")

            cleanup()


@contextmanager
def view_helper_nontrivialview(bc, tabular_connection, view_name, create=True):
    """If create = True, create a non-trivial view with actual data that will get destroyed automatically aftre this function call"""
    schema = "CI"
    if create:
        view_query = f"CREATE OR REPLACE VIEW {view_name} AS (SELECT * FROM {schema}.BODOSQL_ICEBERG_READ_TEST)"
        bc.sql(view_query)
    try:
        yield
    finally:
        bodo.barrier()

        @run_rank0
        def destroy_view():
            get_spark_tabular(tabular_connection).sql(
                f"DROP VIEW IF EXISTS {view_name}"
            )
            assert not check_view_exists(tabular_connection, view_name)

        destroy_view()


@pytest.mark.parametrize("purge", [True, False])
def test_iceberg_drop_table_purge_sql(
    purge, tabular_catalog, tabular_connection, memory_leak_check
):
    """
    DROP TABLE PURGE in Tabular DO NOT delete underlying files instantly.
    Currently, this test case doesn't check anything: only making sure no errors when executing the command
    """
    purge_str = "PURGE" if purge else ""
    table_name = gen_unique_id("TEST_TABLE").upper()
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)

    query_create_table = f"CREATE OR REPLACE TABLE {table_name} AS (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST)"
    bc.sql(query_create_table)

    query_drop_table = f"DROP TABLE IF EXISTS {table_name} {purge_str}"

    bc.sql(query_drop_table)


@pytest.mark.parametrize("purge", [True, False])
def test_iceberg_drop_table_purge(
    purge, tabular_catalog, tabular_connection, memory_leak_check
):
    """
    DROP TABLE PURGE in Tabular DO NOT delete underlying files instantly.
    Currently, this test case doesn't check anything: only making sure no errors when executing the command
    """
    purge_str = "PURGE" if purge else ""
    table_name = gen_unique_id("TEST_TABLE").upper()
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)

    query_create_table = f"CREATE OR REPLACE TABLE {table_name} AS (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST)"
    bc.sql(query_create_table)

    query_drop_table = f"DROP TABLE IF EXISTS {table_name} {purge_str}"

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    impl(bc, query_drop_table)


def _test_equal_par(bodo_output, py_output):
    passed = _test_equal_guard(
        bodo_output,
        py_output,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Parallel test failed"


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_iceberg_describe_view_basic(
    describe_keyword, tabular_catalog, tabular_connection, memory_leak_check
):
    """
    Tests that the tabular catalog correctly describes the view
    """
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query_describe_view = f"{describe_keyword} VIEW {view_name}"

    expected_output = pd.DataFrame(
        {
            "NAME": ["A", "B", "C"],
            "TYPE": ["VARCHAR", "DOUBLE", "BOOLEAN"],
            "KIND": ["COLUMN", "COLUMN", "COLUMN"],
            "NULL?": ["N", "N", "N"],
            "DEFAULT": [None, None, None],
            "PRIMARY_KEY": ["N", "N", "N"],
            "UNIQUE_KEY": ["N", "N", "N"],
            "CHECK": [None, None, None],
            "EXPRESSION": [None, None, None],
            "COMMENT": [None, None, None],
            "POLICY NAME": [None, None, None],
            "PRIVACY DOMAIN": [None, None, None],
        }
    )
    with view_helper_nontrivialview(bc, tabular_connection, view_name, create=True):
        # execute_ddl Version
        output = bc.execute_ddl(query_describe_view)
        _test_equal_par(output, expected_output)

        # Python Version
        output = bc.sql(query_describe_view)
        _test_equal_par(output, expected_output)

        # Jit Version
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_describe_view),
            py_output=expected_output,
            test_str_literal=False,
        )


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_iceberg_describe_view_error_does_not_exist(
    describe_keyword, tabular_catalog, memory_leak_check
):
    """
    Tests that the tabular catalog raises an error when describing an iceberg
    view that does not exist.
    """
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query_describe_view = f"{describe_keyword} VIEW {view_name}"
    with pytest.raises(
        BodoError,
        match=f"View '{view_name}' does not exist or not authorized to describe.",
    ):
        bc.execute_ddl(query_describe_view)

    # Python Version
    with pytest.raises(
        BodoError,
        match=f"View '{view_name}' does not exist or not authorized to describe.",
    ):
        bc.sql(query_describe_view)

    # Jit Version
    # Intentionally returns replicated output
    with pytest.raises(
        BodoError,
        match=f"View '{view_name}' does not exist or not authorized to describe.",
    ):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_describe_view),
            py_output="",
            test_str_literal=False,
        )


##################
#   ALTER tests  #
##################

# TODO: There are problems with bc.sql CREATE TABLE when run on more than 2 ranks.
# This is a known issue. For now, we will run these tests on one rank only.


# Helper functions
def verify_table_created(table_name, tabular_connection, bc):
    assert run_rank0(check_table_exists)(tabular_connection, table_name)
    x = bc.sql(f"SELECT * FROM {table_name}")
    x = bodo.allgatherv(x)
    assert "A" in x
    assert x["A"].shape == (1,)
    assert x["A"][0] == "testtable"


def create_test_table(table_name, bc):
    bc.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT 'testtable' as A")


@run_rank0
def drop_test_table(table_name, tabular_connection):
    # drop created table
    get_spark_tabular(tabular_connection).sql(f"DROP TABLE IF EXISTS {table_name}")
    get_spark_tabular(tabular_connection).sql(
        f"DROP TABLE IF EXISTS {table_name}_renamed"
    )


# Verify view was created.
def verify_view_created(view_name, tabular_connection, bc):
    assert run_rank0(check_view_exists)(tabular_connection, view_name)
    x = bc.sql(f"SELECT * FROM {view_name}")
    x = bodo.allgatherv(x)
    assert "A" in x
    assert x["A"].shape == (1,)
    assert x["A"][0] == "testview"


def create_test_view(view_name, bc):
    bc.sql(f"CREATE OR REPLACE VIEW {view_name} AS SELECT 'testview' as A")


@run_rank0
def drop_test_view(view_name, tabular_connection):
    # drop created view
    get_spark_tabular(tabular_connection).sql(f"DROP VIEW IF EXISTS {view_name}")
    get_spark_tabular(tabular_connection).sql(
        f"DROP VIEW IF EXISTS {view_name}_renamed"
    )


@pytest_mark_one_rank
def test_alter_view(tabular_catalog, tabular_connection, memory_leak_check):
    """Tests that Bodo can rename a view via ALTER VIEW using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    try:
        create_test_view(view_name, bc)
        verify_view_created(view_name, tabular_connection, bc)

        # Alter query
        query = f"ALTER VIEW {view_name} RENAME TO {view_name}_renamed"
        py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)

        # Verify renamed view exists
        verify_view_created(f"{view_name}_renamed", tabular_connection, bc)

    finally:
        # drop created views
        drop_test_view(view_name, tabular_connection)


@pytest_mark_one_rank
def test_alter_view_rename_ifexists(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo can rename a view via ALTER view IF EXISTS using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    try:
        # Create the test view
        create_test_view(view_name, bc)
        verify_view_created(view_name, tabular_connection, bc)

        # Alter query
        query = f"ALTER VIEW IF EXISTS {view_name} RENAME TO {view_name}_renamed"
        py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)

        # Verify renamed view exists
        verify_view_created(f"{view_name}_renamed", tabular_connection, bc)

    finally:
        # drop created view
        drop_test_view(view_name, tabular_connection)


@pytest_mark_one_rank
def test_alter_view_rename_not_found(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo throws an error when rename a non-existent view via ALTER view using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()

    # Verify view does not exist.
    assert not check_view_exists(tabular_connection, view_name)

    # Alter query
    query = f"ALTER VIEW {view_name} RENAME TO {view_name}_renamed"

    # This should throw an error, saying the view does not exist or not authorized.
    with pytest.raises(BodoError, match="does not exist or not authorized"):
        bodo_output = bc.execute_ddl(query)


@pytest_mark_one_rank
def test_alter_view_rename_ifexists_not_found(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo returns gracefully when renaming
    a non-existent view via ALTER view IF EXISTS using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()

    # Verify view does not exist.
    assert not check_view_exists(tabular_connection, view_name)

    # Alter query
    query = f"ALTER VIEW IF EXISTS {view_name} RENAME TO {view_name}_renamed"
    py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})
    bodo_output = bc.execute_ddl(query)
    assert_equal_par(bodo_output, py_output)


@pytest_mark_one_rank
def test_alter_view_rename_to_existing(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """
    Tests that Bodo throws an error when renaming a view to an existing view
    via ALTER view using a Tabular catalog.
    """
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    try:
        # Create the test view
        create_test_view(view_name, bc)
        create_test_view(f"{view_name}_B", bc)

        # Verify view was created.
        verify_view_created(view_name, tabular_connection, bc)
        verify_view_created(f"{view_name}_B", tabular_connection, bc)

        # Alter query
        query = f"ALTER VIEW IF EXISTS {view_name} RENAME TO {view_name}_B"

        # This should throw an error, saying the view does not exist or not authorized.
        with pytest.raises(BodoError, match="already exists"):
            bodo_output = bc.execute_ddl(query)

    finally:
        # drop created view
        drop_test_view(view_name, tabular_connection)
        drop_test_view(f"{view_name}_B", tabular_connection)


@pytest_mark_one_rank
def test_alter_table_on_view(tabular_catalog, tabular_connection, memory_leak_check):
    """Tests Bodo's behavior when attempting to rename a VIEW via ALTER TABLE
    using a Tabular catalog matches Snowflake's behavior. (This should SUCCEED)"""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    try:
        # Create the test view
        create_test_view(view_name, bc)

        verify_view_created(view_name, tabular_connection, bc)

        # Alter query
        query = f"ALTER TABLE {view_name} RENAME TO {view_name}_renamed"
        py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})

        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)

        # Verify renamed view exists
        verify_view_created(f"{view_name}_renamed", tabular_connection, bc)

    finally:
        # drop created view
        drop_test_view(view_name, tabular_connection)


@pytest_mark_one_rank
def test_alter_view_on_table(tabular_catalog, tabular_connection, memory_leak_check):
    """Tests Bodo's behavior when attempting to rename a TABLE via ALTER VIEW
    using a Tabular catalog matches Snowflake's behavior. (This should FAIL)"""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    table_name = gen_unique_id("TEST_TABLE").upper()
    try:
        # Create the test table
        create_test_table(table_name, bc)

        # Verify table was created.
        verify_table_created(table_name, tabular_connection, bc)

        # Alter VIEW query
        query = f"ALTER VIEW {table_name} RENAME TO {table_name}_renamed"
        py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})

        # This should throw an error, saying the view does not exist or not authorized.
        with pytest.raises(BodoError, match="View does not exist"):
            bodo_output = bc.execute_ddl(query)

    finally:
        # drop created table
        drop_test_table(table_name, tabular_connection)


@pytest_mark_one_rank
def test_alter_unsupported_commands(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo throws an error when running unsupported ALTER commands using a Tabular catalog."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    table_name = gen_unique_id("TEST_TABLE").upper()
    view_name = gen_unique_id("TEST_VIEW").upper()
    # Unsupported query
    query = f"ALTER VIEW {view_name} SET SECURE"

    # This should throw an error
    with pytest.raises(BodoError, match="Unable to parse"):
        bodo_output = bc.execute_ddl(query)

    # Unsupported query
    query = f"ALTER TABLE {table_name} SWAP WITH {table_name}_swap"

    # This should throw an error
    with pytest.raises(BodoError, match="currently unsupported"):
        bodo_output = bc.execute_ddl(query)

    # Unsupported query
    query = f"ALTER TABLE {table_name} CLUSTER BY junk_column"

    # This should throw an error
    with pytest.raises(BodoError, match="Unable to parse"):
        bodo_output = bc.execute_ddl(query)


def check_row_exists(output, row):
    """
    Helper function to check if a row exists in the output dataframe.
    """
    return ((output["key"] == row["key"]) & (output["value"] == row["value"])).any()


###################
# DROP VIEW Tests #
###################


@pytest.mark.parametrize("if_exists", [True, False])
def test_iceberg_drop_view(
    if_exists, tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo drops the view successfully when the view exists."""
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP VIEW {if_exists_str} {view_name}"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully dropped."]})

    # execute_ddl Version
    with view_helper(tabular_connection, view_name, create=True):
        bodo_output = bc.execute_ddl(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
        assert not run_rank0(check_view_exists)(tabular_connection, view_name)

    # Python Version
    with view_helper(tabular_connection, view_name, create=True):
        bodo_output = bc.sql(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
        assert not run_rank0(check_view_exists)(tabular_connection, view_name)

    # Jit Version
    # Intentionally returns replicated output
    with view_helper(tabular_connection, view_name, create=True):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_drop_view),
            py_output=py_output,
            test_str_literal=False,
        )
        assert not run_rank0(check_view_exists)(tabular_connection, view_name)


@pytest_mark_one_rank
@pytest.mark.parametrize("if_exists", [True, False])
def test_iceberg_drop_view_error_does_not_exist(
    if_exists, tabular_catalog, tabular_connection, memory_leak_check
):
    """Tests that Bodo drops the view successfully when the view exists and if_exists.
    Tests that Bodo raises error when the view exists and if_exists is False.
    """
    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f"DROP VIEW {if_exists_str} {view_name}"
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully dropped."]})

    # execute_ddl Version
    with view_helper(tabular_connection, view_name, create=False):
        if if_exists:
            bodo_output = bc.execute_ddl(query_drop_view)
            _test_equal_guard(bodo_output, py_output)
            assert not run_rank0(check_view_exists)(tabular_connection, view_name)
        else:
            with pytest.raises(
                BodoError,
                match=f"View '{view_name}' does not exist or not authorized to drop.",
            ):
                bc.execute_ddl(query_drop_view)

    # Python Version
    with view_helper(tabular_connection, view_name, create=False):
        if if_exists:
            bodo_output = bc.sql(query_drop_view)
            _test_equal_guard(bodo_output, py_output)
            assert not run_rank0(check_view_exists)(tabular_connection, view_name)
        else:
            with pytest.raises(
                BodoError,
                match=f"View '{view_name}' does not exist or not authorized to drop.",
            ):
                bc.sql(query_drop_view)

    # Jit Version
    # Intentionally returns replicated output
    with view_helper(tabular_connection, view_name, create=False):
        if if_exists:
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_drop_view),
                py_output=py_output,
                test_str_literal=False,
            )
        else:
            with pytest.raises(
                BodoError,
                match=f"View '{view_name}' does not exist or not authorized to drop.",
            ):
                check_func_seq(
                    lambda bc, query: bc.sql(query),
                    (bc, query_drop_view),
                    py_output="",
                    test_str_literal=False,
                )
