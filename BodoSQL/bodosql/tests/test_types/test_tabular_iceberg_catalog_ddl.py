from contextlib import contextmanager

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.iceberg_database_helpers.utils import get_spark_tabular
from bodo.tests.utils import check_func_seq, pytest_tabular
from bodo.utils.typing import BodoError
from bodo.utils.utils import run_rank0
from bodosql.tests.utils import assert_equal_par, gen_unique_id

pytestmark = pytest_tabular


def check_view_exists(tabular_connection, view_name) -> bool:
    spark = get_spark_tabular(tabular_connection)
    tables = spark.sql(f"SHOW VIEWS LIKE '{view_name}'").toPandas()
    return len(tables) == 1


@contextmanager
def view_helper(tabular_connection, view_name, create=True):
    if create:

        @run_rank0
        def create_view():
            get_spark_tabular(tabular_connection).sql(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT 0"
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
                spark.sql(f"DROP TABLE {schema_1}.TABLE1")
                spark.sql(f"DROP VIEW {schema_1}.VIEW2")

            cleanup()
