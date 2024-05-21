# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests the general Iceberg DDL functionality with the file system catalog.
"""
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.conftest import iceberg_database  # noqa
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import (
    check_func_seq,
    gen_unique_table_id,
)
from bodo.utils.typing import BodoError
from bodosql.tests.utils import assert_equal_par

pytestmark = pytest.mark.iceberg


@pytest.fixture(scope="session")
def iceberg_filesystem_catalog(tmp_path_factory):
    """
    An Iceberg FileSystemCatalog.
    """
    path = None
    if bodo.get_rank() == 0:
        path = str(tmp_path_factory.mktemp("iceberg"))
    path = MPI.COMM_WORLD.bcast(path)
    assert Path(path).exists(), "Failed to create filesystem catalog across all ranks"
    return bodosql.FileSystemCatalog(path, "iceberg")


def gen_unique_id(name_prefix: str) -> str:
    name = None
    if bodo.get_rank() == 0:
        name = gen_unique_table_id(name_prefix)
    name = MPI.COMM_WORLD.bcast(name)
    return name


def create_simple_ddl_table(spark):
    """
    Creates a very basic table for testing DDL functionality.
    """
    comm = MPI.COMM_WORLD
    table_name = gen_unique_id("SIMPLY_DDL_TABLE")
    df = pd.DataFrame({"A": [1, 2, 3]})
    sql_schema = [
        ("A", "long", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    return table_name


@contextmanager
def ddl_schema(schema_path: Path, create=True):
    if create and bodo.get_rank() == 0:
        schema_path.mkdir(exist_ok=True)
        assert schema_path.exists(), "Failed to create schema"

    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            if schema_path.exists():
                schema_path.rmdir()
            assert not schema_path.exists(), "Failed to drop schema"


@pytest.mark.parametrize("if_not_exists", [True, False])
def test_create_schema(if_not_exists, iceberg_filesystem_catalog, memory_leak_check):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    if_exists_str = "IF NOT EXISTS" if if_not_exists else ""
    query = f"CREATE SCHEMA {if_exists_str} {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' successfully created."]}
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    # execute_ddl Version
    with ddl_schema(schema_path, create=False):
        bodo_output: pd.DataFrame = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        assert schema_path.exists()

    # Python Version
    with ddl_schema(schema_path, create=False):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        assert schema_path.exists()

    # Jit Version
    # Intentionally returns replicated output
    with ddl_schema(schema_path, create=False):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=True,
        )
        assert schema_path.exists()


def test_create_schema_already_exists(iceberg_filesystem_catalog, memory_leak_check):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    query = f"CREATE SCHEMA {schema_name}"
    py_out = pd.DataFrame({"STATUS": [f"Schema '{schema_name}' already exists."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    with ddl_schema(schema_path):
        with pytest.raises(BodoError, match=f"Schema '{schema_name}' already exists."):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query),
                py_output=py_out,
                test_str_literal=True,
            )
        assert schema_path.exists(), "Schema should still exist"


def test_create_schema_if_not_exists_already_exists(
    iceberg_filesystem_catalog, memory_leak_check
):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
    py_out = pd.DataFrame(
        {"STATUS": [f"'{schema_name}' already exists, statement succeeded."]}
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    with ddl_schema(schema_path):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_out,
            test_str_literal=True,
        )
        assert schema_path.exists(), "Schema should still exist"


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_schema(if_exists, iceberg_filesystem_catalog, memory_leak_check):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    if_exists_str = "IF EXISTS" if if_exists else ""
    query = f"DROP SCHEMA {if_exists_str} {schema_name}"
    py_output = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' successfully dropped."]}
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    # execute_ddl Version
    with ddl_schema(schema_path):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output)
        assert not schema_path.exists()

    # Python Version
    with ddl_schema(schema_path):
        bodo_output = bc.sql(query)
        assert_equal_par(bodo_output, py_output)
        assert not schema_path.exists()

    # Jit Version
    # Intentionally returns replicated output
    with ddl_schema(schema_path):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_output,
            test_str_literal=True,
        )
        assert not schema_path.exists()


def test_drop_schema_not_exists(iceberg_filesystem_catalog, memory_leak_check):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    query = f"DROP SCHEMA {schema_name}"
    py_out = pd.DataFrame({"STATUS": []})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    with pytest.raises(
        BodoError,
        match=f"Schema '{schema_name}' does not exist or drop cannot be performed.",
    ):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query),
            py_output=py_out,
            test_str_literal=True,
        )
    assert not schema_path.exists()


def test_drop_schema_if_exists_doesnt_exist(
    iceberg_filesystem_catalog, memory_leak_check
):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    query = f"DROP SCHEMA IF EXISTS {schema_name}"
    py_out = pd.DataFrame(
        {"STATUS": [f"Schema '{schema_name}' already dropped, statement succeeded."]}
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    check_func_seq(
        lambda bc, query: bc.sql(query),
        (bc, query),
        py_output=py_out,
        test_str_literal=True,
    )
    assert not schema_path.exists()


def test_drop_table(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    table_name = create_simple_ddl_table(spark)
    db_schema, _ = iceberg_database(table_name)
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'DROP TABLE "{db_schema}"."{table_name}"'
    py_output = pd.DataFrame({"STATUS": [f"{table_name} successfully dropped."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = impl(bc, query)
    assert_equal_par(bodo_output, py_output)

    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_iceberg_drop_table_python(iceberg_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table using
    bc.sql from Python.
    """
    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    table_name = create_simple_ddl_table(spark)
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'DROP TABLE "{db_schema}"."{table_name}"'
    py_output = pd.DataFrame({"STATUS": [f"{table_name} successfully dropped."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = bc.sql(query)
    assert_equal_par(bodo_output, py_output)

    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_iceberg_drop_table_execute_ddl(iceberg_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table using
    bc.execute_ddl from Python.
    """
    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    table_name = create_simple_ddl_table(spark)
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'DROP TABLE "{db_schema}"."{table_name}"'
    py_output = pd.DataFrame({"STATUS": [f"{table_name} successfully dropped."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = bc.execute_ddl(query)
    assert_equal_par(bodo_output, py_output)

    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_drop_table_not_found(iceberg_filesystem_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Iceberg raises an error."""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    # Create an unused table to ensure the database is created
    create_simple_ddl_table(spark)
    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert (
        len(existing_tables) == 0
    ), "Table Name already exists. Please choose a different table name."
    with pytest.raises(BodoError, match=""):
        query = f"DROP TABLE {table_name}"
        bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
        impl(bc, query)


def test_drop_table_not_found_if_exists(iceberg_filesystem_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Iceberg doesn't raise an error
    with IF EXISTS."""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    # Create an unused table to ensure the database is created
    create_simple_ddl_table(spark)
    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert (
        len(existing_tables) == 0
    ), "Table Name already exists. Please choose a different table name."
    query = f"DROP TABLE IF EXISTS {table_name}"
    py_output = pd.DataFrame(
        {
            "STATUS": [
                f"Drop statement executed successfully ({table_name} already dropped)."
            ]
        }
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = impl(bc, query)
    assert_equal_par(bodo_output, py_output)


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_table(
    describe_keyword, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog can describe an iceberg
    table.
    """
    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    table_name = create_simple_ddl_table(spark)
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'{describe_keyword} TABLE "{db_schema}"."{table_name}"'
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = bc.execute_ddl(query)
    expected_output = pd.DataFrame(
        {
            "NAME": ["A"],
            "TYPE": ["BIGINT"],
            "KIND": ["COLUMN"],
            "NULL?": ["Y"],
            "DEFAULT": [None],
            "PRIMARY_KEY": ["N"],
            "UNIQUE_KEY": ["N"],
        }
    )
    assert_equal_par(bodo_output, expected_output)


def test_describe_table_compiles_jit(iceberg_filesystem_catalog, memory_leak_check):
    """
    Verify that describe table compiles in JIT.
    """
    query = f"DESCRIBE TABLE ANY_TABLE"
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bc.validate_query_compiles(query)


def test_show_objects(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can shows objects in iceberg
    schema.
    """
    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
    db_schema = "iceberg_db"
    # Create 2 tables
    table_name1 = create_simple_ddl_table(spark)
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name1}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"
    table_name2 = create_simple_ddl_table(spark)
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name2}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'SHOW TERSE OBJECTS IN "{db_schema}"'
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bodo_output = bc.execute_ddl(query)
    expected_output = pd.DataFrame(
        {
            "CREATED_ON": [None, None],
            "NAME": [table_name2, table_name1],
            "KIND": ["TABLE"] * 2,
            "SCHEMA_NAME": [db_schema] * 2,
        }
    )
    passed = _test_equal_guard(
        bodo_output, expected_output, sort_output=True, reset_index=True
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Objects test failed"


def test_show_objects_compiles_jit(iceberg_filesystem_catalog, memory_leak_check):
    """
    Verify that show objects compiles in JIT.
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    query = f'CREATE SCHEMA "{schema_name}"'
    bc.execute_ddl(query)
    query = f"SHOW TERSE OBJECTS IN '{schema_name}'"
    bc.validate_query_compiles(query)


def test_show_schemas(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can shows schemas in iceberg
    schema.
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    db_name = gen_unique_id("ICEBERG_DB").upper()
    query = f'CREATE SCHEMA "{db_name}"'
    bc.execute_ddl(query)
    schema_name1 = gen_unique_id("TEST_SCHEMA_DDL").upper()
    query = f'CREATE SCHEMA "{db_name}"."{schema_name1}"'
    bc.execute_ddl(query)
    schema_name2 = gen_unique_id("TEST_SCHEMA_DDL").upper()
    query = f'CREATE SCHEMA "{db_name}"."{schema_name2}"'
    bc.execute_ddl(query)

    query = f'SHOW TERSE SCHEMAS IN "{db_name}"'
    bodo_output = bc.execute_ddl(query)
    expected_output = pd.DataFrame(
        {
            "CREATED_ON": [None, None],
            "NAME": [schema_name1, schema_name2],
            "KIND": [None] * 2,
            "SCHEMA_NAME": [f"{db_name}.{schema_name1}", f"{db_name}.{schema_name2}"],
        }
    )
    passed = _test_equal_guard(
        bodo_output, expected_output, sort_output=True, reset_index=True
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Show Objects test failed"
    query = f'DROP SCHEMA "{db_name}"."{schema_name1}"'
    bc.execute_ddl(query)
    query = f'DROP SCHEMA "{db_name}"."{schema_name2}"'
    bc.execute_ddl(query)
    query = f'DROP SCHEMA "{db_name}"'
    bc.execute_ddl(query)


def test_show_schemas_compiles_jit(iceberg_filesystem_catalog, memory_leak_check):
    """
    Verify that show schemas compiles in JIT.
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    query = f'CREATE SCHEMA "{schema_name}"'
    bc.execute_ddl(query)
    query = f"SHOW TERSE SCHEMAS IN '{schema_name}'"
    bc.validate_query_compiles(query)
    query = f'DROP SCHEMA "{schema_name}"'
    bc.execute_ddl(query)
