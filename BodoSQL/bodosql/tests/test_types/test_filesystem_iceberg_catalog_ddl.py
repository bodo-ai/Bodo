"""
Tests the general Iceberg DDL functionality with the file system catalog.

This file is deprecated, and all tests in this file
are in the process of being ported over to the
test_iceberg_ddl.py file, which uses the new DDLTestHarness system.
New tests should be added to test_iceberg_ddl.py instead of this file,
in order to be consistent with the new harness system.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.conftest import iceberg_database  # noqa
from bodo.tests.iceberg_database_helpers.utils import (
    SparkFilesystemIcebergCatalog,
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import (
    _test_equal_guard,
    check_func_seq,
    gen_unique_table_id,
    pytest_mark_one_rank,
)
from bodosql.tests.utils import assert_equal_par

pytestmark = pytest.mark.iceberg


@pytest.fixture(scope="session")
def iceberg_filesystem_catalog():
    """An Iceberg FileSystemCatalog."""
    return bodosql.FileSystemCatalog(os.path.curdir, "iceberg")


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


@pytest.mark.parametrize("if_exists", [True, False])
def test_drop_schema(if_exists, iceberg_filesystem_catalog, memory_leak_check):
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    schema_path = Path(iceberg_filesystem_catalog.connection_string) / schema_name

    if_exists_str = "IF EXISTS" if if_exists else ""
    query = f"DROP SCHEMA {if_exists_str} {schema_name}"
    py_output = pd.DataFrame(
        {
            "STATUS": pd.array(
                [f"Schema '{schema_name}' successfully dropped."],
                dtype=pd.StringDtype(),
            )
        }
    )
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    # execute_ddl Version
    with ddl_schema(schema_path):
        bodo_output = bc.execute_ddl(query)
        assert_equal_par(bodo_output, py_output, check_dtype=True)
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
            check_dtype=False,
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
        ValueError,
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


@pytest_mark_one_rank
def test_drop_table(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
    table_name = create_simple_ddl_table(spark)
    db_schema, _ = iceberg_database(table_name, spark=spark)
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


@pytest_mark_one_rank
def test_iceberg_drop_table_python(iceberg_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table using
    bc.sql from Python.
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
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


@pytest_mark_one_rank
def test_iceberg_drop_table_execute_ddl(iceberg_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table using
    bc.execute_ddl from Python.
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
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


@pytest_mark_one_rank
def test_drop_table_not_found(iceberg_filesystem_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Iceberg raises an error."""
    try:

        @bodo.jit
        def impl(bc, query):
            return bc.sql(query)

        spark = get_spark(
            SparkFilesystemIcebergCatalog(
                catalog_name="hadoop_prod",
                path=iceberg_filesystem_catalog.connection_string,
            )
        )
        # Create an unused table to ensure the database is created
        created_table = create_simple_ddl_table(spark)
        # Create a garbage table name.
        table_name = "FEJWIOPFE13_9029J03C32"
        db_schema = "iceberg_db"
        existing_tables = spark.sql(
            f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
        ).toPandas()
        assert len(existing_tables) == 0, (
            "Table Name already exists. Please choose a different table name."
        )
        with pytest.raises(ValueError, match=""):
            query = f"DROP TABLE {table_name}"
            bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
            impl(bc, query)
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{created_table}"
            spark.sql(query)

        cleanup()


@pytest_mark_one_rank
def test_drop_table_not_found_if_exists(iceberg_filesystem_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Iceberg doesn't raise an error
    with IF EXISTS."""
    try:

        @bodo.jit
        def impl(bc, query):
            return bc.sql(query)

        spark = get_spark(
            SparkFilesystemIcebergCatalog(
                catalog_name="hadoop_prod",
                path=iceberg_filesystem_catalog.connection_string,
            )
        )
        # Create an unused table to ensure the database is created
        created_table = create_simple_ddl_table(spark)
        # Create a garbage table name.
        table_name = "FEJWIOPFE13_9029J03C32"
        db_schema = "iceberg_db"
        existing_tables = spark.sql(
            f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
        ).toPandas()
        assert len(existing_tables) == 0, (
            "Table Name already exists. Please choose a different table name."
        )
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
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{created_table}"
            spark.sql(query)

        cleanup()


@pytest_mark_one_rank
@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_table(
    describe_keyword, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog can describe an iceberg
    table.
    """
    try:
        spark = get_spark(
            SparkFilesystemIcebergCatalog(
                catalog_name="hadoop_prod",
                path=iceberg_filesystem_catalog.connection_string,
            )
        )
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
                "DEFAULT": pd.array([None], dtype=pd.ArrowDtype(pa.large_string())),
                "PRIMARY_KEY": ["N"],
                "UNIQUE_KEY": ["N"],
                "CHECK": pd.array([None], dtype=pd.ArrowDtype(pa.large_string())),
                "EXPRESSION": pd.array([None], dtype=pd.ArrowDtype(pa.large_string())),
                "COMMENT": pd.array([None], dtype=pd.ArrowDtype(pa.large_string())),
                "POLICY NAME": pd.array([None], dtype=pd.ArrowDtype(pa.large_string())),
                "PRIVACY DOMAIN": pd.array(
                    [None], dtype=pd.ArrowDtype(pa.large_string())
                ),
            }
        )
        assert_equal_par(bodo_output, expected_output)
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name}"
            spark.sql(query)

        cleanup()


def test_describe_table_compiles_jit(iceberg_filesystem_catalog, memory_leak_check):
    """
    Verify that describe table compiles in JIT.
    """
    query = "DESCRIBE TABLE ANY_TABLE"
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bc.validate_query_compiles(query)


@pytest.mark.parametrize("if_exists", [True, False])
def test_iceberg_drop_view_unsupported_catalog_error_does_not_exist(
    if_exists, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests on the filesystem catalog to drop an non-exist view
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
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
                py_output="",
                test_str_literal=False,
            )


@pytest_mark_one_rank
@pytest.mark.parametrize("if_exists", [True, False])
def test_iceberg_drop_view_unsupported_catalog_error_non_view(
    if_exists, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests on the filesystem catalog to drop an non view file
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
    table_name = create_simple_ddl_table(spark)
    db_schema = "iceberg_db"
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    view_name = table_name
    if_exists_str = "IF EXISTS" if if_exists else ""
    query_drop_view = f'DROP VIEW {if_exists_str} "{db_schema}"."{table_name}"'
    # execute_ddl Version
    with pytest.raises(
        ValueError, match="DROP VIEW is unimplemented for the current catalog"
    ):
        bc.execute_ddl(query_drop_view)
    # Python Version
    with pytest.raises(
        ValueError, match="DROP VIEW is unimplemented for the current catalog"
    ):
        bc.sql(query_drop_view)
    # Jit Version
    with pytest.raises(
        ValueError, match="DROP VIEW is unimplemented for the current catalog"
    ):
        check_func_seq(
            lambda bc, query: bc.sql(query),
            (bc, query_drop_view),
            py_output="",
            test_str_literal=False,
        )

    # Clean up table generated for this test
    query_drop_table = f'DROP TABLE "{db_schema}"."{view_name}"'
    bc.execute_ddl(query_drop_table)
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


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


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_iceberg_describe_view_unsupported(
    describe_keyword, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog raises an error when describing an iceberg
    view.
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    view_name = gen_unique_id("TEST_VIEW").upper()
    query_describe_view = f"{describe_keyword} VIEW {view_name}"
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


def test_show_tables_compiles_jit(iceberg_filesystem_catalog, memory_leak_check):
    """
    Verify that show tables compiles in JIT.
    """
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    schema_name = gen_unique_id("TEST_SCHEMA_DDL").upper()
    query = f'CREATE SCHEMA "{schema_name}"'
    bc.execute_ddl(query)
    query = f"SHOW TERSE TABLES IN '{schema_name}'"
    bc.validate_query_compiles(query)


@pytest_mark_one_rank
@pytest.mark.parametrize("purge", [True, False])
def test_iceberg_drop_table_purge_sql(
    purge, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog can drop a table and delete all underlying files with or without purge.
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
    table_name = create_simple_ddl_table(spark)
    purge_str = "PURGE" if purge else ""
    db_schema = "iceberg_db"
    spark.sql(f"show tables in hadoop_prod.{db_schema} like '{table_name}'")

    query_data_file = f"SELECT * FROM hadoop_prod.{db_schema}.{table_name}.data_files"
    output = spark.sql(query_data_file).toPandas()
    files_path = [output["file_path"][i] for i in range(len(output["file_path"]))]

    query_drop_table_purge = f'DROP TABLE "{db_schema}"."{table_name}" {purge_str}'
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bc.sql(query_drop_table_purge)
    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"

    for path in files_path:
        with pytest.raises(FileNotFoundError):
            pd.read_parquet(path, dtype_backend="pyarrow")


@pytest_mark_one_rank
@pytest.mark.parametrize("purge", [True, False])
def test_iceberg_drop_table_purge_execute_dll(
    purge, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog can drop a table and delete all underlying files with or without purge.
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
    table_name = create_simple_ddl_table(spark)
    purge_str = "PURGE" if purge else ""
    db_schema = "iceberg_db"
    spark.sql(f"show tables in hadoop_prod.{db_schema} like '{table_name}'")

    query_data_file = f"SELECT * FROM hadoop_prod.{db_schema}.{table_name}.data_files"
    output = spark.sql(query_data_file).toPandas()
    files_path = [output["file_path"][i] for i in range(len(output["file_path"]))]

    query_drop_table_purge = f'DROP TABLE "{db_schema}"."{table_name}" {purge_str}'
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    bc.execute_ddl(query_drop_table_purge)
    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"

    for path in files_path:
        with pytest.raises(FileNotFoundError):
            pd.read_parquet(path, dtype_backend="pyarrow")


@pytest_mark_one_rank
@pytest.mark.parametrize("purge", [True, False])
def test_iceberg_drop_table_purge(purge, iceberg_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can drop a table and delete all underlying files with or without purge.
    """
    spark = get_spark(
        SparkFilesystemIcebergCatalog(
            catalog_name="hadoop_prod",
            path=iceberg_filesystem_catalog.connection_string,
        )
    )
    table_name = create_simple_ddl_table(spark)
    purge_str = "PURGE" if purge else ""
    db_schema = "iceberg_db"
    spark.sql(f"show tables in hadoop_prod.{db_schema} like '{table_name}'")

    query_data_file = f"SELECT * FROM hadoop_prod.{db_schema}.{table_name}.data_files"
    output = spark.sql(query_data_file).toPandas()
    files_path = [output["file_path"][i] for i in range(len(output["file_path"]))]

    query_drop_table_purge = f'DROP TABLE "{db_schema}"."{table_name}" {purge_str}'
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    impl(bc, query_drop_table_purge)
    # Verify we can't find the table.
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"

    for path in files_path:
        with pytest.raises(FileNotFoundError):
            pd.read_parquet(path, dtype_backend="pyarrow")
