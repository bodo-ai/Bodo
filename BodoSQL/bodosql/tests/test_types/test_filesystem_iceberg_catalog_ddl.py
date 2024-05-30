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
    _test_equal_guard,
    check_func_seq,
    gen_unique_table_id,
    reduce_sum,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import run_rank0
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
    try:

        @bodo.jit
        def impl(bc, query):
            return bc.sql(query)

        spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
        # Create an unused table to ensure the database is created
        created_table = create_simple_ddl_table(spark)
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
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{created_table}"
            spark.sql(query)

        cleanup()


def test_drop_table_not_found_if_exists(iceberg_filesystem_catalog, memory_leak_check):
    """Tests a table that doesn't exist in Iceberg doesn't raise an error
    with IF EXISTS."""
    try:

        @bodo.jit
        def impl(bc, query):
            return bc.sql(query)

        spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
        # Create an unused table to ensure the database is created
        created_table = create_simple_ddl_table(spark)
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
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{created_table}"
            spark.sql(query)

        cleanup()


@pytest.mark.parametrize("describe_keyword", ["DESCRIBE", "DESC"])
def test_describe_table(
    describe_keyword, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests that the filesystem catalog can describe an iceberg
    table.
    """
    try:
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
                "CHECK": [None],
                "EXPRESSION": [None],
                "COMMENT": [None],
                "POLICY NAME": [None],
                "PRIVACY DOMAIN": [None],
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
    query = f"DESCRIBE TABLE ANY_TABLE"
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
    py_output = pd.DataFrame({"STATUS": [f"View '{view_name}' successfully dropped."]})
    # execute_ddl Version
    if if_exists:
        bodo_output = bc.execute_ddl(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
    else:
        with pytest.raises(
            BodoError,
            match=f"View '{view_name}' does not exist or not authorized to drop.",
        ):
            bc.execute_ddl(query_drop_view)
    # Python Version
    if if_exists:
        bodo_output = bc.sql(query_drop_view)
        _test_equal_guard(bodo_output, py_output)
    else:
        with pytest.raises(
            BodoError,
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
            BodoError,
            match=f"View '{view_name}' does not exist or not authorized to drop.",
        ):
            check_func_seq(
                lambda bc, query: bc.sql(query),
                (bc, query_drop_view),
                py_output="",
                test_str_literal=False,
            )


@pytest.mark.parametrize("if_exists", [True, False])
def test_iceberg_drop_view_unsupported_catalog_error_non_view(
    if_exists, iceberg_filesystem_catalog, memory_leak_check
):
    """
    Tests on the filesystem catalog to drop an non view file
    """
    spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
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
        BodoError, match=f"DROP VIEW is unimplemented for the current catalog"
    ):
        bc.execute_ddl(query_drop_view)
    # Python Version
    with pytest.raises(
        BodoError, match=f"DROP VIEW is unimplemented for the current catalog"
    ):
        bc.sql(query_drop_view)
    # Jit Version
    with pytest.raises(
        BodoError, match=f"DROP VIEW is unimplemented for the current catalog"
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


def test_show_objects(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can shows objects in iceberg
    schema.
    """
    try:
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
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name1}"
            spark.sql(query)
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name2}"
            spark.sql(query)

        cleanup()


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


def test_show_tables(iceberg_filesystem_catalog, iceberg_database, memory_leak_check):
    """
    Tests that the filesystem catalog can show tables in iceberg
    schema.
    """
    try:
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

        query = f'SHOW TERSE TABLES IN "{db_schema}"'
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
        assert n_passed == bodo.get_size(), "Show tables test failed"
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name1}"
            spark.sql(query)
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name2}"
            spark.sql(query)

        cleanup()


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


def test_show_no_terse_error(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """Tests that SHOW commands without the TERSE option
    raises an appropriate error."""

    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
    # The command will be caught before even connecting to the catalog
    # and thus we don't need to setup any tables or schemas

    with pytest.raises(BodoError, match="Only SHOW TERSE is currently supported"):
        bodo_output = bc.execute_ddl(f"SHOW TABLES in junkSchema")
    with pytest.raises(BodoError, match="Only SHOW TERSE is currently supported"):
        bodo_output = bc.execute_ddl(f"SHOW VIEWS in junkSchema")
    with pytest.raises(BodoError, match="Only SHOW TERSE is currently supported"):
        bodo_output = bc.execute_ddl(f"SHOW OBJECTS in junkSchema")
    with pytest.raises(BodoError, match="Only SHOW TERSE is currently supported"):
        bodo_output = bc.execute_ddl(f"SHOW SCHEMAS in junkSchema")


def test_show_views_error(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """
    Tests that the filesystem catalog appropriately raises an error when
    attempting to show views.
    """
    try:
        spark = get_spark(path=iceberg_filesystem_catalog.connection_string)
        db_schema = "iceberg_db"
        # Create 2 tables
        table_name1 = create_simple_ddl_table(spark)
        existing_tables = spark.sql(
            f"show tables in hadoop_prod.{db_schema} like '{table_name1}'"
        ).toPandas()
        assert len(existing_tables) == 1, "Unable to find testing table"

        query = f'SHOW TERSE VIEWS IN "{db_schema}"'
        bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
        with pytest.raises(
            BodoError, match="SHOW VIEWS is unimplemented for the current catalog"
        ):
            bodo_output = bc.execute_ddl(query)
    finally:
        # Cleanup
        @run_rank0
        def cleanup():
            query = f"DROP TABLE hadoop_prod.{db_schema}.{table_name1}"
            spark.sql(query)

        cleanup()
