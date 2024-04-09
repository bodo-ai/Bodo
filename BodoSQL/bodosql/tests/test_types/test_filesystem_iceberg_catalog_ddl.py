# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests the general Iceberg DDL functionality with the file system catalog.
"""
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
    gen_unique_table_id,
    reduce_sum,
)
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg


@pytest.fixture
def iceberg_filesystem_catalog():
    """
    An Iceberg FileSystemCatalog.
    """
    return bodosql.FileSystemCatalog(".", "iceberg")


def create_simple_ddl_table(spark):
    """
    Creates a very basic table for testing DDL functionality.
    """
    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("SIMPLY_DDL_TABLE")
    table_name = comm.bcast(table_name)
    df = pd.DataFrame({"A": [1, 2, 3]})
    sql_schema = [
        ("A", "long", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    return table_name


def test_iceberg_drop_table(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """
    Tests that the filesystem catalog can drop a table.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark()
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
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_iceberg_drop_table_python(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """
    Tests that the filesystem catalog can drop a table using
    bc.sql from Python.
    """
    spark = get_spark()
    table_name = create_simple_ddl_table(spark)
    db_schema, _ = iceberg_database(table_name)
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'DROP TABLE "{db_schema}"."{table_name}"'
    py_output = pd.DataFrame({"STATUS": [f"{table_name} successfully dropped."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
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
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_iceberg_drop_table_execute_ddl(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """
    Tests that the filesystem catalog can drop a table using
    bc.execute_ddl from Python.
    """
    spark = get_spark()
    table_name = create_simple_ddl_table(spark)
    db_schema, _ = iceberg_database(table_name)
    existing_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(existing_tables) == 1, "Unable to find testing table"

    query = f'DROP TABLE "{db_schema}"."{table_name}"'
    py_output = pd.DataFrame({"STATUS": [f"{table_name} successfully dropped."]})
    bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
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
    remaining_tables = spark.sql(
        f"show tables in hadoop_prod.{db_schema} like '{table_name}'"
    ).toPandas()
    assert len(remaining_tables) == 0, "Table was not dropped"


def test_drop_table_not_found(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """Tests a table that doesn't exist in Iceberg raises an error."""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark()
    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    db_schema, _ = iceberg_database(table_name)
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


def test_drop_table_not_found_if_exists(
    iceberg_filesystem_catalog, iceberg_database, memory_leak_check
):
    """Tests a table that doesn't exist in Iceberg doesn't raise an error
    with IF EXISTS."""

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    spark = get_spark()
    # Create a garbage table name.
    table_name = "FEJWIOPFE13_9029J03C32"
    db_schema, _ = iceberg_database(table_name)
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
    passed = _test_equal_guard(
        bodo_output,
        py_output,
        sort_output=True,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Sequential test failed"
