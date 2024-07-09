# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests the core functionality with the file system catalog.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.utils import check_func
from bodo.utils.testing import ensure_clean2


@pytest.fixture(
    params=[
        pytest.param(bodosql.FileSystemCatalog("."), id="only-path"),
        pytest.param(
            bodosql.FileSystemCatalog(".", "iceberg"),
            marks=pytest.mark.slow,
            id="iceberg-write",
        ),
        pytest.param(
            bodosql.FileSystemCatalog(".", "parquet"),
            marks=pytest.mark.slow,
            id="parquet-write",
        ),
    ]
)
def dummy_filesystem_catalog(request):
    """
    A dummy filesystem catalog used for most tests without any real functionality.
    """
    return request.param


def test_filesystem_catalog_boxing(dummy_filesystem_catalog, memory_leak_check):
    """
    Tests that the filesystem catalog can be boxed and unboxed.
    """

    def impl(catalog):
        return catalog

    check_func(impl, (dummy_filesystem_catalog,))


def test_bodosql_context_boxing(dummy_filesystem_catalog, memory_leak_check):
    """
    Tests that the BodoSQL context can be boxed and unboxed when it contains
    a filesystem catalog.
    """

    def impl(bc):
        return bc

    bc = bodosql.BodoSQLContext(catalog=dummy_filesystem_catalog)
    check_func(impl, (bc,))


def test_filesystem_parquet_write(memory_leak_check):
    """
    Tests that the filesystem catalog can write parquet files based on the
    default schema.

    To simplify the test we only run it with 1D Var.
    """
    comm = MPI.COMM_WORLD

    def write_impl(bc, query):
        bc.sql(query)
        # Return a constant value so we can use check_func.
        return 0

    df = pd.DataFrame({"A": np.arange(100), "B": [str(i) for i in range(100)]})

    filename = None
    if bodo.get_rank() == 0:
        filename = tempfile.mkdtemp()
    filename = comm.bcast(filename)
    with ensure_clean2(filename):
        # Generate the catalog from the path
        root = os.path.dirname(filename)
        catalog = bodosql.FileSystemCatalog(root, "parquet")
        bc = bodosql.BodoSQLContext({"TABLE1": df}, catalog=catalog)
        # Generate the write query
        schema = os.path.basename(filename)
        write_query = f'create table "{schema}".TABLE2 as select * from TABLE1'
        # Write the table
        check_func(write_impl, (bc, write_query), py_output=0, only_1DVar=True)
        # Read the table with pandas and validate the result.
        result = pd.read_parquet(os.path.join(root, schema, "TABLE2"))
        result = result.sort_values("A").reset_index(drop=True)
        df = df.sort_values("A").reset_index(drop=True)
        pd.testing.assert_frame_equal(result, df)


def test_default_schema_filesystem_parquet_write(memory_leak_check):
    """
    Tests the FileSystem catalog properly obeys the provided default
    schema when writing a file.

    To simplify the test we only run it with 1D Var.
    """
    comm = MPI.COMM_WORLD

    def write_impl(bc, query):
        bc.sql(query)
        # Return a constant value so we can use check_func.
        return 0

    # Write two different DataFrames to verify what gets written where.
    df1 = pd.DataFrame({"A": np.arange(100), "B": [str(i) for i in range(100)]})
    df2 = pd.DataFrame(
        {"A": np.arange(100, 200), "B": [str(i) for i in range(100, 200)]}
    )
    tables = {"TABLE1": df1, "TABLE2": df2}
    schema1 = "schema1"
    schema2 = "schema2"
    schema3 = "schema3"

    filename = None
    if bodo.get_rank() == 0:
        filename = tempfile.mkdtemp()
        # Make the directories for the schema levels
        os.mkdir(os.path.join(filename, schema1))
        os.mkdir(os.path.join(filename, schema1, schema2))
        os.mkdir(os.path.join(filename, schema1, schema2, schema3))
    filename = comm.bcast(filename)
    with ensure_clean2(filename):
        # Generate the catalog from the path
        root = filename
        catalog1 = bodosql.FileSystemCatalog(root, "parquet", f'"{schema1}"')
        catalog2 = bodosql.FileSystemCatalog(
            root, "parquet", f'"{schema1}"."{schema2}"."{schema3}"'
        )
        bc1 = bodosql.BodoSQLContext(tables, catalog=catalog1)
        bc2 = bodosql.BodoSQLContext(tables, catalog=catalog2)
        # Write to the first location
        write_query1 = f"create table OUT_TABLE as select * from TABLE1"
        check_func(write_impl, (bc1, write_query1), py_output=0, only_1DVar=True)
        # Write to the second location
        write_query2 = f"create table OUT_TABLE as select * from TABLE2"
        check_func(write_impl, (bc2, write_query2), py_output=0, only_1DVar=True)
        # Read the table with pandas and validate the result.
        path1 = os.path.join(root, schema1, "OUT_TABLE")
        path2 = os.path.join(root, schema1, schema2, schema3, "OUT_TABLE")
        result1 = pd.read_parquet(path1)
        result2 = pd.read_parquet(path2)
        pd.testing.assert_frame_equal(result1, df1)
        pd.testing.assert_frame_equal(result2, df2)


def test_filesystem_parquet_write_no_schema(memory_leak_check):
    """
    Tests that the filesystem catalog can write parquet files without
    any schema.

    To simplify the test we only run it with 1D Var.
    """
    comm = MPI.COMM_WORLD

    def write_impl(bc, query):
        bc.sql(query)
        # Return a constant value so we can use check_func.
        return 0

    df1 = pd.DataFrame({"A": np.arange(100), "B": [str(i) for i in range(100)]})
    tables = {"TABLE1": df1}

    filename = None
    if bodo.get_rank() == 0:
        filename = tempfile.mkdtemp()
    filename = comm.bcast(filename)
    with ensure_clean2(filename):
        # Generate the catalog from the path
        root = filename
        catalog = bodosql.FileSystemCatalog(root, "parquet")
        bc = bodosql.BodoSQLContext(tables, catalog=catalog)
        # Write the data.
        write_query = f"create table OUT_TABLE as select * from TABLE1"
        check_func(write_impl, (bc, write_query), py_output=0, only_1DVar=True)
        # Read the table with pandas and validate the result.
        path = os.path.join(root, "OUT_TABLE")
        result1 = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result1, df1)
