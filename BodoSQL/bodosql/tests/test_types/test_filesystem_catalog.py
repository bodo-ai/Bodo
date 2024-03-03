# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests the core functionality with the file system catalog.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.utils import check_func


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
    Tests that the filesystem catalog can write parquet files. This test generates a dummy
    directory and write a local DataFrame to a table in that location. Then the table is read
    and the data is checked.

    To simplify the test we only run it with 1D Var.
    """
    comm = MPI.COMM_WORLD

    def write_impl(bc, query):
        bc.sql(query)
        # Return a constant value so we can use check_func.
        return 0

    df = pd.DataFrame({"A": np.arange(100), "B": [str(i) for i in range(100)]})

    try:
        filename = None
        if bodo.get_rank() == 0:
            filename = tempfile.mkdtemp()
        filename = comm.bcast(filename)
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
        pd.testing.assert_frame_equal(result, df)
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(filename, ignore_errors=True)
        bodo.barrier()
