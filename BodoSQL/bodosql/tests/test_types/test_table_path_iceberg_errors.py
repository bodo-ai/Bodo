# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests that Bodosql throws reasonable errors when supplied iceberg tables.

Reuses several fixtures from the engine, but using bodosql's TablePath
"""
import re

import bodosql
import pytest

import bodo
from bodo.tests.conftest import (  # pragma: no cover
    iceberg_database,
    iceberg_table_conn,
)
from bodo.utils.typing import BodoError


def test_iceberg_tablepath_errors(iceberg_database, iceberg_table_conn):
    """
    Test that TablePath raises an error when passing the wrong arguments
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn("simple_string_table", db_schema, warehouse_loc)

    with pytest.raises(
        BodoError,
        match=".*"
        + re.escape(
            "bodosql.TablePath(): `db_schema` is required for iceberg database type"
        )
        + ".*",
    ):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    "simple_string_table",
                    "sql",
                    conn_str=conn,
                )
            }
        )

    with pytest.raises(
        BodoError,
        match=".*"
        + re.escape("bodosql.TablePath(): `db_schema` must be a string")
        + ".*",
    ):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    "simple_string_table",
                    "sql",
                    conn_str=conn,
                    db_schema=10,
                )
            }
        )


@pytest.mark.slow
def test_iceberg_tablepath_errors_jit(iceberg_database, iceberg_table_conn):
    """
    Test that TablePath raises an error when passing the wrong arguments in JIT code
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn("simple_string_table", db_schema, warehouse_loc)

    with pytest.raises(
        BodoError,
        match=".*"
        + re.escape(
            "bodosql.TablePath(): `db_schema` is required for iceberg database type"
        )
        + ".*",
    ):

        def impl():
            bc = bodosql.BodoSQLContext(
                {
                    "iceberg_tbl": bodosql.TablePath(
                        "simple_string_table",
                        "sql",
                        conn_str=conn,
                    )
                }
            )

        impl()

    with pytest.raises(
        BodoError,
        match=".*"
        + re.escape("bodosql.TablePath(): `db_schema` must be a string")
        + ".*",
    ):

        def impl():
            bc = bodosql.BodoSQLContext(
                {
                    "iceberg_tbl": bodosql.TablePath(
                        "simple_string_table",
                        "sql",
                        conn_str=conn,
                        db_schema=10,
                    )
                }
            )

        impl()


def test_iceberg_tablepath_DNE(iceberg_database, iceberg_table_conn):
    """
    Tests that TablePath raises a reasonable error when the table
    does not exist.
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn("simple_string_table", db_schema, warehouse_loc)

    # test outside of JIT
    with pytest.raises(
        BodoError,
        match=".*" + re.escape("No such Iceberg table found") + ".*",
    ):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    "does_not_exist", "sql", conn_str=conn, db_schema=db_schema
                )
            }
        )

    # test inside of JIT
    with pytest.raises(
        BodoError,
        match=".*" + re.escape("No such Iceberg table found") + ".*",
    ):

        # Note, need to actually use the bodosqlContext, otherwise the error is not raised
        @bodo.jit()
        def test_func(conn, db_schema):
            bc = bodosql.BodoSQLContext(
                {
                    "iceberg_tbl": bodosql.TablePath(
                        "does_not_exist", "sql", conn_str=conn, db_schema=db_schema
                    )
                }
            )
            return bc.sql("SELECT * FROM iceberg_tbl")

        bc = test_func(conn, db_schema)
