"""
Tests that Bodosql throws reasonable errors when supplied iceberg tables.

Reuses several fixtures from the engine, but using bodosql's TablePath
"""

import re

import pytest

import bodo
import bodosql
from bodo.tests.conftest import (  # noqa: F401
    iceberg_database,
    iceberg_table_conn,
)
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg


@pytest.mark.slow
def test_iceberg_tablepath_errors(iceberg_database, iceberg_table_conn):
    """
    Test that TablePath raises an error when passing the wrong arguments
    """
    db_schema, warehouse_loc = iceberg_database(["SIMPLE_STRING_TABLE"])
    conn = iceberg_table_conn("SIMPLE_STRING_TABLE", db_schema, warehouse_loc)

    with pytest.raises(
        ValueError,
        match=".*"
        + re.escape(
            "bodosql.TablePath(): `db_schema` is required for iceberg database type"
        )
        + ".*",
    ):
        bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    "SIMPLE_STRING_TABLE",
                    "sql",
                    conn_str=conn,
                )
            }
        )

    with pytest.raises(
        ValueError,
        match=".*"
        + re.escape("bodosql.TablePath(): `db_schema` must be a string")
        + ".*",
    ):
        bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    "SIMPLE_STRING_TABLE",
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
    db_schema, warehouse_loc = iceberg_database(["SIMPLE_STRING_TABLE"])
    conn = iceberg_table_conn("SIMPLE_STRING_TABLE", db_schema, warehouse_loc)

    with pytest.raises(
        ValueError,
        match=".*"
        + re.escape(
            "bodosql.TablePath(): `db_schema` is required for iceberg database type"
        )
        + ".*",
    ):

        def impl():
            bodosql.BodoSQLContext(
                {
                    "iceberg_tbl": bodosql.TablePath(
                        "SIMPLE_STRING_TABLE",
                        "sql",
                        conn_str=conn,
                    )
                }
            )

        impl()

    with pytest.raises(
        ValueError,
        match=".*"
        + re.escape("bodosql.TablePath(): `db_schema` must be a string")
        + ".*",
    ):

        def impl():
            bodosql.BodoSQLContext(
                {
                    "iceberg_tbl": bodosql.TablePath(
                        "SIMPLE_STRING_TABLE",
                        "sql",
                        conn_str=conn,
                        db_schema=10,
                    )
                }
            )

        impl()


@pytest.mark.slow
def test_iceberg_tablepath_DNE(iceberg_database, iceberg_table_conn):
    """
    Tests that TablePath raises a reasonable error when the table
    does not exist.
    """
    db_schema, warehouse_loc = iceberg_database(["SIMPLE_STRING_TABLE"])
    conn = iceberg_table_conn("SIMPLE_STRING_TABLE", db_schema, warehouse_loc)

    # test outside of JIT
    with pytest.raises(
        ValueError,
        match=".*"
        + re.escape("No table with identifier iceberg_db.does_not_exist exists")
        + ".*",
    ):
        bc = bodosql.BodoSQLContext(
            {
                "ICEBERG_TBL": bodosql.TablePath(
                    "does_not_exist", "sql", conn_str=conn, db_schema=db_schema
                )
            }
        )
        bc.sql("SELECT * FROM iceberg_tbl")

    # test inside of JIT
    with pytest.raises(
        BodoError,
        match=".*"
        + re.escape("No table with identifier iceberg_db.does_not_exist exists")
        + ".*",
    ):
        # Note, need to actually use the bodosqlContext, otherwise the error is not raised
        @bodo.jit()
        def test_func(conn, db_schema):
            bc = bodosql.BodoSQLContext(
                {
                    "ICEBERG_TBL": bodosql.TablePath(
                        "does_not_exist", "sql", conn_str=conn, db_schema=db_schema
                    )
                }
            )
            return bc.sql("SELECT * FROM iceberg_tbl")

        test_func(conn, db_schema)
