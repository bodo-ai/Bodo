# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests that check for correct behavior when there is a typing
error with a variety of filters that may be pushed down.
"""

import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.utils.typing import BodoError
from bodosql import BodoSQLContext

pytestmark = pytest.mark.iceberg


def test_requires_transform(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Tests that we avoid an infinite loop while compiling code that requires
    a transformation in filter pushdown.
    """
    # We need a real table to ensure we reach filter pushdown.
    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = BodoSQLContext()

    @bodo.jit
    def impl_single(table_name, conn, db_schema, bc):
        # Just return df because sort_output, reset_index don't work when
        # returning tuples.
        df, _, _ = pd.read_sql_table(table_name, conn, db_schema, _bodo_merge_into=True)
        df = df[df.B.notna()]
        # Force a compiler error that needs a transformation
        if len(df) < 5:
            sql_str = "select * from table1"
        else:
            sql_str = "select * from table2"
        return bc.sql(sql_str)

    @bodo.jit
    def impl_and(table_name, conn, db_schema, bc):
        # Just return df because sort_output, reset_index don't work when
        # returning tuples.
        df, _, _ = pd.read_sql_table(table_name, conn, db_schema, _bodo_merge_into=True)
        df = df[df.B.notna() & df.B.isna()]
        # Force a compiler error that needs a transformation
        if len(df) < 5:
            sql_str = "select * from table1"
        else:
            sql_str = "select * from table2"
        return bc.sql(sql_str)

    @bodo.jit
    def impl_or(table_name, conn, db_schema, bc):
        # Just return df because sort_output, reset_index don't work when
        # returning tuples.
        df, _, _ = pd.read_sql_table(table_name, conn, db_schema, _bodo_merge_into=True)
        df = df[df.B.notna() | df.B.isna()]
        # Force a compiler error that needs a transformation
        if len(df) < 5:
            sql_str = "select * from table1"
        else:
            sql_str = "select * from table2"
        return bc.sql(sql_str)

    @bodo.jit
    def impl_and_or(table_name, conn, db_schema, bc):
        # Just return df because sort_output, reset_index don't work when
        # returning tuples.
        df, _, _ = pd.read_sql_table(table_name, conn, db_schema, _bodo_merge_into=True)
        df = df[(df.B.notna() & df.B.isna()) | (df.B.isna() & df.B.notna())]
        # Force a compiler error that needs a transformation
        if len(df) < 5:
            sql_str = "select * from table1"
        else:
            sql_str = "select * from table2"
        return bc.sql(sql_str)

    with pytest.raises(BodoError, match="Invalid BodoSQLContext"):
        impl_single(table_name, conn, db_schema, bc)
    with pytest.raises(BodoError, match="Invalid BodoSQLContext"):
        impl_and(table_name, conn, db_schema, bc)
    with pytest.raises(BodoError, match="Invalid BodoSQLContext"):
        impl_or(table_name, conn, db_schema, bc)
    with pytest.raises(BodoError, match="Invalid BodoSQLContext"):
        impl_and_or(table_name, conn, db_schema, bc)
