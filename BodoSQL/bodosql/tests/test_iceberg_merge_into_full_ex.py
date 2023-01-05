# Copyright (C) 2022 Bodo Inc. All rights reserved.
import bodosql
import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table
from bodo.tests.utils import check_func

pytestmark = pytest.mark.iceberg


def test_merge_into_full_example(iceberg_database, iceberg_table_conn):
    """

    Test that MERGE INTO works in a more realistic e2e scenario.

    The test will do the following:
    * Create an iceberg table
    * Merge some data into table (acting as append)
    * Merge another table into table (acting as update)
    * Merge another table into table (acting as delete)
    * Write table back to iceberg
    * Read table back from iceberg
    * Assert that the data is as expected
    """

    initial_table = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
            "data": ["a", "b", "c", "d", "e"],
        }
    )

    append_source = pd.DataFrame(
        {
            "id": pd.Series([6, 7, 8, 9, 10], dtype="int64"),
            "data": ["f", "g", "h", "i", "j"],
        }
    )

    update_source = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
            "data": ["h", "i", "j", "k", "l"],
        }
    )

    delete_source = pd.DataFrame(
        {
            "id": pd.Series([1, 3, 5, 7, 9], dtype="int64"),
            "data": ["m", "n", "o", "p", "q"],
        }
    )

    expected = pd.DataFrame(
        {
            "id": pd.Series([2, 4, 6, 8, 10], dtype="int64"),
            "data": ["i", "k", "f", "h", "j"],
        }
    )

    # Create initial table
    table_name = "merge_into_e2e"
    db_schema, warehouse_loc = iceberg_database

    if bodo.get_rank() == 0:
        sql_schema = [("id", "long", True), ("data", "string", True)]
        create_iceberg_table(
            initial_table,
            sql_schema,
            table_name,
        )

    bodo.barrier()
    # Construct Connection String
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    bc = bodosql.BodoSQLContext(
        {
            "target_table": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "append_source": append_source,
            "update_source": update_source,
            "delete_source": delete_source,
        }
    )

    def impl(bc):
        # merge append
        bc.sql(
            "MERGE INTO target_table AS t USING append_source AS s "
            "ON t.id = s.id "
            "WHEN NOT MATCHED THEN "
            "INSERT (id, data) VALUES (s.id, s.data)"
        )

        # merge update
        bc.sql(
            "MERGE INTO target_table AS t USING update_source AS s "
            "ON t.id = s.id "
            "WHEN MATCHED THEN "
            "UPDATE SET data = s.data"
        )

        # merge delete
        bc.sql(
            "MERGE INTO target_table AS t USING delete_source AS s "
            "ON t.id = s.id "
            "WHEN MATCHED THEN "
            "DELETE"
        )
        return bc.sql("select * from target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        reset_index=True,
        sort_output=True,
    )
