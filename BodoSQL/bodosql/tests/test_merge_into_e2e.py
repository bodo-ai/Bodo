# Copyright (C) 2022 Bodo Inc. All rights reserved.

import os
import bodo
import pytest
import bodosql
import pandas as pd

from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    get_spark,
    create_iceberg_table,
)


@pytest.mark.skip(reason="Waiting for merge into support")
def test_merge_into_e2e(memory_leak_check):
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
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
        }
    )

    append_source = pd.DataFrame(
        {
            "id": [6, 7, 8, 9, 10],
            "data": ["f", "g", "h", "i", "j"],
        }
    )

    update_source = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "data": ["h", "i", "j", "k", "l"],
        }
    )

    delete_source = pd.DataFrame(
        {
            "id": [1, 3, 5, 7, 9],
            "data": ["m", "n", "o", "p", "q"],
        }
    )

    expected = pd.DataFrame(
        {
            "id": [2, 4, 6, 8, 10],
            "name": ["i", "k", "f", "h", "j"],
        }
    )

    # open connection and create initial table
    spark = get_spark()
    warehouse_loc = os.path.abspath(os.getcwd())
    db_schema = DATABASE_NAME
    table_name = "merge_into_e2e"
    sql_schema = [("id", "int", False), ("data", "string", False)]
    create_iceberg_table(
        initial_table,
        sql_schema,
        table_name,
        spark,
    )
    conn = f"iceberg://{warehouse_loc}"

    def impl(table_name, conn, db_schema):
        target_table = pd.read_sql_table(table_name, conn, db_schema)

        bc = bodosql.BodoSQLContext(
            {
                "target_table": target_table,
                "append_source": append_source,
                "update_source": update_source,
                "delete_source": delete_source,
            }
        )

        # merge append
        bc.sql(
            "MERGE INTO target_table AS t USING append_source AS s "
            "ON t.id = s.id "
            "WHEN NOT MATCHED THEN "
            "  INSERT (id, data) VALUES (s.id, s.data)"
        )

        # merge update
        bc.sql(
            "MERGE INTO target_table AS t USING update_source AS s "
            "ON t.id = s.id "
            "WHEN MATCHED THEN "
            "  UPDATE SET t.data = s.data"
        )

        # merge delete
        bc.sql(
            "MERGE INTO target_table AS t USING delete_source AS s "
            "ON t.id = s.id "
            "WHEN MATCHED THEN "
            "  DELETE"
        )

        # write back to iceberg
        target_table = bc.sql(f"SELECT * FROM target_table ORDER BY id")
        target_table.to_sql(table_name, conn, db_schema, if_exists="replace")

    impl(table_name, conn, db_schema)

    # read back from iceberg
    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    # assert equality
    pd.testing.assert_frame_equal(py_out, expected)
