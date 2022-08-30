# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests write with a Snowflake SQL TablePath.
"""
import os
import platform

import bodosql
import numpy as np
import pandas as pd
import pytest
from numba.core.utils import PYVERSION

import bodo
from bodo.tests.utils import check_func, get_snowflake_connection_string


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_insert_into(memory_leak_check):
    """Tests writing to a Snowflake DB with copy into.
    To enable the full suite of check_func we recreate the
    table on each run.
    """

    def impl(table_name, schema, new_df, read_query):
        # Create the Snowflake table.
        df = pd.DataFrame(
            {
                "A": np.arange(5),
                "B": ["a", "@42", "42", "@32", "12"],
                "C": [23.1, 12.1, 11, 4242.2, 95],
            }
        )
        df.to_sql(table_name, conn_str, if_exists="replace", index=False)
        bc = bodosql.BodoSQLContext(
            {
                "dest_table": bodosql.TablePath(
                    table_name, "sql", conn_str=conn_str, db_schema=schema
                ),
                "t1": new_df,
            }
        )
        bc.sql("INSERT INTO dest_table(B, C) SELECT B, C FROM T1 WHERE A > 5")
        bodo.barrier()
        return pd.read_sql(read_query, conn_str)

    py_str = f"{PYVERSION[0]}_{PYVERSION[1]}"
    table_name = (
        f"bodosql_write_test_{platform.system()}_{bodo.get_size()}_{py_str}".lower()
    )
    schema = "PUBLIC"
    db = "TEST_DB"
    conn_str = get_snowflake_connection_string(db, schema)
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )
    if bodo.get_rank() == 0:
        # Create the expected output on rank 0.
        df = pd.DataFrame(
            {
                "A": np.arange(5),
                "B": ["a", "@42", "42", "@32", "12"],
                "C": [23.1, 12.1, 11, 4242.2, 95],
            }
        )
        df.to_sql(table_name, conn_str, schema=schema, if_exists="replace", index=False)
        written_df = new_df[new_df.A > 5][["B", "C"]]
        written_df.to_sql(
            table_name, conn_str, schema=schema, if_exists="append", index=False
        )
    bodo.barrier()
    # Load the expected output on all ranks.
    read_query = f"select * from {table_name}"
    # We read the result direclty from snowflake in case it changes
    # the data in any way (types, names, etc.)
    py_output = pd.read_sql(read_query, conn_str)
    bodo.barrier()
    check_func(
        impl,
        (table_name, schema, new_df, read_query),
        py_output=py_output,
        only_1D=True,
        reset_index=True,
        sort_output=True,
    )
