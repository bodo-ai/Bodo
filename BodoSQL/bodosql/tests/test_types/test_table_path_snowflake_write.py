# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests write with a Snowflake SQL TablePath.
"""
import datetime
import os

import bodosql
import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (
    check_func,
    create_snowflake_table,
    get_snowflake_connection_string,
)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_insert_into(memory_leak_check):
    """Tests writing to a Snowflake DB with copy into.
    To enable the full suite of check_func we recreate the
    table on each run.
    """

    def impl(table_name, schema, create_df, append_df, read_query):
        # Create the Snowflake table.
        create_df.to_sql(table_name, conn_str, if_exists="replace", index=False)
        bc = bodosql.BodoSQLContext(
            {
                "dest_table": bodosql.TablePath(
                    table_name, "sql", conn_str=conn_str, db_schema=schema
                ),
                "t1": append_df,
            }
        )
        bc.sql("INSERT INTO dest_table(B, C) SELECT B, C FROM T1 WHERE A > 5")
        bodo.barrier()
        return pd.read_sql(read_query, conn_str)

    create_df = pd.DataFrame(
        {
            "a": np.arange(5),
            "b": ["a", "@42", "42", "@32", "12"],
            "c": [23.1, 12.1, 11, 4242.2, 95],
        }
    )
    append_df = pd.DataFrame(
        {"a": [1, 3, 5, 7, 9] * 10, "b": ["Afe", "fewfe"] * 25, "c": 1.1}
    )
    comm = MPI.COMM_WORLD
    schema = "PUBLIC"
    db = "TEST_DB"
    with create_snowflake_table(
        create_df, "bodosql_basic_write_test", db, schema
    ) as table_name:
        conn_str = get_snowflake_connection_string(db, schema)
        read_query = f"select * from {table_name}"
        py_output = None
        if bodo.get_rank() == 0:
            # Append to the output so we are consistent for types.
            filtered_df = append_df[append_df.a > 5][["b", "c"]]
            filtered_df.to_sql(
                table_name, conn_str, schema=schema, if_exists="append", index=False
            )
            # We read the result direclty from snowflake in case it changes
            # the data in any way (types, names, etc.)
            py_output = pd.read_sql(read_query, conn_str)
        py_output = comm.bcast(py_output)
        check_func(
            impl,
            (table_name, schema, create_df, append_df, read_query),
            py_output=py_output,
            only_1D=True,
            reset_index=True,
            sort_output=True,
        )


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_insert_into_date(memory_leak_check):
    """Tests writing to a Snowflake DB with copy into
    using date columns. Date columns currently require
    BodoSQL to generate a cast from date -> datetime64
    when performing the query and from datetime64 -> date
    to then complete the write.

    To enable the full suite of check_func we recreate the
    table on each run.
    """

    def impl(table_name, schema, create_df, append_df, read_query):
        # Create the Snowflake table.
        create_df.to_sql(table_name, conn_str, if_exists="replace", index=False)
        bc = bodosql.BodoSQLContext(
            {
                "dest_table": bodosql.TablePath(
                    table_name, "sql", conn_str=conn_str, db_schema=schema
                ),
                "t1": append_df,
            }
        )
        bc.sql("INSERT INTO dest_table(A) SELECT A from t1 where C > 2")
        bodo.barrier()
        return pd.read_sql(read_query, conn_str)

    create_df = pd.DataFrame(
        {
            "a": [
                datetime.date(2022, 1, 1),
                datetime.date(2022, 2, 1),
                datetime.date(2022, 3, 1),
                datetime.date(2022, 4, 1),
                datetime.date(2022, 5, 1),
            ],
            "b": [
                datetime.date(2022, 1, 1),
                datetime.date(2022, 1, 2),
                datetime.date(2022, 1, 3),
                datetime.date(2022, 1, 4),
                datetime.date(2022, 1, 5),
            ],
        }
    )
    append_df = pd.DataFrame(
        {
            "a": [
                datetime.date(2022, 1, 1),
                datetime.date(2021, 1, 1),
                datetime.date(2020, 1, 1),
                datetime.date(2019, 1, 1),
                datetime.date(2018, 1, 1),
            ],
            "c": np.arange(5),
        }
    )
    comm = MPI.COMM_WORLD
    schema = "PUBLIC"
    db = "TEST_DB"
    with create_snowflake_table(
        create_df, "bodosql_date_write_test", db, schema
    ) as table_name:
        conn_str = get_snowflake_connection_string(db, schema)
        read_query = f"select * from {table_name}"
        py_output = None
        if bodo.get_rank() == 0:
            # Append to the output so we are consistent for types.
            filtered_df = append_df[append_df.c > 2][["a"]]
            filtered_df.to_sql(
                table_name, conn_str, schema=schema, if_exists="append", index=False
            )
            # We read the result direclty from snowflake in case it changes
            # the data in any way (types, names, etc.)
            py_output = pd.read_sql(read_query, conn_str)
        py_output = comm.bcast(py_output)
        check_func(
            impl,
            (table_name, schema, create_df, append_df, read_query),
            py_output=py_output,
            only_1D=True,
            reset_index=True,
            sort_output=True,
        )
