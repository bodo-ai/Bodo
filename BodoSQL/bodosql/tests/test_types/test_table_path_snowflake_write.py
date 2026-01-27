"""
Tests write with a Snowflake SQL TablePath.
"""

import datetime

import numpy as np
import pandas as pd
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    create_snowflake_table,
    get_snowflake_connection_string,
    pytest_snowflake,
)

pytestmark = pytest_snowflake


def test_insert_into(memory_leak_check):
    """Tests writing to a Snowflake DB with copy into.
    To enable the full suite of check_func we recreate the
    table on each run.
    """

    def write_impl(table_name, schema, create_df, append_df, conn_str):
        # Create the Snowflake table.
        create_df.to_sql(table_name, conn_str, if_exists="replace", index=False)
        bc = bodosql.BodoSQLContext(
            {
                "DEST_TABLE": bodosql.TablePath(
                    table_name, "sql", conn_str=conn_str, db_schema=schema
                ),
                "T1": append_df,
            }
        )
        bc.sql("INSERT INTO dest_table(B, C) SELECT B, C FROM T1 WHERE A > 5")

    def read_impl(read_query, conn_str):
        return pd.read_sql(read_query, conn_str)

    create_df = pd.DataFrame(
        {
            "A": np.arange(5),
            "B": ["a", "@42", "42", "@32", "12"],
            "C": [23.1, 12.1, 11, 4242.2, 95],
        }
    )
    append_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
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
            # Generate the reference answer
            filtered_df = append_df[append_df.A > 5][["B", "C"]]
            py_output = pd.concat((create_df, filtered_df), ignore_index=True)

        py_output = comm.bcast(py_output)
        py_output.columns = ["a", "b", "c"]

        # Since we separate read and write into different impl's,
        # we must do the 1D parallel work from check_func done
        # on read_impl for write_impl as well
        create_df = bodo.scatterv(create_df)
        append_df = bodo.scatterv(append_df)
        bodo.jit(distributed=["create_df", "append_df"])(write_impl)(
            table_name, schema, create_df, append_df, conn_str
        )

        bodo.barrier()

        check_func(
            read_impl,
            (read_query, conn_str),
            py_output=py_output,
            only_1D=True,
            reset_index=True,
            sort_output=True,
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

    def write_impl(table_name, schema, create_df, append_df, conn_str):
        # Create the Snowflake table.
        create_df.to_sql(table_name, conn_str, if_exists="replace", index=False)
        bc = bodosql.BodoSQLContext(
            {
                "DEST_TABLE": bodosql.TablePath(
                    table_name, "sql", conn_str=conn_str, db_schema=schema
                ),
                "T1": append_df,
            }
        )
        bc.sql("INSERT INTO dest_table(A) SELECT A from t1 where C > 2")

    def read_impl(read_query, conn_str):
        return pd.read_sql(read_query, conn_str)

    create_df = pd.DataFrame(
        {
            "A": [
                datetime.date(2022, 1, 1),
                datetime.date(2022, 2, 1),
                datetime.date(2022, 3, 1),
                datetime.date(2022, 4, 1),
                datetime.date(2022, 5, 1),
            ],
            "B": [
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
            "A": [
                datetime.date(2022, 1, 1),
                datetime.date(2021, 1, 1),
                datetime.date(2020, 1, 1),
                datetime.date(2019, 1, 1),
                datetime.date(2018, 1, 1),
            ],
            "C": np.arange(5),
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
            # Generate reference answer
            filtered_df = append_df[append_df.C > 2][["A"]]
            py_output = pd.concat((create_df, filtered_df), ignore_index=True)

        py_output = comm.bcast(py_output)
        py_output.columns = ["a", "b"]

        # Since we separate read and write into different impl's,
        # we must do the 1D parallel work from check_func done
        # on read_impl for write_impl as well
        create_df = bodo.scatterv(create_df)
        append_df = bodo.scatterv(append_df)
        bodo.jit(distributed=["create_df", "append_df"])(write_impl)(
            table_name, schema, create_df, append_df, conn_str
        )

        bodo.barrier()

        check_func(
            read_impl,
            (read_query, conn_str),
            py_output=py_output,
            only_1D=True,
            reset_index=True,
            sort_output=True,
        )
