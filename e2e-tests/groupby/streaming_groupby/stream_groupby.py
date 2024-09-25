import os
import time
from uuid import uuid4

import numba
import numpy as np
import pandas as pd
from utils.utils import checksum_str_df, drop_sf_table, get_sf_table

import bodo
import bodosql
from bodo.mpi4py import MPI

comm = MPI.COMM_WORLD


def checksum(df):
    """
    Calculate checksum for the output dataframe.
    The checksum is global, i.e. we compute checksum
    locally on each rank and then we do an allreduce
    sum.

    Args:
        df (pd.DataFrame): Output Dataframe to get checksum of.

    Returns:
        int: Checksum
    """
    str_cols = []
    for col_name, col_dtype in zip(df.columns, df.dtypes):
        if isinstance(col_dtype, pd.StringDtype):
            str_cols.append(col_name)

    df_no_str = df.drop(str_cols, axis=1)

    # multiplying by 100 so cent values don't get truncated from the checksum
    df_no_str_sum = np.int64(np.floor(df_no_str.sum().sum() * 100))
    df_no_str_sum = comm.allreduce(df_no_str_sum, op=MPI.SUM)

    df_str = df[str_cols]
    df_str_sum = checksum_str_df(df_str)
    df_checksum = df_no_str_sum + df_str_sum
    return df_checksum


@bodo.jit(cache=True)
def run_query_agg(bc, input_schema, output_table_name):
    """
    Run a simple groupby that would go through the
    incremental aggregation code path in stream
    groupby.
    """
    query = f"""
        create or replace transient table {output_table_name} as
        select
            l_suppkey,
            l_shipmode,
            sum(l_quantity),
            max(l_extendedprice),
            min(l_extendedprice),
            stddev(l_discount),
            count(l_comment)
        from {input_schema}.lineitem
        group by 1, 2
    """
    bc.sql(query)


@bodo.jit(cache=True)
def run_query_acc(bc, input_schema, output_table_name):
    """
    Run a simple groupby that would go through the
    input accumulation code path in stream
    groupby.
    """
    query = f"""
        create or replace transient table {output_table_name} as
        select
            l_suppkey,
            l_shipmode,
            count(distinct l_comment),
            max(l_shipinstruct),
            median(l_tax),
            median(l_quantity)
        from {input_schema}.lineitem
        group by 1, 2
    """
    bc.sql(query)


@bodo.jit(cache=True)
def run_query_drop_duplicates(bc, input_schema, output_table_name):
    """
    Run a simple drop-duplicates (select distinct) query.
    """
    # Requires some compute before the select-distinct because
    # without it, BodoSQL will push the entire query into
    # Snowflake
    query = f"""
        create or replace transient table {output_table_name} as
        select distinct
            l_suppkey,
            l_shipmode
        from (
            select l_suppkey, l_shipmode, o_comment 
            from 
                {input_schema}.lineitem, {input_schema}.orders 
            where l_orderkey = o_orderkey
        )
    """
    bc.sql(query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ftype", type=str, choices=["acc", "agg", "drop_duplicates"], required=True
    )
    parser.add_argument("--require_cache", action="store_true", default=False)
    parser.add_argument("--input_schema", type=str, default="tpch_sf10")
    parser.add_argument("--expected_out_len", type=int)
    parser.add_argument("--expected_checksum_lower", type=int)
    parser.add_argument("--expected_checksum_upper", type=int)
    args = parser.parse_args()
    ftype = args.ftype
    input_schema = args.input_schema
    expected_out_len = args.expected_out_len
    expected_checksum_lower = args.expected_checksum_lower
    expected_checksum_upper = args.expected_checksum_upper

    run_query_func = {
        "acc": run_query_acc,
        "agg": run_query_agg,
        "drop_duplicates": run_query_drop_duplicates,
    }[ftype]

    username = os.environ["SF_USERNAME"]
    password = os.environ["SF_PASSWORD"]
    account = os.environ["SF_ACCOUNT"]
    warehouse = "DEMO_WH"
    db = "E2E_TESTS_DB"

    catalog = bodosql.SnowflakeCatalog(
        username,
        password,
        account,
        warehouse,
        db,
    )

    bc = bodosql.BodoSQLContext(
        catalog=catalog,
    )

    comm = MPI.COMM_WORLD
    # Create a random table name to write to and broadcast to all ranks
    output_table_name = None
    output_schema = "public"
    if comm.Get_rank() == 0:
        # We use uppercase due to an issue reading tables
        # with lowercase letters.
        # https://bodo.atlassian.net/browse/BE-3534
        output_table_name = (
            f"{output_schema}.groupby_streaming_e2e_out_{str(uuid4())[:8]}".upper()
        )
        print("output_table_name: ", output_table_name)
    output_table_name = comm.bcast(output_table_name)

    ## Write output to Snowflake table
    start_time = time.time()
    out = run_query_func(bc, input_schema, output_table_name)
    end_time = time.time()

    # Add a barrier for synchronization purposes so that
    # the get length and drop commands are done after all the writes.
    bodo.barrier()
    if bodo.get_rank() == 0:
        print("Total time: ", end_time - start_time, " s")

    ## Read table from Snowflake and compute checksum
    conn = f"snowflake://{username}:{password}@{account}/{db}/{output_schema}?warehouse={warehouse}"
    # Compute checksum and length and make sure they're correct
    df = get_sf_table(output_table_name, conn)
    sf_df_checksum = checksum(df)
    len_sf_df = comm.allreduce(len(df), op=MPI.SUM)

    if bodo.get_rank() == 0:
        print("Output table checksum: ", sf_df_checksum)
        print("Output table length: ", len_sf_df)

    assert (
        expected_out_len == len_sf_df
    ), f"Expected length ({expected_out_len}) != Table length in Snowflake ({len_sf_df})"

    assert (
        expected_checksum_lower <= sf_df_checksum <= expected_checksum_upper
    ), f"Expected checksum (between {expected_checksum_lower} and {expected_checksum_upper}) != checksum of table in Snowflake ({sf_df_checksum})"

    if args.require_cache and isinstance(
        run_query_func, numba.core.dispatcher.Dispatcher
    ):
        assert (
            run_query_func._cache_hits[run_query_func.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"

    # Drop the table, to avoid dangling tables on our account.
    # This is done on a single rank, and any errors are then
    # broadcasted and raised on all ranks to avoid any hangs.

    drop_err = None
    if bodo.get_rank() == 0:
        try:
            drop_result = drop_sf_table(
                output_table_name,
                conn,
            )
            print("drop_result: ", drop_result)
            assert (
                isinstance(drop_result, list)
                and (len(drop_result) == 1)
                and (len(drop_result[0]) == 1)
                and "successfully dropped" in drop_result[0][0]
            ), "Snowflake DROP table failed, see result above. Might require manual cleanup."
        except Exception as e:
            drop_err = e
    drop_err = comm.bcast(drop_err)
    if isinstance(drop_err, Exception):
        raise drop_err
