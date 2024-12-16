import os
import time
from uuid import uuid4

import numba
import numpy as np
from utils.utils import checksum_str_df_jit, drop_sf_table, get_sf_table

import bodo
import bodosql


@bodo.jit
def checksum(df):
    """
    Compute checksum of output of join query.

    Args:
        df (pd.DataFrame): Dataframe to compute checksum of.

    Returns:
        int: Checksum
    """
    df_no_str = df.drop(
        [
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
            "o_orderpriority",
            "o_comment",
        ],
        axis=1,
    )
    # multiplying by 100 so cent values don't get truncated from the checksum
    df_no_str_sum = np.int64(np.floor(df_no_str.sum().sum() * 100))

    df_str = df[
        ["l_shipinstruct", "l_shipmode", "l_comment", "o_orderpriority", "o_comment"]
    ]
    df_str_sum = checksum_str_df_jit(df_str)

    return df_no_str_sum + df_str_sum


@bodo.jit(spawn=True)
def read_and_compute_checksum_and_len(table_name, conn):
    df = get_sf_table(table_name, conn)
    df_checksum = checksum(df)
    return df_checksum, len(df)


@bodo.jit(cache=True, spawn=True)
def run_query(bc, input_schema, output_table_name):
    query = f"""
        create or replace table {output_table_name} as
        select
            l_orderkey,
            l_extendedprice,
            l_shipinstruct,
            l_shipmode,
            l_comment,
            o_orderpriority,
            o_comment,
            o_custkey
        from
            {input_schema}.lineitem,
            {input_schema}.orders
        where
            l_orderkey = o_orderkey
    """
    bc.sql(query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--require_cache", action="store_true", default=False)
    parser.add_argument("--input_schema", type=str, default="tpch_sf10")
    parser.add_argument("--expected_out_len", type=int)
    parser.add_argument("--expected_checksum_lower", type=int)
    parser.add_argument("--expected_checksum_upper", type=int)
    args = parser.parse_args()
    input_schema = args.input_schema
    expected_out_len = args.expected_out_len
    expected_checksum_lower = args.expected_checksum_lower
    expected_checksum_upper = args.expected_checksum_upper

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

    # Create a random table name to write to and broadcast to all ranks
    output_table_name = None
    output_schema = "public"

    # We use uppercase due to an issue reading tables
    # with lowercase letters.
    # https://bodo.atlassian.net/browse/BE-3534
    output_table_name = (
        f"{output_schema}.hash_join_streaming_e2e_out_{str(uuid4())[:8]}".upper()
    )
    print("output_table_name: ", output_table_name)

    ## Write output to Snowflake table
    start_time = time.time()
    out = run_query(bc, input_schema, output_table_name)
    end_time = time.time()

    # Add a barrier for synchronization purposes so that
    # the get length and drop commands are done after all the writes.
    print("Total time: ", end_time - start_time, " s")

    ## Read table from Snowflake and compute checksum
    conn = f"snowflake://{username}:{password}@{account}/{db}/{output_schema}?warehouse={warehouse}"
    # Compute checksum and length and make sure they're correct
    sf_df_checksum, len_sf_df = read_and_compute_checksum_and_len(
        output_table_name, conn
    )

    print("Output table checksum: ", sf_df_checksum)
    print("Output table length: ", len_sf_df)

    assert (
        expected_out_len == len_sf_df
    ), f"Expected length ({expected_out_len}) != Table length in Snowflake ({len_sf_df})"

    assert (
        expected_checksum_lower <= sf_df_checksum <= expected_checksum_upper
    ), f"Expected checksum (between {expected_checksum_lower} and {expected_checksum_upper}) != checksum of table in Snowflake ({sf_df_checksum})"

    if args.require_cache and isinstance(run_query, numba.core.dispatcher.Dispatcher):
        assert (
            run_query._cache_hits[run_query.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"

    # Drop the table, to avoid dangling tables on our account.
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
