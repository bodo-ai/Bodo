import argparse
import io
import time
from uuid import uuid4

import numba
import numpy as np
import pandas as pd
from mpi4py import MPI
from utils.utils import (
    checksum_str_df_jit,
    drop_sf_table,
    get_sf_read_conn,
    get_sf_table,
    get_sf_write_conn,
)

import bodo
from bodo.tests.user_logging_utils import (
    create_string_io_logger,
    set_logging_stream,
)

comm = MPI.COMM_WORLD

desired_out = [427576, 427846, 427868, 428616, 428976, 429176, 429458]


@bodo.jit
def checksum(df):
    """
    Compute checksum of lineitem dataframe.

    Args:
        df (pd.DataFrame): Dataframe to compute checksum of.

    Returns:
        int: Checksum
    """
    df_no_str = df.drop(
        ["l_shipinstruct", "l_shipmode", "l_comment", "paded_mode"],
        axis=1,
    )
    # multiplying by 100 so cent values don't get truncated from the checksum
    df_no_str_sum = np.int64(np.floor(df_no_str.sum().sum() * 100))

    df_str = df[["l_shipinstruct", "l_shipmode", "l_comment", "paded_mode"]]

    df_str_sum = checksum_str_df_jit(df_str)

    return df_no_str_sum + df_str_sum


@bodo.jit(spawn=True)
def read_and_compute_checksum_and_len(table_name, conn):
    df = get_sf_table(table_name, conn)
    df_checksum = checksum(df)
    return df_checksum, len(df)


bodo.set_verbose_level(1)


@bodo.jit(cache=True, spawn=True)
def main(sf_read_conn, sf_write_conn, table_name):
    """
    Snowflake E2E read test. We read in three string columns from TPCH-SF1 on Snowflake,
    perform some groupby and string functions to the DataFrame, and write the transformed
    data frame back to snowflake.

    Args:
        sf_conn (str): snowflake connection string
    """
    query = "SELECT l_shipmode, l_shipinstruct, l_comment FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY"
    # Load in the data
    t0 = time.time()
    print("Starting read...")
    df = pd.read_sql(
        query, sf_read_conn, _bodo_read_as_dict=["l_shipmode", "l_shipinstruct"]
    )
    bodo.barrier()
    print("Length of input dataframe: ", len(df))
    print("Read time: ", time.time() - t0)

    # Perform simple series.str operation on string columns
    t1 = time.time()
    df["a_count"] = df["l_shipinstruct"].str.count("A")
    df["paded_mode"] = df["l_shipmode"].str.ljust(10)
    bodo.barrier()
    print("string operations(count & ljust) time: ", time.time() - t1)

    # Perform simple transformations
    # the transformations bear little meaning, and
    # they serve only as a means to demonstrate our ability
    # to carry out operations on dictionary-encoded arrays.
    t2 = time.time()
    print("Starting groupby transformation...")
    gb = df.groupby(["l_shipmode"])
    bodo.barrier()
    print("Groupby time: ", time.time() - t2)
    print("Number of groups: ", gb.size())

    t3 = time.time()
    agg_strs = gb["l_shipinstruct"].agg(["min", "nunique"])
    bodo.barrier()
    print("Agg time: ", time.time() - t3)
    print("Number of unique l_shipinstruct for each group, sorted from high to low:")
    print("nunique:\n", agg_strs["nunique"])

    # The summation is for testing correctness only and is
    # not related to dictionary encoding
    agg_counts = gb["a_count"].sum()

    # Write the transformed dataframe to a Snowflake table.
    t4 = time.time()
    print("Starting Snowflake write...")
    df.to_sql(table_name, sf_write_conn, if_exists="replace", index=False)
    bodo.barrier()
    print("Snowflake Write time: ", time.time() - t4)

    return agg_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require_cache", action="store_true", default=False)
    args = parser.parse_args()

    # Create a random table name to write to and broadcast to all ranks
    table_name = None
    # We use uppercase due to an issue reading tables
    # with lowercase letters.
    # https://bodo.atlassian.net/browse/BE-3534
    table_name = f"lineitem_out_{str(uuid4())[:8]}".upper()
    print("table_name: ", table_name)

    # Get Snowflake connection strings
    sf_read_conn = get_sf_read_conn()
    sf_write_conn = get_sf_write_conn()

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        # Perform the e2e test
        t0 = time.time()
        agg_counts = main(sf_read_conn, sf_write_conn, table_name)
        # Print the stream for debugging purposes
        print(stream.getvalue())
        # The logging messages are not shown when loading from cache
        # TODO BSE-4167: Send logger to the workers
        # if not args.require_cache:
        #     # Validate that the columns are loaded with dictionary encoding
        #     check_logger_msg(
        #         stream,
        #         "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
        #     )
    sorted_a_count = sorted(agg_counts.tolist())

    # Add a barrier for synchronization purposes so that
    # the get length and drop commands are done after all the writes. This is
    # technically not required since the main function
    # has barriers, but still good to have.
    print("Total time: ", time.time() - t0, " s")

    # Compare our output to the desired output
    assert sorted_a_count == desired_out, (
        f"Expected out {desired_out} != output calculated from dictionary-encoded arrays from Snowflake {sorted_a_count}"
    )

    # Verify that cache was used
    if args.require_cache and isinstance(main, numba.core.dispatcher.Dispatcher):
        assert main._cache_hits[main.signatures[0]] == 1, (
            "ERROR: Bodo did not load from cache"
        )

    # Compute checksum and length and make sure they're correct
    sf_df_checksum, len_sf_df = read_and_compute_checksum_and_len(
        table_name, sf_write_conn
    )

    assert 6001215 == len_sf_df, (
        f"Expected length (6001215) != Table length in Snowflake ({len_sf_df})"
    )

    assert 3716926710 <= sf_df_checksum <= 3716926730, (
        f"Expected checksum (between 3716926710 and 3716926730) != checksum of table in Snowflake ({sf_df_checksum})"
    )

    # Drop the table, to avoid dangling tables on our account.
    drop_result = drop_sf_table(
        table_name,
        sf_write_conn,
    )
    print("drop_result: ", drop_result)
    assert (
        isinstance(drop_result, list)
        and (len(drop_result) == 1)
        and (len(drop_result[0]) == 1)
        and "successfully dropped" in drop_result[0][0]
    ), "Snowflake DROP table failed, see result above. Might require manual cleanup."
