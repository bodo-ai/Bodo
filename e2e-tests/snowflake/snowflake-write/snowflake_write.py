import argparse
import time
from datetime import datetime
from uuid import uuid4

import numba
import numpy as np
import pandas as pd
from mpi4py import MPI
from utils.utils import (
    checksum_str_df,
    drop_sf_table,
    get_sf_table,
    get_sf_write_conn,
)

import bodo

comm = MPI.COMM_WORLD


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
        ["l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode", "l_comment"],
        axis=1,
    )
    # multiplying by 100 so cent values don't get truncatted from the checksum
    df_no_str_sum = np.int64(np.floor(df_no_str.sum().sum() * 100))

    df_str = df[
        ["l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode", "l_comment"]
    ]
    with bodo.objmode(df_str_sum="int64"):
        df_str_sum = checksum_str_df(df_str)

    return df_no_str_sum + df_str_sum


@bodo.jit(cache=True)
def main(read_path, fdate, table_name, sf_conn):
    """
    Snowflake E2E Write test. This reads the lineitem table from S3,
    performs some simple transformations to it, and then writes it to
    a Snowflake table.

    Args:
        read_path (str): S3 URL of the lineitem dataset to read
        fdate (date): Date to perform the filter using (lineitem.L_SHIPDATE <= fdate)
        table_name (str): Name of Snowflake table to write the output to
        sf_conn (str): snowflake connection string

    Returns:
        pd.DataFrame: Dataframe written to Snowflake (for validation)
    """

    # Read lineitem from S3
    t0 = time.time()
    print("Starting read...")
    lineitem = pd.read_parquet(read_path)
    bodo.barrier()
    print("Length of input dataframe: ", len(lineitem))
    print("Read time: ", time.time() - t0)

    # Perform some simple transformations
    t0 = time.time()
    print("Starting transform...")
    sel = lineitem.L_SHIPDATE <= fdate
    flineitem = lineitem[sel]
    flineitem["DISC_PRICE"] = flineitem.L_EXTENDEDPRICE * (1 - flineitem.L_DISCOUNT)
    flineitem["CHARGE"] = (
        flineitem.L_EXTENDEDPRICE * (1 - flineitem.L_DISCOUNT) * (1 + flineitem.L_TAX)
    )
    bodo.barrier()
    print("Length of output dataframe: ", len(flineitem))
    print("Transform time: ", time.time() - t0)

    # Write the transformed dataframe to a Snowflake table
    t0 = time.time()
    print("Starting Snowflake write...")
    flineitem.to_sql(table_name, sf_conn, if_exists="replace", index=False)
    bodo.barrier()
    print("Snowflake Write time: ", time.time() - t0)

    return flineitem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require_cache", action="store_true", default=False)
    args = parser.parse_args()

    # Create a random table name to write to and broadcast to all ranks
    table_name = None
    if comm.Get_rank() == 0:
        # We use uppercase due to an issue reading tables
        # with lowercase letters.
        # https://bodo.atlassian.net/browse/BE-3534
        table_name = f"lineitem_out_{str(uuid4())[:8]}".upper()
    table_name = comm.bcast(table_name)

    # Get Snowflake connection string
    sf_conn = get_sf_write_conn()

    # Perform the e2e test
    t0 = time.time()
    out_df = main(
        "s3://tpch-data-parquet/SF1/lineitem.pq/",
        datetime.strptime("1998-09-02", "%Y-%m-%d").date(),
        table_name,
        sf_conn,
    )
    # Add a barrier for synchronization purposes so that
    # the get length and drop commands are done after all the writes. This is
    # technically not required since the main function
    # has barriers, but still good to have.
    bodo.barrier()
    if bodo.get_rank() == 0:
        print("Total time: ", time.time() - t0, " s")
    if args.require_cache and isinstance(main, numba.core.dispatcher.Dispatcher):
        assert (
            main._cache_hits[main.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"

    if bodo.get_rank() == 0:
        print("Getting table from Snowflake for comparison")
    # Get the table from Snowflake, so we can verify that
    # all data was written as expected.
    sf_df = get_sf_table(table_name, sf_conn)

    # Snowflake returns column names as lowercase, so for consistency
    # we convert all column names to lowercase
    out_df.columns = list(map(str.lower, out_df.columns))
    sf_df.columns = list(map(str.lower, sf_df.columns))

    # Compute lengths and compare
    len_out_df = comm.allreduce(len(out_df), op=MPI.SUM)
    len_sf_df = comm.allreduce(len(sf_df), op=MPI.SUM)

    assert (
        len_out_df == len_sf_df
    ), f"Expected length ({len_out_df}) != Table length in Snowflake ({len_sf_df})"

    # Compute checksum and make sure it's correct
    sf_df_checksum = checksum(sf_df)

    # Checksum can be off-by-one due to rounding issues (e.g. in disc_price and charge columns)
    assert (
        1903874716870990 <= sf_df_checksum <= 1903874716871000
    ), f"Expected checksum (between 1903874716870990 and 1903874716871000) != Checksum of table in Snowflake ({sf_df_checksum})"

    # Drop the table, to avoid dangling tables on our account.
    # This is done on a single rank, and any errors are then
    # broadcasted and raised on all ranks to avoid any hangs.
    drop_err = None
    if bodo.get_rank() == 0:
        try:
            drop_result = drop_sf_table(
                table_name,
                sf_conn,
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
