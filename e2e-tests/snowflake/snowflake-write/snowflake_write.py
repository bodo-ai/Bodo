import argparse
import os
import time
from datetime import datetime
from uuid import uuid4

import numba
import numpy as np
import pandas as pd

import bodo
import bodo.io.snowflake
from bodo.mpi4py import MPI

bodo.set_verbose_level(2)


def checksum_str_df(df):
    """
    Compute checksum of the a DataFrame with all string columns.
    We sum up the ascii encoding and compute modulo 256 for
    each element and then add it up across all processes.

    Args:
        df (pd.DataFrame): DataFrame with all string columns

    Returns:
        int64: Checksum
    """
    comm = MPI.COMM_WORLD
    df_hash = df.map(lambda x: sum(x.encode("ascii")) % 256)
    str_sum = np.int64(df_hash.sum().sum())
    str_sum = comm.allreduce(str_sum, op=MPI.SUM)
    return np.int64(str_sum)


@bodo.jit
def checksum_str_df_jit(df):
    """JIT version of above"""
    str_sum = 0
    for c in df.columns:
        str_sum += df[c].str.encode("ascii").map(lambda x: sum(x) % 256).sum()
    return str_sum


@bodo.jit
def get_sf_table(table_name, sf_conn):
    """
    Get length of Snowflake table.

    Args:
        table_name (str): name of table
        sf_conn (str): snowflake connection string

    Returns:
        pd.DataFrame: Snowflake Table
    """
    df = pd.read_sql(f"select * from {table_name}", sf_conn)
    return df


def drop_sf_table(table_name, sf_conn):
    """
    Drop a Snowflake Table.

    Args:
        table_name (str): name of table
        sf_conn (str): snowflake connection string

    Returns:
        list: list of results from the drop command
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(sf_conn)
    connection = engine.connect()
    result = connection.execute(text(f"drop table {table_name}"))
    result = result.fetchall()
    connection.close()
    engine.dispose()

    return result


def get_sf_write_conn(user=1):
    """
    Get Snowflake connection string of the form:
    "snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"

    When user=1, we use an AWS based account
        username is derived from the SF_USERNAME environment variable
        password is derived from the SF_PASSWORD environment variable
        account is derived from the SF_ACCOUNT environment variable
    When user=2, we use an Azure based account
        username is derived from the SF_AZURE_USER environment variable
        password is derived from the SF_AZURE_PASSWORD environment variable
        account is derived from the SF_AZURE_ACCOUNT environment variable

    database is E2E_TESTS_DB
    schema is 'public'
    warehouse is DEMO_WH

    Returns:
        str: snowflake connection string
    """
    if user == 1:
        username = os.environ["SF_USERNAME"]
        password = os.environ["SF_PASSWORD"]
        account = os.environ["SF_ACCOUNT"]
    elif user == 2:
        username = os.environ["SF_AZURE_USER"]
        password = os.environ["SF_AZURE_PASSWORD"]
        account = os.environ["SF_AZURE_ACCOUNT"]
    else:
        raise ValueError(
            f"Unsupported user value {user} for get_sf_write_conn. Supported values: [1, 2]"
        )
    database = "E2E_TESTS_DB"
    schema = "public"
    warehouse = "DEMO_WH"
    return f"snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


@bodo.jit(spawn=True)
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

    df_str_sum = checksum_str_df_jit(df_str)

    return df_no_str_sum + df_str_sum


@bodo.jit(spawn=True)
def read_and_compute_checksum_and_len(table_name, conn):
    df = get_sf_table(table_name, conn)
    df_checksum = checksum(df)
    return df_checksum, len(df)


@bodo.jit(cache=True, spawn=True)
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

    return len(flineitem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user", type=int)
    parser.add_argument("--use_put_method", action="store_true", default=False)
    parser.add_argument("--require_cache", action="store_true", default=False)
    args = parser.parse_args()

    # Create a random table name to write to and broadcast to all ranks
    table_name = None
    # We use uppercase due to an issue reading tables
    # with lowercase letters.
    # https://bodo.atlassian.net/browse/BE-3534
    table_name = f"lineitem_out_{str(uuid4())[:8]}".upper()

    # Get Snowflake connection string based on the user.
    # user=1 is an AWS based account, and user=2 is an Azure based account.
    sf_conn = get_sf_write_conn(args.user)

    if args.use_put_method:
        # Force PUT method for testing purposes
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = True

    # Perform the e2e test
    t0 = time.time()
    len_out_df = main(
        "s3://tpch-data-parquet/SF1/lineitem.pq/",
        datetime.strptime("1998-09-02", "%Y-%m-%d").date(),
        table_name,
        sf_conn,
    )

    print("Total time: ", time.time() - t0, " s")
    if args.require_cache and isinstance(main, numba.core.dispatcher.Dispatcher):
        assert (
            main._cache_hits[main.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"

    print("Getting table from Snowflake for comparison")
    # Get the table from Snowflake, so we can verify that
    # all data was written as expected.

    # Compute lengths and compare
    # Compute checksum and make sure it's correct
    sf_df_checksum, len_sf_df = read_and_compute_checksum_and_len(table_name, sf_conn)

    assert (
        len_out_df == len_sf_df
    ), f"Expected length ({len_out_df}) != Table length in Snowflake ({len_sf_df})"

    # Checksum can be off-by-one due to rounding issues (e.g. in disc_price and charge columns)
    assert (
        1903874716870990 <= sf_df_checksum <= 1903874716871000
    ), f"Expected checksum (between 1903874716870990 and 1903874716871000) != Checksum of table in Snowflake ({sf_df_checksum})"

    # Drop the table, to avoid dangling tables on our account.
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
