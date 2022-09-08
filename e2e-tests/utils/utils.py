import os
import subprocess

import numpy as np
import pandas as pd
from mpi4py import MPI

import bodo


def checksum_str_df(df):
    """
    Compute checksum of the a dataframe with all string columns.
    We sum up the ascii encoding and compute modulo 256 for
    each element and then add it up across all processes.

    Args:
        df (pd.DataFrame): DataFrame with all string columns

    Returns:
        int64: Checksum
    """
    comm = MPI.COMM_WORLD
    df_hash = df.applymap(lambda x: sum(x.encode("ascii")) % 256)
    str_sum = np.int64(df_hash.sum().sum())
    str_sum = comm.allreduce(str_sum, op=MPI.SUM)
    return np.int64(str_sum)


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
    from sqlalchemy import create_engine

    engine = create_engine(sf_conn)
    connection = engine.connect()
    result = connection.execute(f"drop table {table_name}")
    result = result.fetchall()
    return result


def get_sf_write_conn():
    """
    Get Snowflake connection string of the form:
    "snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"

    username is derived from the SF_USERNAME environment variable
    password is derived from the SF_PASSWORD environment variable
    account is derived from the SF_ACCOUNT environment variable
    database is E2E_TESTS_DB
    schema is 'public'
    warehouse is DEMO_WH

    Returns:
        str: snowflake connection string
    """
    username = os.environ["SF_USERNAME"]
    password = os.environ["SF_PASSWORD"]
    account = os.environ["SF_ACCOUNT"]
    database = "E2E_TESTS_DB"
    schema = "public"
    warehouse = "DEMO_WH"
    return f"snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


def get_sf_read_conn():
    """
    Get Snowflake connection string of the form:
    "snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"

    username is derived from the SF_USERNAME environment variable
    password is derived from the SF_PASSWORD environment variable
    account is derived from the SF_ACCOUNT environment variable
    database is SNOWFLAKE_SAMPLE_DATA
    schema is TPCH_SF1
    warehouse is DEMO_WH

    Returns:
        str: snowflake connection string
    """
    username = os.environ["SF_USERNAME"]
    password = os.environ["SF_PASSWORD"]
    account = os.environ["SF_ACCOUNT"]
    database = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    warehouse = "DEMO_WH"
    return f"snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


def run_cmd(cmd, print_output=True):
    # TODO: specify a timeout to check_output or will the CI handle this? (e.g.
    # situations where Bodo hangs for some reason)
    # stderr=subprocess.STDOUT to also capture stderr in the result
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, errors="replace"
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    if print_output:
        print(output)
    return output
