import os
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd
from mpi4py import MPI

import bodo


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


def run_cmd(
    cmd,
    print_output=True,
    timeout=3600,
    additional_envs: dict[str, str] | None = None,
):
    # TODO: specify a timeout to check_output or will the CI handle this? (e.g.
    # situations where Bodo hangs for some reason)
    # stderr=subprocess.STDOUT to also capture stderr in the result
    try:
        additional_envs = {} if additional_envs is None else additional_envs
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            env=dict(os.environ, **additional_envs),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(e.output)
        raise
    if print_output:
        print(output)
    return output


def update_env_vars(env_vars):
    """Update the current environment variables with key-value pairs provided
    in a dictionary.

    Args
        env_vars (Dict(str, str or None)): A dictionary of environment variables to set.
            A value of None indicates a variable should be removed.

    Returns
        old_env_vars (Dict(str, str or None)): Previous value of any overwritten
            environment variables. A value of None indicates an environment
            variable was previously unset.
    """
    old_env_vars = {}
    for k, v in env_vars.items():
        if k in os.environ:
            old_env_vars[k] = os.environ[k]
        else:
            old_env_vars[k] = None

        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v
    return old_env_vars


@contextmanager
def temp_env_override(env_vars):
    """Update the current environment variables with key-value pairs provided
    in a dictionary and then restore it after.

    Args
        env_vars (Dict(str, str or None)): A dictionary of environment variables to set.
            A value of None indicates a variable should be removed.
    """

    old_env = {}
    try:
        old_env = update_env_vars(env_vars)
        yield
    finally:
        update_env_vars(old_env)
