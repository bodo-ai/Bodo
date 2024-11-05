# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Attempt to fetch all of the tables used by BodoSQL. This works by calling
generate_plan() on each query and then relying on logging to output
which tables were accessed. This script also records which files successfully
validated + created a plan and which files failed to do so.

Note: This script currently runs the plan optimizations since we removed the
unoptimized API. If we want to use this again we should consider exposing a
validate query instead.
"""

import argparse
import json
import os

import bodo
import bodosql


def timeout(func, args=(), timeout_duration=300):
    # Timeout function influenced from stack overflow.
    # https://stackoverflow.com/a/13821695
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args)
    except TimeoutError as exc:
        print("REACHED A TIMEOUT")
        raise exc
    finally:
        signal.alarm(0)

    return result


def check_validation(bc, sql_text: str, filename: str):
    """Attempt to validate a query by running bc.generate_plan(sql_text).
    Logs if a file succeeds or fails.

    Args:
        bc (BodoSQLContext): The BodoSQLContext
        sql_text (str): _description_
        filename (str): The filename for the query. Used for logging success/failure.
    """
    try:
        timeout(bc.generate_plan, (sql_text,))
        print(f"Successfully validated query file: {filename}")
    except Exception as e:
        print(f"Encountered error: {str(e)}")
        print(f"Failed to validate query file: {filename}")


def main(args):
    # Fetch and create catalog
    with open(args.catalog_creds) as f:
        catalog = json.load(f)

    username = catalog["SF_USERNAME"]
    password = catalog["SF_PASSWORD"]
    account = catalog["SF_ACCOUNT"]
    warehouse = catalog["SF_WAREHOUSE"]
    database = catalog["SF_DATABASE"]

    snowflake_catalog = bodosql.SnowflakeCatalog(
        username=username,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
    )
    # Set the verbose level to 3
    bodo.set_verbose_level(3)

    # Create context
    bc = bodosql.BodoSQLContext(catalog=snowflake_catalog)
    query_dir = args.dir
    filenames = os.listdir(query_dir)
    # Sort the filenames to enable batching
    filenames = sorted(filenames)
    # Determine the slice to investigate
    start = args.offset
    end = args.batch_size
    filenames_batch = filenames[start : start + end]
    for query_filename in filenames_batch:
        # Read in the query text from the file
        full_path = f"{query_dir}/{query_filename}"
        with open(full_path) as f:
            sql_text = f.read()
            check_validation(bc, sql_text, query_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BodoSQLFetchTables",
        description="Attempt to compute generate_plan() for each query file to find the tables used",
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="Path to parquet file containing data.",
    )
    parser.add_argument(
        "-c",
        "--catalog_creds",
        required=True,
        help="Path to Snowflake credentials file. The following keys must be present: SF_USERNAME, SF_PASSWORD and SF_ACCOUNT. The following keys are optional: SF_WAREHOUSE, SF_DATABASE",
    )
    parser.add_argument(
        "-o",
        "--offset",
        required=True,
        type=int,
        help="Offset into the sorted directory file list to start fetching queries. This is to enable mini-batching in case a query hangs.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=True,
        type=int,
        help="Maximum number of queries to process in 1 batch.",
    )

    args = parser.parse_args()
    main(args)
