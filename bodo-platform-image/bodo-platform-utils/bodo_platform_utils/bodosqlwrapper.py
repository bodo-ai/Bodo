# Example usage: python -m bodo_platform_utils.bodosqlwrapper --catalog CATALOG -f query.sql
# To see all options, run: python -m bodo_platform_utils.bodosqlwrapper --help

import argparse
import os
import time
from mpi4py import MPI

import bodo
import bodosql

from .catalog import create_catalog
from .logger import setup_bodo_logger, setup_service_logger
from .utils import read_sql_query, read_sql_query_param
from .query import (
    execute_query,
    compile_query,
    generate_query_plan,
    convert_query_to_pandas,
    get_query_metrics_file,
    check_cache_plan_location,
)

# Set ENV variable for AzureFS authentication to look for credentials
# Otherwise it will use anonymous access by default, only for public buckets
# Must be set before calling `fsspec.open`
os.environ["AZURE_STORAGE_ANON"] = "false"

logger = setup_service_logger()


def main(args):
    setup_bodo_logger(args.verbose_filename)
    is_root_rank = bodo.get_rank() == 0

    if is_root_rank:
        logger.info(f"Read in the query text from the file: {args.filename}")
    query_sql = read_sql_query(args.filename)

    if bodo.get_rank() == 0 and args.query_param_file:
        logger.info(f"Read in the query params from the file: {args.query_param_file}")
    query_params = read_sql_query_param(args.query_param_file)

    if is_root_rank:
        logger.info(f"Get catalog: {args.catalog}")
    bsql_catalog = create_catalog(args)

    if is_root_rank:
        logger.info(f"Create Bodo SQL context: {args.catalog}")
    bc = bodosql.BodoSQLContext(catalog=bsql_catalog)

    # Generate the plan and write it to a file
    generate_query_plan(query_sql, bc, args.generate_plan_filename)

    # Convert to pandas and write to file
    convert_query_to_pandas(query_sql, bc, args.pandas_out_filename)

    # Get query metrics file
    metrics_file = get_query_metrics_file(args.metrics_filename)

    if is_root_rank:
        logger.info(f"Started compiling SQL query")
        mpi_version: str = MPI.Get_library_version()
        logger.info(f"Using MPI Version: {mpi_version.strip()}")

    start_time = time.time()
    dispatcher, compilation_time = compile_query(query_sql, bc, query_params)

    # Check if cache hit match
    cache_hit: bool = dispatcher._cache_hits[dispatcher.signatures[0]] != 0
    if is_root_rank:
        logger.info(f"SQL Binary loaded from cache: {cache_hit}")
        if metrics_file:
            metrics_file.write(
                f"SQL Binary loaded from cache: {cache_hit}\n".encode("utf-8")
            )

    # Get the cache key based on the SQL string
    cache_plan_hit = check_cache_plan_location(query_sql)
    if is_root_rank:
        logger.info(f"SQL Plan loaded from cache: {cache_plan_hit}")
        if metrics_file:
            metrics_file.write(
                f"SQL Plan loaded from cache: {cache_plan_hit}\n".encode("utf-8")
            )

    # Run the query if not compile only
    if not args.compile_only:
        if is_root_rank:
            logger.info("Started executing SQL query")

        result_dir = args.pq_out_filename
        result_stdout = bool(args.print_output)
        trace_filename = args.trace_filename

        execution_time, consume_time = execute_query(
            dispatcher,
            query_sql,
            bc,
            query_params,
            result_dir,
            result_stdout,
            trace_filename,
        )

        if is_root_rank:
            logger.info(f"Execution time: {execution_time} seconds")
            logger.info(f"Consume result time: {consume_time} seconds")

            if metrics_file is not None:
                metrics_file.write(
                    f"Execution time: {float(execution_time)}\n".encode("utf-8")
                )
                metrics_file.write(
                    f"Consume Query Result time: {float(consume_time)}\n".encode(
                        "utf-8"
                    )
                )

    bodo.barrier()  # Wait for all ranks to finish execution
    total_time = time.time() - start_time

    if is_root_rank:
        logger.info(f"Compilation time: {compilation_time} seconds")
        logger.info(f"Total time: {total_time} seconds")

        if metrics_file:
            metrics_file.write(
                f"Compilation time: {float(compilation_time)}\n".encode("utf-8")
            )
            metrics_file.write(f"Total time: {float(total_time)}\n".encode("utf-8"))
            metrics_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BodoSQLWrapper",
        description="Runs SQL queries from files",
    )

    parser.add_argument(
        "-c",
        "--catalog",
        required=True,
        help="Name of the platform catalog to use",
    )
    parser.add_argument(
        "-f", "--filename", required=True, help="Path to .sql file with the query."
    )
    parser.add_argument(
        "-w",
        "--warehouse",
        required=False,
        help="Optional: Snowflake warehouse to use for getting metadata, as well as I/O. When provided, this will override the default value in the credentials file.",
    )
    parser.add_argument(
        "-d",
        "--database",
        required=False,
        help="Optional: Snowflake Database which has the required tables. When provided, this will override the default value in the credentials file.",
    )
    parser.add_argument(
        "--schema",
        required=False,
        help="Optional: Snowflake Schema (within the database) which has the required tables. When provided, this will override the default value in the credentials file.",
    )
    parser.add_argument(
        "-o",
        "--pq_out_filename",
        required=False,
        help="Optional: Write the query output as a parquet dataset to this location.",
    )
    parser.add_argument(
        "-p",
        "--pandas_out_filename",
        required=False,
        help="Optional: Write the pandas code generated from the SQL query to this location.",
    )
    parser.add_argument(
        "-t",
        "--trace_filename",
        required=False,
        help="Optional: If provided, the tracing will be used and the trace file will be written to this location",
    )
    parser.add_argument(
        "-g",
        "--generate_plan_filename",
        required=False,
        help="Optional: Write the SQL plan to this location.",
    )
    parser.add_argument(
        "-u",
        "--print_output",
        required=False,
        action="store_true",
        help="Optional: If provided, the result will printed to std. Useful when testing and don't necessarily want to save results.",
    )
    parser.add_argument(
        "-m",
        "--metrics_filename",
        required=False,
        help="Optional: If provided, Write the metrics logs to this location.",
    )
    parser.add_argument(
        "-v",
        "--verbose_filename",
        required=False,
        help="Optional: If provided, verbose logs will be written to this location.",
    )
    parser.add_argument(
        "-co",
        "--compile_only",
        required=False,
        action="store_true",
        help="Optional: If provided, the query will be compiled and the execution will be skipped.",
    )
    parser.add_argument(
        "--iceberg_volume",
        required=False,
        default=None,
        help="Optional: Iceberg volume to use for writing as an iceberg table",
    )
    parser.add_argument(
        "--iceberg_rest_url",
        required=False,
        default=None,
        help="Optional: url for iceberg rest API server",
    )

    parser.add_argument(
        "-qpf",
        "--query-param-file",
        required=False,
        help="Path to .json file with the query params",
    )

    args = parser.parse_args()
    main(args)
