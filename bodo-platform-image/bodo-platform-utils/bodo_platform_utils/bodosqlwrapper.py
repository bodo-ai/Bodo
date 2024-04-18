# Example usage: python -m bodo_platform_utils.bodosqlwrapper --catalog CATALOG -f query.sql
# To see all options, run: python -m bodo_platform_utils.bodosqlwrapper --help

import argparse
import os
import requests
import time
from urllib.parse import urlencode
import logging
from enum import Enum

import bodo
import bodosql
from bodosql.context_ext import typeof_bodo_sql
import bodo.utils.tracing as tracing
from bodo.sql_plan_cache import BodoSqlPlanCache
import numba

from .catalog import get_data

# constants
AUTH_URL_SUFFIX = "/v1/oauth/tokens"

class CatalogType(Enum):
    SNOWFLAKE = "SNOWFLAKE"
    TABULAR = "TABULAR"


# Turn verbose mode on
bodo.set_verbose_level(2)
bodo_logger = bodo.user_logging.get_current_bodo_verbose_logger()


def run_sql_query(
    query_str,
    bc,
):
    """Boilerplate function to execute a query string.

    Args:
        query_str (str): Query text to execute
        bc (bodosql.BodoSQLContext): BodoSQLContext to use for query execution.
    """

    print(f"Started executing query:\n{query_str}")
    t0 = time.time()
    output = bc.sql(query_str)
    execution_time = time.time() - t0
    print(f"Finished executing the query. It took {execution_time} seconds.")
    return output, execution_time


@bodo.jit(cache=True)
def consume_query_result(
    output, pq_out_filename, print_output
):
    """Function to consume the query result.
    Args:
        pq_out_filename (str): When provided (i.e. not ''), the query output is written to this location as a parquet file.
        print_output: Flag to print query result.
    """
    print("Output Shape: ", output.shape)
    if print_output:
        print("Output:")
        print(output)
    if pq_out_filename != "":
        print("Saving output as parquet dataset to: ", pq_out_filename)
        t0 = time.time()
        output.to_parquet(pq_out_filename)
        print(f"Finished parquet write. It took {time.time() - t0} seconds.")


def get_cache_loc_from_dispatcher(dispatcher) -> str:
    """
    Get the location of the cached binary from the dispatcher
    object of a function.
    In case we aren't able to get the location, None
    will be returned.

    Args:
        dispatcher: Dispatcher function for the query.
    """
    try:
        cache_dir = dispatcher.stats.cache_path
        cache_key = dispatcher._cache._index_key(
            dispatcher.signatures[0], dispatcher.targetctx.codegen()
        )
        cache_file_name = dispatcher._cache._cache_file._load_index().get(cache_key)
        return os.path.join(cache_dir, cache_file_name)
    except Exception:
        return None


def run_sql_query_wrapper(
    dispatcher,
    sql_text,
    bc,
    print_output,
    write_metrics,
    args,
    metrics_file,
):
    """
    Wrapper function to run the query and consume the result.
    Args:
        dispatcher: Dispatcher function for the query.
        sql_text (str): Query text to execute
        bc (bodosql.BodoSQLContext): BodoSQLContext to use for query execution.
        print_output(bool): Flag to print query result.
        write_metrics(bool): Flag to write metrics.
        args: Arguments passed to the script.
        metrics_file(Union(File, None)): File to write metrics to.
    """
    if args.trace:
        tracing.start()
    output, execution_time = dispatcher(
        numba.types.literal(sql_text),
        bc,
    )
    if write_metrics:
        metrics_file.write(f"Execution time: {float(execution_time)}\n".encode("utf-8"))

    if output is not None:
        t_cqr = time.time()
        consume_query_result(
            output,
            args.pq_out_filename if args.pq_out_filename else "",
            print_output,
        )

        bodo.barrier()
        cqr_time = time.time() - t_cqr
        if write_metrics:
            metrics_file.write(
                f"Consume Query Result time: {float(cqr_time)}\n".encode("utf-8")
            )
    if args.trace:
        tracing.dump(fname=args.trace)


def create_snowflake_catalog(catalog, args):
    """
    Create the Snowflake catalog from the catalog data.
    Args:
        catalog: Catalog object from the secret store.
    """
    warehouse = args.warehouse if args.warehouse else catalog.get("warehouse")
    if warehouse is None:
        raise ValueError(
            "No warehouse specified in either the catalog data or through the arguments."
        )

    database = args.database if args.database else catalog.get("database")
    if database is None:
        raise ValueError(
            "No database specified in either the catalog data or through the arguments."
        )

    # Schema can be None for backwards compatibility
    schema = args.schema if args.schema else catalog.get("schema")

    iceberg_volume = args.iceberg_volume if args.iceberg_volume else catalog.get("icebergVolume")

    # Create connection params
    connection_params = {"role": catalog["role"]} if "role" in catalog else {}
    if schema is not None:
        connection_params["schema"] = schema

    return bodosql.SnowflakeCatalog(
        username=catalog["username"],
        password=catalog["password"],
        account=catalog["accountName"],
        warehouse=warehouse,
        database=database,
        connection_params=connection_params,
        iceberg_volume=iceberg_volume,
    )


def create_tabular_catalog(catalog, args):
    """
    Create the tabular catalog from the catalog data.
    Args:
        catalog: Catalog object from the secret store.
    """
    warehouse = args.warehouse if args.warehouse else catalog.get("warehouse")
    if warehouse is None:
        raise ValueError(
            "No warehouse specified in either the catalog data or through the arguments."
        )
    
    iceberg_rest_url = args.iceberg_rest_url if args.iceberg_rest_url else catalog.get("icebergRestUrl")
    if iceberg_rest_url is None:
        raise ValueError(
            "No icebergRestUrl specified in either the catalog data or through the arguments."
        )

    client_id = catalog.get("clientId")
    if client_id is None:
        raise ValueError(
            "No clientId specified in the catalog data."
        )
    
    client_secret = catalog.get("clientSecret")
    if client_secret is None:
        raise ValueError(
            "No clientSecret specified in the catalog data."
        )

    # Gets a user access token
    auth_url = iceberg_rest_url + AUTH_URL_SUFFIX
    oauth_response = requests.post(
        auth_url,
        data={'client_id': client_id, 'client_secret': client_secret, 'grant_type': 'client_credentials'},
        headers={'Content-Type': 'application/x-www-form-urlencoded'})
    user_session_token = oauth_response.json()['access_token']

    return bodosql.TabularCatalog(
        warehouse=warehouse,
        iceberg_rest_url=iceberg_rest_url,
        user_session_token=user_session_token,
    )


def main(args):
    if args.verbose_filename:
        # Write verbose logs to the file
        metrics_handler = logging.FileHandler(args.verbose_filename, mode="w")
        bodo_logger.addHandler(metrics_handler)

    # Read in the query text from the file
    with open(args.filename, "r") as f:
        sql_text = f.read()

    # Fetch and create catalog
    catalog = get_data(args.catalog)
    if catalog is None:
        raise ValueError("Catalog not found in the secret store.")
    
    catalog_type_str = catalog.get("catalogType")
    if catalog_type_str is None:
        catalog_type = CatalogType.SNOWFLAKE
    else:
        catalog_type = CatalogType(catalog_type_str)

    if catalog_type == CatalogType.TABULAR:
        bsql_catalog = create_tabular_catalog(catalog, args)
    else:
        # default to Snowflake for backward compatibility
        bsql_catalog = create_snowflake_catalog(catalog, args)

    # Create context
    bc = bodosql.BodoSQLContext(catalog=bsql_catalog)

    # Generate the plan and write it to a file
    if args.generate_plan_filename:
        plan_text = bc.generate_plan(sql_text, show_cost=True)
        if bodo.get_rank() == 0:
            with open(args.generate_plan_filename, "w") as f:
                f.write(plan_text)
            print("[INFO] Saved Plan to: ", args.generate_plan_filename)

    # Convert to pandas and write to file
    if args.pandas_out_filename:
        pandas_text = bc.convert_to_pandas(sql_text)
        if bodo.get_rank() == 0:
            with open(args.pandas_out_filename, "w") as f:
                f.write(pandas_text)
            print("[INFO] Saved Pandas Version to: ", args.pandas_out_filename)

    print_output = False
    if args.print_output:
        print_output = True

    write_metrics = args.metrics_filename is not None and bodo.get_rank() == 0
    metrics_file = (
        open(args.metrics_filename, "wb", buffering=0) if write_metrics else None
    )
    # Compile the query
    t0 = time.time()
    dispatcher = bodo.jit(
        (numba.types.literal(sql_text), typeof_bodo_sql(bc, None)), cache=True
    )(run_sql_query)
    compilation_time = time.time() - t0
    bodo.barrier()  # Wait for all ranks to finish compilation

    cache_hit: bool = dispatcher._cache_hits[dispatcher.signatures[0]] != 0
    if write_metrics:
        metrics_file.write(
            f"Compilation time: {float(compilation_time)}\n".encode("utf-8")
        )
        metrics_file.write(f"Ran from cache: {cache_hit}\n".encode("utf-8"))

    cache_loc = get_cache_loc_from_dispatcher(dispatcher)
    if cache_loc and (bodo.get_rank() == 0):
        print(
            "[INFO] Binary {} {}".format(
                "loaded from" if cache_hit else "saved to",
                cache_loc,
            )
        )

    # Get the cache key based on the sql string
    plan_location: str | None = BodoSqlPlanCache.get_cache_loc(sql_text)
    if plan_location and (bodo.get_rank() == 0):
        if os.path.isfile(plan_location):
            print(f"[INFO] SQL Plan cached at {plan_location}.")
        else:
            print(
                f"[WARN] Expected SQL Plan to be cached at {plan_location}, but it wasn't found."
            )

    # Run the query if not compile only
    if not args.compile_only:
        run_sql_query_wrapper(
            dispatcher,
            sql_text,
            bc,
            print_output,
            write_metrics,
            args,
            metrics_file,
        )

    bodo.barrier()  # Wait for all ranks to finish execution
    total_time = time.time() - t0
    if write_metrics:
        metrics_file.write(f"Total time: {float(total_time)}\n".encode("utf-8"))
        metrics_file.close()

    if bodo.get_rank() == 0:
        print("Total (compilation + execution) time:", total_time)


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
        "--trace",
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

    args = parser.parse_args()

    main(args)
