# Example usage: python -m bodo_platform_utils.bodosqlwrapper --catalog CATALOG -f query.sql
# To see all options, run: python -m bodo_platform_utils.bodosqlwrapper --help

import argparse
import os
import requests
import time
import logging
import datetime
from enum import Enum
import json
import numpy as np
import pandas as pd
import typing as pt
import pyarrow as pa
from mpi4py import MPI

import bodo
import bodosql
from bodosql.context_ext import typeof_bodo_sql
import bodo.utils.tracing as tracing
from bodo.sql_plan_cache import BodoSqlPlanCache
import numba

from .catalog import get_data
from .type_convertor import get_value_for_type

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
def consume_query_result(output, pq_out_filename, print_output):
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


def get_cache_loc_from_dispatcher(dispatcher) -> pt.Optional[str]:
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


def parse_output_types(output: "pd.DataFrame") -> pt.List[pt.Dict]:
    """
    Parse the output types of the query result
    Args:
        output: Query result dataframe
    Return:
        List of dictionaries containing the column name, type and nullable status
    """

    def _get_col_type(col_name: str, col_dtype) -> pt.Tuple[str, bool, pt.Dict]:
        """
        Inner function to get the column type for specific columns

        Returns:
            Tuple of
                - Column type
                - Nullable status
                - Dictionary of additional type information. For example, timezone
                  for timestamp with timezone columns. Otherwise None
        """
        # Special handling for object type
        # TODO: Remove this after removing all Bodo object boxing
        if col_dtype == np.dtype("O"):
            first_val = output[col_name][0]
            if isinstance(first_val, str):
                return ("STRING", True, {})
            if isinstance(first_val, bytes):
                return ("BINARY", True, {})
            if isinstance(first_val, datetime.date):
                return ("DATE", True, {})
            if isinstance(first_val, datetime.datetime):
                return ("TIMESTAMP_NTZ_NS", True, {})
            if isinstance(first_val, datetime.time):
                return ("TIME", True, {})
            raise ValueError(
                f"Unsupported object type for column {col_name}: {type(first_val)}"
            )

        # Special PyArrow-based Arrays
        if isinstance(col_dtype, pd.ArrowDtype):
            pa_type = col_dtype.pyarrow_dtype
            if pa.types.is_boolean(pa_type):
                return ("BOOL", True, {})
            if pa.types.is_int8(pa_type):
                return ("INT8", True, {})
            if pa.types.is_int16(pa_type):
                return ("INT16", True, {})
            if pa.types.is_int32(pa_type):
                return ("INT32", True, {})
            if pa.types.is_int64(pa_type):
                return ("INT64", True, {})
            if pa.types.is_float32(pa_type):
                return ("FLOAT32", True, {})
            if pa.types.is_float64(pa_type):
                return ("FLOAT64", True, {})
            if pa.types.is_date32(pa_type):
                return ("DATE", True, {})
            if pa.types.is_decimal128(pa_type):
                return (f"DECIMAL({pa_type.precision}, {pa_type.scale})", True, {})
            if pa.types.is_large_binary(pa_type) or pa.types.is_binary(pa_type):
                return ("BINARY", True, {})
            if pa.types.is_large_string(pa_type) or pa.types.is_string(pa_type):
                return ("STRING", True, {})
            if pa.types.is_time32(pa_type):
                return (f"TIME_{pa_type.unit.upper()}", True, {})
            if pa.types.is_time64(pa_type):
                return (f"TIME_{pa_type.unit.upper()}", True, {})
            if pa.types.is_null(pa_type):
                return ("NULL", True, {})
            if pa.types.is_large_list(pa_type) or pa.types.is_list(pa_type):
                return ("ARRAY", True, {})
            if pa.types.is_map(pa_type):
                return ("MAP", True, {})
            if pa.types.is_struct(pa_type):
                return ("STRUCT", True, {})
            if pa.types.is_timestamp(pa_type):
                if pa_type.tz is None:
                    return (f"TIMESTAMP_NTZ_{pa_type.unit.upper()}", True, {})
                else:
                    return (
                        f"TIMESTAMP_LTZ_{pa_type.unit.upper()}",
                        True,
                        {"timezone": pa_type.tz},
                    )

            raise ValueError(
                f"Unsupported PyArrow dtype {pa_type} for column {col_name}"
            )

        # Numpy and Nullable dtypes
        # Integers
        if col_dtype == np.int8:
            return ("INT8", False, {})
        if col_dtype == np.int16:
            return ("INT16", False, {})
        if col_dtype == np.int32:
            return ("INT32", False, {})
        if col_dtype == np.int64:
            return ("INT64", False, {})
        if isinstance(col_dtype, pd.Int8Dtype):
            return ("INT8", True, {})
        if isinstance(col_dtype, pd.Int16Dtype):
            return ("INT16", True, {})
        if isinstance(col_dtype, pd.Int32Dtype):
            return ("INT32", True, {})
        if isinstance(col_dtype, pd.Int64Dtype):
            return ("INT64", True, {})

        # Floats
        if col_dtype == np.float32:
            return ("FLOAT32", False, {})
        if col_dtype == np.float64:
            return ("FLOAT64", False, {})
        if isinstance(col_dtype, pd.Float32Dtype):
            return ("FLOAT32", True, {})
        if isinstance(col_dtype, pd.Float64Dtype):
            return ("FLOAT64", True, {})

        # Booleans
        if col_dtype == np.bool_:
            return ("BOOL", False, {})
        if isinstance(col_dtype, pd.BooleanDtype):
            return ("BOOL", True, {})

        # Datetime
        if col_dtype == np.datetime64:
            return ("TIMESTAMP_NTZ_NS", False, {})
        if isinstance(col_dtype, pd.DatetimeTZDtype):
            return (
                f"TIMESTAMP_LTZ_{col_dtype.unit.upper()}",
                True,
                {"timezone": col_dtype.tz},
            )

        # Strings
        if isinstance(col_dtype, pd.StringDtype):
            return ("STRING", True, {})

        raise ValueError(f"Unsupported dtype {col_dtype} for column {col_name}")

    output_types = [
        (col_name, *_get_col_type(col_name, col_type))
        for col_name, col_type in output.dtypes.iter()
    ]
    return [{"name": a, "type": b, "nullable": c} | d for a, b, c, d in output_types]


def handle_query_params(query_params):
    """
    Handle query params
    Args:
        query_params (str): Query params in JSON format
    """
    query_params_dict = json.loads(query_params)

    return query_params_dict


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
    output: "pd.DataFrame"
    output, execution_time = dispatcher(
        numba.types.literal(sql_text),
        bc,
    )
    if write_metrics:
        metrics_file.write(f"Execution time: {float(execution_time)}\n".encode("utf-8"))

    if output is not None:
        # Parse output for type specification and number of rows
        # only if we're writing out for JDBC / SDK
        if args.pq_out_filename:
            total_len = MPI.COMM_WORLD.reduce(len(output), op=MPI.SUM, root=0)
            if bodo.get_rank() == 0:
                output_types = parse_output_types(output)
                with open(
                    os.path.join(args.pq_out_filename, "metadata.json"), "w"
                ) as f:
                    json.dump({"num_rows": total_len, "schema": output_types}, f)

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

    iceberg_volume = (
        args.iceberg_volume if args.iceberg_volume else catalog.get("icebergVolume")
    )

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

    iceberg_rest_url = (
        args.iceberg_rest_url
        if args.iceberg_rest_url
        else catalog.get("icebergRestUrl")
    )
    if iceberg_rest_url is None:
        raise ValueError(
            "No icebergRestUrl specified in either the catalog data or through the arguments."
        )

    credential = catalog.get("credential")
    if credential is None:
        raise ValueError("No credential specified in the catalog data.")
    assert (
        ":" in credential
    ), "Credential should be in the format 'client_id:client_secret'"
    client_id, client_secret = credential.split(":")

    # Gets a user access token
    auth_url = iceberg_rest_url + AUTH_URL_SUFFIX
    oauth_response = requests.post(
        auth_url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    user_session_token = oauth_response.json()["access_token"]

    return bodosql.TabularCatalog(
        warehouse=warehouse,
        token=user_session_token,
    )


def main(args):
    if args.verbose_filename:
        # Write verbose logs to the file
        metrics_handler = logging.FileHandler(args.verbose_filename, mode="w")
        bodo_logger.addHandler(metrics_handler)

    # Read in the query text from the file
    with open(args.filename, "r") as f:
        sql_text = f.read()

    query_params = []

    # Read in the query params from the file
    if args.query_param_file:
        with open(args.query_param_file, "r") as f:
            query_params_dict = handle_query_params(f.read())
            for i in range(0, len(query_params_dict) + 1):
                if str(i) in query_params_dict:
                    query_params.append(get_value_for_type(query_params_dict[str(i)]))

    query_params = tuple(query_params)
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
    bodo.barrier()  # Wait for all rankps to finish compilation

    cache_hit: bool = dispatcher._cache_hits[dispatcher.signatures[0]] != 0
    if write_metrics:
        assert metrics_file is not None
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
        assert metrics_file is not None
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

    parser.add_argument(
        "-qpf",
        "--query-param-file",
        required=False,
        help="Path to .json file with the query params",
    )

    args = parser.parse_args()

    main(args)
