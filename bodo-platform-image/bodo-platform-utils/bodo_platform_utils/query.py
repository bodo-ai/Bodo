import json
import os
import time
import typing as pt

import fsspec
import numba
import pandas as pd
from bodo.mpi4py import MPI

import bodo
import bodo.utils.tracing as tracing
from bodo.sql_plan_cache import BodoSqlPlanCache
from bodosql.context_ext import typeof_bodo_sql

from .type_convertor import parse_output_types


def generate_query_plan(sql: str, bc, filename: str = None):
    """Function generate query plan and save to filename.

    :param sql: SQL query text to generate plan
    :param bc: BodoSQLContext (bodosql.BodoSQLContext) to use for query execution
    :param filename: Filename where plan should be generated
    :return:
    """
    if not filename:
        return

    plan_text = bc.generate_plan(sql, show_cost=True)
    if bodo.get_rank() == 0:
        with open(filename, "w") as f:
            f.write(plan_text)


def convert_query_to_pandas(sql: str, bc, filename: str = None):
    """Function convert query to pandas and save to filename

    :param sql: SQL query text to convert
    :param bc: BodoSQLContext (bodosql.BodoSQLContext) to use for query execution
    :param filename: Filename where plan should be generated
    """
    if not filename:
        return

    pandas_text = bc.convert_to_pandas(sql)
    if bodo.get_rank() == 0:
        with open(filename, "w") as f:
            f.write(pandas_text)


def get_query_metrics_file(filename: str = None):
    """Function that return metrics file for rank 0 if filename provided

    :param filename: Filename metric
    :return: Metrics file
    """
    if filename and bodo.get_rank() == 0:
        return open(filename, "wb", buffering=0)
    return None


def get_cache_query_location(dispatcher) -> pt.Optional[str]:
    """Get the location of the cached binary from the dispatcher object of a function.
    In case we aren't able to get the location, None will be returned.

    :param dispatcher: Dispatcher function for the query.
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


def check_cache_plan_location(sql: str):
    plan_location: str | None = BodoSqlPlanCache.get_cache_loc(sql)

    if not plan_location:
        return False

    return os.path.isfile(plan_location)


def compile_query(sql: str, bc, params):
    """Compile the query

    :param sql: SQL query text to execute
    :param bc: BodoSQLContext (bodosql.BodoSQLContext) to use for query execution
    :param params: Dynamic params list for sql
    :return:
    """
    t0 = time.time()
    dispatcher = bodo.jit(
        (
            numba.types.literal(sql),
            typeof_bodo_sql(bc, None),
            bodo.typeof(params),
        ),
        cache=True,
    )(execute_query_helper)

    bodo.barrier()  # Wait for all ranks to finish compilation
    compilation_time = time.time() - t0

    return dispatcher, compilation_time


def execute_query(
    dispatcher,
    sql,
    bc,
    query_params,
    output_dir: str,
    output_stdout: bool = False,
    trace_filename: str = None,
):
    """Wrapper function to run the query and consume the result.

    :param dispatcher: Dispatcher function for the query.
    :param sql: Query text to execute
    :param bc: BodoSQLContext(bodosql.BodoSQLContext) to use for query execution.
    :param query_params: Query bind variables from Python.
    :param output_dir: Directory path to save results.
    :param output_stdout: Flag to print query result to standard input.

    :param trace_filename:
    """
    if trace_filename:
        tracing.start()

    output: "pd.DataFrame"
    output, execution_time = dispatcher(
        numba.types.literal(sql),
        bc,
        query_params,
    )
    consume_time = None

    if output is not None:
        start_time = time.time()
        output_pq: str = ""

        if output_dir:
            total_len = MPI.COMM_WORLD.reduce(len(output), op=MPI.SUM, root=0)
            if bodo.get_rank() == 0:
                # Parse output for type specification and number of rows
                # only if we're writing out for JDBC / SDK
                schema = parse_output_types(output)

                # Write schema to metadata.json
                metadata_file = os.path.join(output_dir, "metadata.json")
                write_query_metadata(schema, total_len, metadata_file)

            output_pq = os.path.join(output_dir, "output.pq")

        # Write output to parquet file or to standard output
        consume_query_result(
            output,
            output_pq,
            output_stdout,
        )

        bodo.barrier()
        consume_time = time.time() - start_time

    if trace_filename:
        tracing.dump(fname=trace_filename)

    return execution_time, consume_time


@bodo.jit(cache=True)
def consume_query_result(output: "pd.DataFrame", output_pq: str, output_stdout: bool):
    """Function to consume the query result.

    :param output: Output that should be consumed
    :param output_pq: When provided the query output is written to this location as a parquet file
    :param output_stdout: Flag to print query result
    """
    print("Output Shape: ", output.shape)
    if output_stdout:
        print("Output:")
        print(output)

    # Depends on mode (replicated, distributed) it will save one file or multiple
    if output_pq != "":
        output.to_parquet(output_pq)


def execute_query_helper(sql, bc, params):
    """Boilerplate function to execute a query string.

    :param sql: SQL query text to execute
    :param bc: BodoSQLContext (bodosql.BodoSQLContext) to use for query execution
    :param params: Dynamic params list for sql
    """
    t0 = time.time()
    output = bc.sql(sql, dynamic_params_list=params)
    execution_time = time.time() - t0
    return output, execution_time


def write_query_metadata(schema, num_rows: int, metadata_file: str):
    """Write query metadata to metadata.json file

    :param schema: Schema metadata
    :param num_rows: Number of rows
    :param metadata_file: Path to root metadata file
    """
    with fsspec.open(metadata_file, "w") as f:
        json.dump({"num_rows": num_rows, "schema": schema}, f)
