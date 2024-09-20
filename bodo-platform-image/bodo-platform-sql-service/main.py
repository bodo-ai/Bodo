import os
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from mpi4py import MPI
from pydantic import BaseModel

import bodo
import bodosql
from bodo_platform_utils.catalog import create_catalog
from bodo_platform_utils.logger import setup_bodo_logger, setup_service_logger
from bodo_platform_utils.query import (
    execute_query,
    compile_query,
    generate_query_plan,
    get_query_metrics_file,
    check_cache_plan_location,
)
from bodo_platform_utils.utils import read_sql_query_param, read_sql_query

# Set ENV variable for AzureFS authentication to look for credentials
# Otherwise it will use anonymous access by default, only for public buckets
# Must be set before calling `fsspec.open`
os.environ["AZURE_STORAGE_ANON"] = "false"

logger = setup_service_logger()

app = FastAPI()

# Initialize the MPI communicator
comm = MPI.COMM_WORLD
rank = bodo.get_rank()


class HealthCheck(BaseModel):
    status: str = "OK"


class SQLQueryModel(BaseModel):
    job_uuid: str

    catalog: str
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    iceberg_volume: str = None
    iceberg_rest_url: str = None

    result_dir: str

    sql_query_text: Optional[str]
    sql_query_filename: Optional[str]
    sql_query_param_filename: Optional[str] = None

    verbose_filename: Optional[str] = None
    metrics_filename: Optional[str] = None
    generate_plan_filename: Optional[str] = None


@app.get("/health")
def health_endpoint():
    return HealthCheck(status="OK")


@app.post("/api/v1/sql")
def sql_endpoint(data: SQLQueryModel):
    logger.info(f"rank {rank} received query model: {data}")
    req = comm.Ibarrier()
    req.Wait()

    comm.bcast(data, root=rank)
    run_sql_query(data)
    return {}


def run_sql_query(data: SQLQueryModel):
    setup_bodo_logger(data.verbose_filename)
    is_root_rank = rank == 0

    if is_root_rank:
        logger.info(f"Read in the query text from the file: {data.sql_query_filename}")

    sql_query_text = data.sql_query_text
    if data.sql_query_filename:
        sql_query_text = read_sql_query(data.sql_query_filename)

    if is_root_rank and data.sql_query_param_filename:
        logger.info(
            f"Read in the query params from the file: {data.sql_query_param_filename}"
        )
    sql_query_params = read_sql_query_param(data.sql_query_param_filename)

    if is_root_rank:
        logger.info(f"Get catalog: {data.catalog}")
    bsql_catalog = create_catalog(data)

    if is_root_rank:
        logger.info(f"Create Bodo SQL context: {data.catalog}")
    bc = bodosql.BodoSQLContext(catalog=bsql_catalog)

    # Generate the plan and write it to a file
    generate_query_plan(sql_query_text, bc, data.generate_plan_filename)

    # Get query metrics file
    metrics_file = get_query_metrics_file(data.metrics_filename)

    if is_root_rank:
        logger.info(f"Started compiling SQL query")
        mpi_version: str = MPI.Get_library_version()
        logger.info(f"Using MPI Version: {mpi_version.strip()}")

    start_time = time.time()
    dispatcher, compilation_time = compile_query(sql_query_text, bc, sql_query_params)

    # Check if cache hit match
    cache_hit: bool = dispatcher._cache_hits[dispatcher.signatures[0]] != 0
    if is_root_rank:
        logger.info(f"Ran from cache: {cache_hit}")
        if metrics_file:
            metrics_file.write(f"Ran from cache: {cache_hit}\n".encode("utf-8"))

    # Get the cache key based on the SQL string
    cache_plan_hit = check_cache_plan_location(sql_query_text)
    if is_root_rank:
        logger.info(f"SQL Plan from cache: {cache_plan_hit}")
        if metrics_file:
            metrics_file.write(
                f"SQL Plan from cache: {cache_plan_hit}\n".encode("utf-8")
            )

    if is_root_rank:
        logger.info("Started executing SQL query")

    execution_time, consume_time = execute_query(
        dispatcher, sql_query_text, bc, sql_query_params, data.result_dir
    )

    if is_root_rank:
        logger.info(f"Execution time: {execution_time} seconds")
        logger.info(f"Consume result time: {consume_time} seconds")

        if metrics_file is not None:
            metrics_file.write(
                f"Execution time: {float(execution_time)}\n".encode("utf-8")
            )
            metrics_file.write(
                f"Consume Query Result time: {float(consume_time)}\n".encode("utf-8")
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


def is_primary():
    primary_metadata_file = Path("/home/bodo/agentPrimaryMetadata")
    return primary_metadata_file.is_file()


def main():
    if is_primary() and rank in bodo.get_nodes_first_ranks():
        logger.info(f"rank {rank} waiting for query from endpoint ...")
        comm.allreduce(rank, op=MPI.SUM)
        uvicorn.run(app, port=5000)
    else:
        bcast_rank = comm.allreduce(0, op=MPI.SUM)

        while True:
            logger.info(f"rank {rank} waiting for query from rank {bcast_rank} ...")

            req = comm.Ibarrier()
            while not req.Test():
                time.sleep(0.05)

            query = comm.bcast(None, root=bcast_rank)
            run_sql_query(query)


if __name__ == "__main__":
    main()
