import datetime
import os
import re
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
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
from bodo_platform_utils.query import write_query_metadata, consume_query_result
from bodo_platform_utils.type_convertor import parse_output_types
from bodo_platform_utils.utils import read_sql_query_param, read_sql_query

from bodosql.context import update_schema, _PlannerType
from bodosql.py4j_gateway import get_gateway
from bodosql.utils import error_to_string
from bodo.libs.distributed_api import bcast_scalar
import numba
import numpy as np
import pandas as pd

SQL_QUERY_TIMEOUT_SECONDS = os.environ.get('SQL_QUERY_TIMEOUT_SECONDS', 3600)
# Set ENV variable for AzureFS authentication to look for credentials
# Otherwise it will use anonymous access by default, only for public buckets
# Must be set before calling `fsspec.open`
os.environ["AZURE_STORAGE_ANON"] = "false"

logger = setup_service_logger()

app = FastAPI()

# Initialize the MPI communicator
comm = MPI.COMM_WORLD
rank = bodo.get_rank()

CATALOG_CACHE = {}
PLANNER_CACHE = {}
QUERY_DB = {}


class NotDDLQuery(Exception):
    pass


class HealthCheck(BaseModel):
    status: str = "OK"


class SQLQueryModel(BaseModel):
    job_uuid: str

    catalog: str
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    iceberg_volume: Optional[str] = None
    iceberg_rest_url: Optional[str] = None

    result_dir: Optional[str] = None

    sql_query_text: Optional[str] = None
    sql_query_filename: Optional[str] = None
    sql_query_param_filename: Optional[str] = None

    verbose_filename: Optional[str] = None
    metrics_filename: Optional[str] = None
    generate_plan_filename: Optional[str] = None


@app.get("/health")
def health_endpoint():
    return HealthCheck(status="OK")


@app.post("/api/v1/sql")
def sql_endpoint(data: SQLQueryModel):
    QUERY_DB[data.job_uuid] = {"status": "RUNNING", "start": datetime.datetime.now()}
    try:
        return DDLExecutor(data).execute()
    except NotDDLQuery:
        logger.info(f"rank {rank} received query model: {data}")
        req = comm.Ibarrier()
        req.Wait()

        comm.bcast(data, root=rank)
        handle_non_ddl_query(data)
        return {}
    finally:
        del QUERY_DB[data.job_uuid]


@app.get("/api/v1/sql")
def get_running_queries():
    return [{"job_uuid": uuid, "status": data.get("status"), "start": data.get('start')} for uuid, data in QUERY_DB.items()]


class DDLExecutor:
    def __init__(self, data: SQLQueryModel):
        self.data = data
        bsql_catalog = self.get_catalog_data()
        logger.info(f"Job: {data.job_uuid} create context")
        self.bc = bodosql.BodoSQLContext(catalog=bsql_catalog)
        logger.info(f"Job: {data.job_uuid} context created")
        logger.info(f"Job: {data.job_uuid} create generator")
        self.plan_generator = self.get_plan_generator()
        self.plan_generator.resetPlanner()
        logger.info(f"Job: {data.job_uuid} planner reset")
        logger.info(f"Job: {data.job_uuid} execute query")

    def execute(self):
        try:
            result = self.execute_if_ddl_query()
        except NotDDLQuery:
            raise
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=f"{error_to_string(e)}")
        return self.parse_result(result)

    def parse_result(self, result):
        if result is not None:
            total_len = len(result)
            try:
                schema = parse_output_types(result)
            except Exception:
                logger.error('Clould not parse schema')
                schema = {"error": "Could not parse schema"}
            if self.data.result_dir:
                # Write schema to metadata.json
                metadata_file = os.path.join(self.data.result_dir, "metadata.json")
                write_query_metadata(schema, total_len, metadata_file)

                # Write output to parquet file
                result_pq = os.path.join(self.data.result_dir, "output.pq")
                consume_query_result(result, result_pq, False)

            return {
                "result": result.to_dict(orient="records"),
                "metadata": {"num_rows": total_len, "schema": schema},
            }
        return None

    def execute_if_ddl_query(self):
        sql_query_text = self.get_query()
        match = re.match("^select ([0-9]+)$", sql_query_text, re.IGNORECASE)

        if match:
            sql_query_text = match.groups()[0]
            logger.info(f"Job: {self.data.job_uuid} query is select {sql_query_text}")
            return self.execute_select_const(sql_query_text)

        if self.is_ddl_query(sql_query_text):
            logger.info(f"Job: {self.data.job_uuid} query is ddl")
            ddl_result = self.plan_generator.executeDDL(sql_query_text)
            # Convert the output to a DataFrame.
            column_names = [x for x in ddl_result.getColumnNames()]
            data = [pd.array(column, dtype=object) for column in ddl_result.getColumnValues()]
            df_dict = {column_names[i]: data[i] for i in range(len(column_names))}
            return pd.DataFrame(df_dict)
        raise NotDDLQuery

    def get_query(self):
        sql_query_text = self.data.sql_query_text
        if self.data.sql_query_filename:
            sql_query_text = read_sql_query(self.data.sql_query_filename)
        return sql_query_text

    def is_ddl_query(self, query):
        try:
            logger.info(f"Job: {self.data.job_uuid} parse query")
            self.plan_generator.parseQuery(query)

            logger.info(f"Job: {self.data.job_uuid} get write type")
            write_type = self.plan_generator.getWriteType(query)
            logger.info(f"Job: {self.data.job_uuid} write type is {write_type}")

            logger.info(f"Job: {self.data.job_uuid} update schema")
            update_schema(
                self.bc.schema,
                self.bc.names,
                self.bc.df_types,
                self.bc.estimated_row_counts,
                self.bc.estimated_ndvs,
                self.bc.orig_bodo_types,
                False,
                write_type,
            )
        except Exception as e:
            err = {
                "type": "string_type",
                "loc": ["body", "query"],
                "msg": f"Unable to parse SQL Query. Error message:\n{e}",
                "input": query,
            }

            raise RequestValidationError([err])

        logger.info(f"Job: {self.data.job_uuid} process ddl")
        is_ddl = self.plan_generator.isDDLProcessedQuery()
        is_ddl = bcast_scalar(is_ddl)
        logger.info(f"Job: {self.data.job_uuid} ddl: {is_ddl}")
        return is_ddl

    def execute_select_const(self, constant):
        return pd.DataFrame({
            'EXP': np.int64(constant)
        }, index=[0])

    @staticmethod
    def get_relation_algebra_generator():
        gateway = get_gateway()
        gateway.jvm.System.currentTimeMillis()
        return gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator

    def get_plan_generator(self):
        if self.data.catalog in PLANNER_CACHE:
            logger.info(
                f"Job: {self.data.job_uuid} plan generator catalog: {self.data.catalog} is in cache"
            )
            return PLANNER_CACHE[self.data.catalog]

        logger.info(
            f"Job: {self.data.job_uuid} plan generator catalog: {self.data.catalog} not in cache"
        )
        if bodo.bodosql_use_streaming_plan:
            planner_type = _PlannerType.Streaming.value
        else:
            planner_type = _PlannerType.Volcano.value

        logger.info(
            f"Job: {self.data.job_uuid} create new plan generator catalog {self.data.catalog} type {planner_type}"
        )

        relation_algebra_generator = self.get_relation_algebra_generator()
        plan_generator = relation_algebra_generator(
            self.bc.catalog.get_java_object(),
            self.bc.schema,
            planner_type,
            0,
            0,
            bodo.bodosql_streaming_batch_size,
            False,
            bodo.enable_snowflake_iceberg,
            bodo.enable_timestamp_tz,
            bodo.enable_runtime_join_filters,
            bodo.enable_streaming_sort,
            bodo.enable_streaming_sort_limit_offset,
            bodo.bodo_sql_style,
            bodo.bodosql_full_caching,
        )

        logger.info(
            f"Job: {self.data.job_uuid} plan generator catalog: {self.data.catalog} type: {planner_type} created"
        )
        PLANNER_CACHE[self.data.catalog] = plan_generator
        return plan_generator

    def get_catalog_data(self):
        if not self.data.catalog:
            return bodosql.FileSystemCatalog(".")

        # Catalog can be cached by name since names are guaranteed to be unique
        if self.data.catalog not in CATALOG_CACHE:
            logger.info(f"Catalog {self.data.catalog} NOT in cache")
            bsql_catalog = create_catalog(self.data)
            CATALOG_CACHE[self.data.catalog] = bsql_catalog
        else:
            logger.info(f"Catalog {self.data.catalog} in cache")

        return CATALOG_CACHE.get(self.data.catalog)


def handle_non_ddl_query(data: SQLQueryModel):
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
        uvicorn.run(app, port=5000, timeout_keep_alive=SQL_QUERY_TIMEOUT_SECONDS)
    else:
        bcast_rank = comm.allreduce(0, op=MPI.SUM)

        while True:
            logger.info(f"rank {rank} waiting for query from rank {bcast_rank} ...")

            req = comm.Ibarrier()
            while not req.Test():
                time.sleep(0.05)

            query = comm.bcast(None, root=bcast_rank)
            handle_non_ddl_query(query)


if __name__ == "__main__":
    main()
