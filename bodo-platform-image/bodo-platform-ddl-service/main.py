import datetime
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numba
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic import BaseModel

import bodo
import bodosql
from bodo.libs.distributed_api import bcast_scalar
from bodo_platform_utils.catalog import create_catalog
from bodo_platform_utils.logger import setup_service_logger
from bodo_platform_utils.query import write_query_metadata, consume_query_result
from bodo_platform_utils.type_convertor import parse_output_types
from bodosql.context import update_schema, _PlannerType
from bodosql.py4j_gateway import get_gateway
from bodosql.utils import error_to_string

# Set ENV variable for AzureFS authentication to look for credentials
# Otherwise it will use anonymous access by default, only for public buckets
# Must be set before calling `fsspec.open`
os.environ["AZURE_STORAGE_ANON"] = "false"

logger = setup_service_logger("bodo-ddl-service")

app = FastAPI()


class SQLQueryModel(BaseModel):
    job_uuid: str
    query: str

    catalog: Optional[str] = None
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    result_dir: Optional[str] = None

    iceberg_volume: str = None
    iceberg_rest_url: str = None


catalog_cache = {}
planner_cache = {}


@app.post("/api/v1/query")
def query_endpoint(data: SQLQueryModel):
    logger.info(f"Job: {data.job_uuid} receive query")

    bsql_catalog = get_catalog_data(data)
    logger.info(f"Job: {data.job_uuid} create context")

    bc = bodosql.BodoSQLContext(catalog=bsql_catalog)
    logger.info(f"Job: {data.job_uuid} context created")

    logger.info(f"Job: {data.job_uuid} create generator")
    plan_generator = get_plan_generator(bc, data)
    logger.info(f"Job: {data.job_uuid} generator created")

    logger.info(f"Job: {data.job_uuid} resetting planner")
    plan_generator.resetPlanner()
    logger.info(f"Job: {data.job_uuid} planner reset")

    logger.info(f"Job: {data.job_uuid} execute query")
    result = run_sql_query(data, bc, plan_generator)
    logger.info(f"Job: {data.job_uuid} query executed")

    if result is not None:
        total_len = len(result)
        schema = parse_output_types(result)

        if data.result_dir:
            # Write schema to metadata.json
            metadata_file = os.path.join(data.result_dir, "metadata.json")
            write_query_metadata(schema, total_len, metadata_file)

            # Write output to parquet file
            result_pq = os.path.join(data.result_dir, "output.pq")
            consume_query_result(result, result_pq, False)

        return {
            "result": result.to_dict(orient="records"),
            "metadata": {"num_rows": total_len, "schema": schema},
        }

    return None


def run_sql_query(data: SQLQueryModel, bc, generator):
    query_sql = data.query
    match = re.match("^select ([0-9]+)$", query_sql, re.IGNORECASE)
    if match:
        constant = match.groups()[0]
        logger.info(f"Job: {data.job_uuid} query is select {constant}")
        return execute_select_const(bc, generator, constant)

    if is_ddl_query(bc, data, generator):
        try:
            logger.info(f"Job: {data.job_uuid} query is ddl")
            return execute_ddl(query_sql, generator)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=f"{error_to_string(e)}")

    logger.info(f"Job: {data.job_uuid} can't execute query")
    raise HTTPException(
        status_code=409, detail="Non-DDL queries should be executed by slurm"
    )


def is_ddl_query(bc, data, plan_generator):
    try:
        logger.info(f"Job: {data.job_uuid} parse query")
        plan_generator.parseQuery(data.query)

        logger.info(f"Job: {data.job_uuid} get write type")
        write_type = plan_generator.getWriteType(data.query)
        logger.info(f"Job: {data.job_uuid} write type is {write_type}")

        logger.info(f"Job: {data.job_uuid} update schema")
        update_schema(
            bc.schema,
            bc.names,
            bc.df_types,
            bc.estimated_row_counts,
            bc.estimated_ndvs,
            bc.orig_bodo_types,
            False,
            write_type,
        )
    except Exception as e:
        err = {
            "type": "string_type",
            "loc": ["body", "query"],
            "msg": f"Unable to parse SQL Query. Error message:\n{e}",
            "input": data.query,
        }

        raise RequestValidationError([err])

    logger.info(f"Job: {data.job_uuid} process ddl")
    is_ddl = plan_generator.isDDLProcessedQuery()
    is_ddl = bcast_scalar(is_ddl)
    logger.info(f"Job: {data.job_uuid} ddl: {is_ddl}")
    return is_ddl


def execute_select_const(bc, generator, constant):
    func_text, lowered_globals = bc._convert_to_pandas(
        f"SELECT {constant}",
        [],
        dict(),
        generator,
        False,
    )

    glbls = {
        "np": np,
        "pd": pd,
        "bodosql": bodosql,
        "re": re,
        "bodo": bodo,
        "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
        "MetaType": bodo.utils.typing.MetaType,
        "numba": numba,
        "time": time,
        "datetime": datetime,
        "bif": bodo.ir.filter,
    }

    glbls.update(lowered_globals)
    loc_vars = {}
    exec(
        func_text,
        glbls,
        loc_vars,
    )
    impl = loc_vars["impl"]

    # Add table argument name prefix to user provided distributed flags to match
    # stored names
    return bodo.jit(impl)(*(list(bc.tables.values())))


def execute_ddl(sql: str, generator):
    ddl_result = generator.executeDDL(sql)
    # Convert the output to a DataFrame.
    column_names = [x for x in ddl_result.getColumnNames()]
    data = [pd.array(column, dtype=object) for column in ddl_result.getColumnValues()]
    df_dict = {column_names[i]: data[i] for i in range(len(column_names))}
    return pd.DataFrame(df_dict)


def get_relation_algebra_generator():
    gateway = get_gateway()
    gateway.jvm.System.currentTimeMillis()
    return gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator


def get_plan_generator(bc, data):
    if data.catalog in planner_cache:
        logger.info(
            f"Job: {data.job_uuid} plan generator catalog: {data.catalog} is in cache"
        )
        return planner_cache[data.catalog]

    logger.info(
        f"Job: {data.job_uuid} plan generator catalog: {data.catalog} not in cache"
    )
    if bodo.bodosql_use_streaming_plan:
        planner_type = _PlannerType.Streaming.value
    else:
        planner_type = _PlannerType.Volcano.value

    logger.info(
        f"Job: {data.job_uuid} create new plan generator catalog {data.catalog} type {planner_type}"
    )

    relation_algebra_generator = get_relation_algebra_generator()
    plan_generator = relation_algebra_generator(
        bc.catalog.get_java_object(),
        bc.schema,
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
        f"Job: {data.job_uuid} plan generator catalog: {data.catalog} type: {planner_type} created"
    )
    planner_cache[data.catalog] = plan_generator
    return plan_generator


def get_catalog_data(data):
    if not data.catalog:
        return bodosql.FileSystemCatalog(".")

    # Catalog can be cached by name since names are guaranteed to be unique
    if data.catalog not in catalog_cache:
        logger.info(f"Catalog {data.catalog} NOT in cache")
        bsql_catalog = create_catalog(data)
        catalog_cache[data.catalog] = bsql_catalog
    else:
        logger.info(f"Catalog {data.catalog} in cache")

    return catalog_cache.get(data.catalog)


def exec_select_one(bc, generator):
    func_text, lowered_globals = bc._convert_to_pandas(
        "SELECT 1",
        [],
        dict(),
        generator,
        False,
    )

    glbls = {
        "np": np,
        "pd": pd,
        "bodosql": bodosql,
        "re": re,
        "bodo": bodo,
        "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
        "MetaType": bodo.utils.typing.MetaType,
        "numba": numba,
        "time": time,
        "datetime": datetime,
        "bif": bodo.ir.filter,
    }

    glbls.update(lowered_globals)
    loc_vars = {}
    exec(
        func_text,
        glbls,
        loc_vars,
    )
    impl = loc_vars["impl"]

    # Add table argument name prefix to user provided distributed flags to match
    # stored names
    return bodo.jit(impl)(*(list(bc.tables.values())))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Bodo DDL Service loading dependencies ...")
    catalog = bodosql.FileSystemCatalog(".")
    bc = bodosql.BodoSQLContext(catalog=catalog)
    bc.sql("SELECT 1")
    bc.sql("BEGIN")
    logger.info("Bodo DDL Service dependencies loaded")

    yield
    logger.info("Bodo DDL Service shutdown")


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, port=8888)
