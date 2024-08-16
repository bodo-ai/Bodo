import datetime
import io
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import bodo
import bodosql
import fsspec
import numba
import numpy as np
import pandas as pd
import uvicorn
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError
from bodo_platform_utils.bodosqlwrapper import (
    CatalogType,
    create_tabular_catalog,
    create_glue_catalog,
    create_snowflake_catalog,
)
from bodo_platform_utils.catalog import get_data
from bodo_platform_utils.type_convertor import parse_output_types
from bodosql.context import update_schema, _PlannerType
from bodosql.py4j_gateway import get_gateway
from bodosql.utils import error_to_string
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from mpi4py import MPI
from pydantic import BaseModel

logger = logging.getLogger("bodo-ddl-service")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
BODO_PLATFORM_WORKSPACE_UUID = os.environ.get("BODO_PLATFORM_WORKSPACE_UUID")
BODO_PLATFORM_CLOUD_PROVIDER = os.environ.get("BODO_PLATFORM_CLOUD_PROVIDER")
BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES = os.environ.get(
    "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES", None
)
if BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES:
    parts = BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES.split("@")
    account_part = parts[1] if len(parts) > 1 else ""
    BODO_STORAGE_ACCOUNT_NAME = account_part.split(".")[0]
else:
    BODO_STORAGE_ACCOUNT_NAME = None
logger.addHandler(handler)
app = FastAPI()


class QueryModel(BaseModel):
    job_uuid: str
    query: str
    catalog: Optional[str] = None
    save_results: Optional[bool] = False


class MockArgs:
    warehouse = None
    iceberg_rest_url = None
    database = None
    schema = None
    iceberg_volume = None


catalog_cache = {}
planner_cache = {}


@app.post("/api/v1/query")
def query_endpoint(data: QueryModel):
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
    if data.query == "SELECT 1":
        logger.info(f"Job: {data.job_uuid} query is select 1")
        result = exec_select_one(bc, plan_generator)
    elif check_is_ddl(bc, data, plan_generator):
        try:
            logger.info(f"Job: {data.job_uuid} query is ddl")
            result = execute_ddl(data, plan_generator)
        except BodoError as e:
            raise HTTPException(status_code=500, detail=f"{e}")
    else:
        logger.info(f"Job: {data.job_uuid} can't execute query")
        raise HTTPException(
            status_code=409, detail="Not DDL queries should be executed by slurm"
        )
    logger.info(f"Job: {data.job_uuid} query executed")

    total_len = len(result)
    output_types = parse_output_types(result)
    if data.save_results and result is not None:
        save_results(data, result, total_len, output_types)

    if result is not None:
        return {
            "result": result.to_dict(orient="records"),
            "metadata": {"num_rows": total_len, "schema": output_types},
        }

    return None


def execute_ddl(data, plan_generator):
    result, error = None, None
    try:
        logger.info(f"Job: {data.job_uuid} execute ddl query")
        ddl_result = plan_generator.executeDDL(data.query)
        logger.info(f"Job: {data.job_uuid} converting results")

        # Convert the output to a DataFrame.
        column_names = [x for x in ddl_result.getColumnNames()]
        data = [
            pd.array(column, dtype=object) for column in ddl_result.getColumnValues()
        ]
        df_dict = {column_names[i]: data[i] for i in range(len(column_names))}
        result = pd.DataFrame(df_dict)
    except Exception as e:
        error = error_to_string(e)
        raise BodoError(error)

    logger.info(f"Job: {data.job_uuid} results converted")
    return result


def check_is_ddl(bc, data, plan_generator):
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
    if data.catalog not in catalog_cache:
        logger.info("Catalog NOT in cache")
        catalog = get_data(data.catalog)
        if catalog is None:
            raise ValueError("Catalog not found in the secret store.")

        catalog_type_str = catalog.get("catalogType")
        if catalog_type_str is None:
            catalog_type = CatalogType.SNOWFLAKE
        else:
            catalog_type = CatalogType(catalog_type_str)

        if catalog_type == CatalogType.TABULAR:
            bsql_catalog = create_tabular_catalog(catalog, MockArgs())
        elif catalog_type == CatalogType.GLUE:
            bsql_catalog = create_glue_catalog(catalog, MockArgs())
        else:
            # default to Snowflake for backward compatibility
            bsql_catalog = create_snowflake_catalog(catalog, MockArgs())
        catalog_cache[data.catalog] = bsql_catalog
    else:
        bsql_catalog = catalog_cache.get(data.catalog)
        logger.info("Catalog in cache")

        # Create context
    return bsql_catalog


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


def save_results(data, result, total_len, output_types):
    logger.info(f"Job: {data.job_uuid} save results: {total_len}")

    try:
        if BODO_PLATFORM_CLOUD_PROVIDER == "AWS":
            s3_location = f"s3://bodoai-fs-{BODO_PLATFORM_WORKSPACE_UUID}/query_results/job-{data.job_uuid}"
            result_path = f"{s3_location}/part-00.parquet"
            metadata_path = f"{s3_location}/metadata.json"

            result.to_parquet(result_path)
            with fsspec.open(metadata_path, "w") as f:
                json.dump({"num_rows": total_len, "schema": output_types}, f)

        if BODO_PLATFORM_CLOUD_PROVIDER == "AZURE":
            container_name = f"bodoai-fs-{BODO_PLATFORM_WORKSPACE_UUID}"
            credential = DefaultAzureCredential()
            account_url = f"https://{BODO_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

            blob_service_client = BlobServiceClient(
                account_url=account_url, credential=credential
            )
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=f"query_results/job-{data.job_uuid}/part-00.parquet",
            )
            buffer = io.BytesIO()
            result.to_parquet(buffer, index=False)
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=True)
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=f"query_results/job-{data.job_uuid}/metadata.json",
            )
            buffer = io.StringIO()
            json.dump({"num_rows": total_len, "schema": output_types}, buffer)
            buffer.seek(0)
            blob_client.upload_blob(
                io.BytesIO(buffer.getvalue().encode("utf-8")), overwrite=True
            )
    except Exception as ex:
        logger.error(
            f"Job [{data.job_uuid}] can't save results in cloud provider: {ex}"
        )


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
