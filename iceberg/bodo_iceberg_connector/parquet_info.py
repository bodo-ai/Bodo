"""
API used to translate Java BodoParquetInfo objects into
Python Objects usable inside Bodo.
"""

import os
import typing as pt
from dataclasses import dataclass
from urllib.parse import urlparse

from py4j.protocol import Py4JError

from bodo_iceberg_connector.catalog_conn import parse_conn_str
from bodo_iceberg_connector.errors import IcebergJavaError
from bodo_iceberg_connector.filter_to_java import FilterExpr
from bodo_iceberg_connector.py4j_support import get_catalog
from bodo_iceberg_connector.schema_helper import arrow_schema_j2py

if pt.TYPE_CHECKING:
    import pyarrow as pa


@dataclass
class IcebergParquetInfo:
    """Named Tuple for Parquet info"""

    # Original path of the parquet file from Iceberg metadata
    orig_path: str
    # Standardized path to the parquet file for Bodo use
    standard_path: str
    # Number of rows in the parquet file from Iceberg metadata
    row_count: int
    # Iceberg Schema ID the parquet file was written with
    schema_id: int


def get_bodo_parquet_info(
    conn_str: str, db_name: str, table: str, filters: FilterExpr | None
) -> tuple[list[IcebergParquetInfo], dict[int, "pa.Schema"], int]:
    """
    Returns the IcebergParquetInfo for a table.
    Port is unused and kept in case we opt to switch back to py4j
    """

    try:
        catalog_type, warehouse_loc = parse_conn_str(conn_str)

        handler = get_catalog(conn_str, catalog_type)

        filters = FilterExpr.default() if filters is None else filters
        filter_expr = filters.to_java()
        java_out = handler.getParquetInfo(db_name, table, filter_expr)
        java_parquet_infos = java_out.getFirst()
        java_all_schemas = java_out.getSecond()
        get_file_to_schema_us = java_out.getThird() // 1000

    except Py4JError as e:
        raise IcebergJavaError.from_java_error(e)

    return (
        java_pq_info_to_python(java_parquet_infos, warehouse_loc),
        {
            item.getKey(): arrow_schema_j2py(item.getValue())
            for item in java_all_schemas.entrySet()
        },
        get_file_to_schema_us,
    )


def java_pq_info_to_python(
    java_parquet_infos, warehouse_loc: str | None
) -> list[IcebergParquetInfo]:
    """
    Converts an Iterable of Java BodoParquetInfo objects
    to an equivalent list of IcebergParquetInfo.
    """
    pq_infos = []
    for java_pq_info in java_parquet_infos:
        if bool(java_pq_info.hasDeleteFile()):
            raise RuntimeError(
                "Iceberg Dataset contains DeleteFiles, which is not yet supported by Bodo"
            )
        orig_path = str(java_pq_info.getFilepath())
        pq_infos.append(
            IcebergParquetInfo(
                orig_path,
                standardize_path(orig_path, warehouse_loc),
                int(java_pq_info.getRowCount()),
                int(java_pq_info.getSchemaID()),
            )
        )
    return pq_infos


def standardize_path(path: str, warehouse_loc: str | None) -> str:
    if warehouse_loc is not None:
        warehouse_loc = warehouse_loc.replace("s3a://", "s3://").removeprefix("file:")

    if _has_uri_scheme(path):
        return (
            path.replace("s3a://", "s3://")
            .replace("wasbs://", "abfss://")
            .replace("wasb://", "abfs://")
            .replace("blob.core.windows.net", "dfs.core.windows.net")
            .removeprefix("file:")
        )
    elif warehouse_loc is not None:
        return os.path.join(warehouse_loc, path)
    else:
        return path


def _has_uri_scheme(path: str):
    """return True of path has a URI scheme, e.g. file://, s3://, etc."""
    try:
        return urlparse(path).scheme != ""
    except Exception:
        return False


def bodo_connector_get_total_num_pq_files_in_table(
    conn_str: str, db_name: str, table: str
) -> int:
    """
    Returns the number of parquet files in the given Iceberg table.
    Throws a IcebergJavaError if an error occurs.
    """
    try:
        catalog_type, _ = parse_conn_str(conn_str)

        bodo_iceberg_table_reader = get_catalog(conn_str, catalog_type)

        return bodo_iceberg_table_reader.getNumParquetFiles(db_name, table)

    except Py4JError as e:
        raise IcebergJavaError.from_java_error(e)
