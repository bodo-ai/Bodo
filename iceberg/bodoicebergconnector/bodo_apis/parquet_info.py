"""
API used to translate Java BodoParquetInfo objects into
Python Objects usable inside Bodo.
"""
from collections import namedtuple

import jpype

import bodoicebergconnector.bodo_apis.jpype_support
from bodoicebergconnector.bodo_apis.config import DEFAULT_PORT
from bodoicebergconnector.bodo_apis.errors import IcebergError
from bodoicebergconnector.bodo_apis.filter_to_java import convert_expr_to_java_parsable

# Named Tuple for Parquet info
BodoIcebergParquetInfo = namedtuple("BodoIcebergParquetInfo", "filepath start length")


def bodo_connector_get_parquet_file_list(warehouse, schema, table, filters):
    """
    Gets the list of files for use by Bodo. The port value here
    is set and controlled by a default value for the bodo_iceberg_connector
    package.
    """
    pq_infos = get_bodo_parquet_info(DEFAULT_PORT, warehouse, schema, table, filters)
    return [f"{warehouse}/{x.filepath}" for x in pq_infos]


def bodo_connector_get_parquet_info(warehouse, schema, table, filters):
    """
    Gets the BodoIcebergParquetInfo for use by Bodo. The port value here
    is set and controlled by a default value for the bodo_iceberg_connector
    package.
    """
    return get_bodo_parquet_info(DEFAULT_PORT, warehouse, schema, table, filters)


def get_bodo_parquet_info(port, warehouse, schema, table, filters):
    """
    Returns the BodoIcebergParquetInfo for a table.

    Port is unused and kept in case we opt to switch back to py4j
    """
    try:
        bodo_iceberg_table_reader = (
            bodoicebergconnector.bodo_apis.jpype_support.get_iceberg_java_table_reader(
                warehouse,
                schema,
                table,
            )
        )

        filter_expr = convert_expr_to_java_parsable(filters)
        java_parquet_infos = get_java_parquet_info(
            bodo_iceberg_table_reader, filter_expr
        )
    except jpype.JException as e:
        raise IcebergError.from_java_exception(e)

    return java_to_python(java_parquet_infos)


def get_java_parquet_info(bodo_iceberg_table_reader, filter_expr):
    """
    Returns the parquet info as a Java object
    """
    return bodo_iceberg_table_reader.getParquetInfo(filter_expr)


def java_to_python(java_parquet_infos):
    """
    Converts an Iterable of Java BodoParquetInfo objects
    to an equivalent list of Named Tuples.
    """
    pq_infos = []
    for java_pq_info in java_parquet_infos:
        if bool(java_pq_info.hasDeleteFile()):
            raise RuntimeError(
                "Iceberg Dataset contains DeleteFiles, which is not yet supported by Bodo"
            )
        pq_infos.append(
            BodoIcebergParquetInfo(
                str(java_pq_info.getFilepath()),
                int(java_pq_info.getStart()),
                int(java_pq_info.getLength()),
            )
        )
    return pq_infos
