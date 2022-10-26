"""
API used to translate Java BodoParquetInfo objects into
Python Objects usable inside Bodo.
"""
from collections import namedtuple
from typing import List, Tuple
from urllib.parse import urlparse

from bodo_iceberg_connector.catalog_conn import _remove_prefix, parse_conn_str
from bodo_iceberg_connector.config import DEFAULT_PORT
from bodo_iceberg_connector.errors import IcebergJavaError
from bodo_iceberg_connector.filter_to_java import convert_expr_to_java_parsable
from bodo_iceberg_connector.py4j_support import get_java_table_handler
from py4j.protocol import Py4JJavaError

# Named Tuple for Parquet info
BodoIcebergParquetInfo = namedtuple("BodoIcebergParquetInfo", "filepath start length")


def bodo_connector_get_parquet_file_list(
    conn_str: str, db_name: str, table: str, filters
) -> Tuple[List[str], List[str]]:
    """
    Gets the list of files for use by Bodo. The port value here
    is set and controlled by a default value for the bodo_iceberg_connector
    package.

    Here we return two lists:
        List 1: The Iceberg paths that have been cleaned up. Here we standardize the s3
        filepath and remove any "file:" header. In addition we convert files not on s3
        to their absolute path

        List 2: The Iceberg paths exactly as given. These may have various headers and
        are generally relative paths.

        For example if the absolute path was /Users/bodo/iceberg_db/my_table/part01.pq
        and the iceberg directory is iceberg_db, then the path in list 1 would be
        /Users/bodo/iceberg_db/my_table/part01.pq and the path in list 2 would be
        iceberg_db/my_table/part01.pq.
    """
    pq_infos = get_bodo_parquet_info(DEFAULT_PORT, conn_str, db_name, table, filters)

    # filepath is a URI (file:///User/sw/...) or a relative path that needs converted to
    # a full path
    # Replace Hadoop S3A URI scheme with regular S3 Scheme
    file_paths = [x.filepath for x in pq_infos]
    return (
        [
            _remove_prefix(path.replace("s3a://", "s3://"), "file:")
            if _has_uri_scheme(path)
            else f"{_remove_prefix(conn_str, 'file:')}/{path}"
            for path in file_paths
        ],
        file_paths,
    )


def bodo_connector_get_parquet_info(warehouse, schema, table, filters):
    """
    Gets the BodoIcebergParquetInfo for use by Bodo. The port value here
    is set and controlled by a default value for the bodo_iceberg_connector
    package.
    """
    return get_bodo_parquet_info(DEFAULT_PORT, warehouse, schema, table, filters)


def get_bodo_parquet_info(port, conn_str: str, db_name: str, table: str, filters):
    """
    Returns the BodoIcebergParquetInfo for a table.

    Port is unused and kept in case we opt to switch back to py4j
    """

    try:
        catalog_type, _ = parse_conn_str(conn_str)

        bodo_iceberg_table_reader = get_java_table_handler(
            conn_str,
            catalog_type,
            db_name,
            table,
        )

        filter_expr = convert_expr_to_java_parsable(filters)
        java_parquet_infos = get_java_parquet_info(
            bodo_iceberg_table_reader, filter_expr
        )

    except Py4JJavaError as e:
        raise IcebergJavaError.from_java_error(e)

    return java_to_python(java_parquet_infos)


def get_java_parquet_info(bodo_iceberg_table_reader, filter_expr):
    """Returns the parquet info as a Java object"""
    return bodo_iceberg_table_reader.getParquetInfo(filter_expr)


def java_to_python(java_parquet_infos) -> List[BodoIcebergParquetInfo]:
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


def _has_uri_scheme(path: str):
    """return True of path has a URI scheme, e.g. file://, s3://, etc."""
    try:
        return urlparse(path).scheme != ""
    except:
        return False
