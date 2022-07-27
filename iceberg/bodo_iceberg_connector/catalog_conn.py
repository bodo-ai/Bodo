import os
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from bodo_iceberg_connector.errors import IcebergError


def _get_first(elems: Dict[str, List[str]], param: str) -> Optional[str]:
    elem = elems.get(param, None)
    return elem[0] if elem and len(elem) > 0 else None


def parse_conn_str(
    conn_str: str,
) -> Tuple[str, Optional[str]]:
    """
    Parse catalog / metastore connection string to determine catalog type
    and potentially the warehouse location
    """
    # TODO: To Enum or Literal["hadoop", "hive", "nessie", "glue"]?
    parsed_conn = urlparse(conn_str)
    conn_query = parse_qs(parsed_conn.query)

    # Determine Catalog Type
    catalog_type = _get_first(conn_query, "catalog")
    if catalog_type is None:
        if parsed_conn.scheme == "thrift":
            catalog_type = "hive"
        elif parsed_conn.scheme == "" and parsed_conn.path == "glue":
            catalog_type = "glue"
        elif parsed_conn.scheme == "s3":
            catalog_type = "hadoop-s3"
        elif parsed_conn.scheme == "" or parsed_conn.scheme == "file":
            catalog_type = "hadoop"
        else:
            raise IcebergError(
                f"Cannot detect Iceberg catalog type from connection string:\n  {conn_str}"
            )

    assert catalog_type in ["hadoop-s3", "hadoop", "hive", "nessie", "glue"]

    # Get Warehouse Location
    # TODO: Do something more intelligent with thrift
    if catalog_type in ["hadoop", "hadoop-s3"]:
        # TODO: assert warehouse query parameter is None ???
        fs_prefix = "s3://" if parsed_conn.scheme == "s3" else ""
        warehouse = f"{fs_prefix}{parsed_conn.netloc}{parsed_conn.path}"
    else:
        warehouse = _get_first(conn_query, "warehouse")

    # TODO: Pass more parsed results to use in java
    return catalog_type, warehouse


def gen_table_loc(
    catalog_type: str, warehouse: str, db_name: str, table_name: str
) -> str:
    """Construct Table Data Location from Warehouse and Connection Info"""
    inner_name = (
        db_name + ".db"
        if catalog_type == "glue" or catalog_type == "nessie"
        else db_name
    )

    # We attach `data` since C++ code expects the directory
    # where the parquet files should be written
    return os.path.join(warehouse, inner_name, table_name)


def gen_file_loc(
    catalog_type: str, table_loc: str, db_name: str, table_name: str, file_name: str
) -> str:
    """Construct Valid Paths for Files Written to Iceberg"""

    if catalog_type == "hadoop":
        return os.path.join(db_name, table_name, "data", file_name)
    elif catalog_type in ["glue", "nessie", "hadoop-s3"]:
        return os.path.join(table_loc, file_name)
    else:
        return file_name


def normalize_loc(loc: str):
    loc = _remove_prefix(loc.replace("s3a://", "s3://"), "file:")
    return os.path.join(loc, "data")


def _remove_prefix(input: str, prefix: str) -> str:
    """
    Remove Prefix from String if Available
    This is part of Python's Standard Library starting from 3.9
    TODO: Remove once Python 3.8 is deprecated
    """
    if sys.version_info.minor < 9:
        return input[len(prefix) :] if input.startswith(prefix) else input
    else:
        return input.removeprefix(prefix)
