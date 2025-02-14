import os
import warnings
from typing import Literal
from urllib.parse import parse_qs, urlparse

from bodo_iceberg_connector.errors import IcebergError, IcebergWarning

CatalogType = Literal[
    "hadoop", "hive", "nessie", "glue", "snowflake", "rest", "s3tables"
]


def _get_first(elems: dict[str, list[str]], param: str) -> str | None:
    elem = elems.get(param, None)
    return elem[0] if elem and len(elem) > 0 else None


def parse_conn_str(
    conn_str: str,
) -> tuple[str, str | None]:
    """
    Parse catalog / metastore connection string to determine catalog type
    and potentially the warehouse location
    """
    # TODO: To Enum or Literal["hadoop", "hive", "nessie", "glue"]?
    parsed_conn = urlparse(conn_str)
    conn_query = parse_qs(parsed_conn.query)

    # Determine Catalog Type
    catalog_type = _get_first(conn_query, "type")
    warehouse = _get_first(conn_query, "warehouse")
    if catalog_type is None:
        if parsed_conn.scheme == "iceberg+thrift":
            catalog_type = "hive"

        elif parsed_conn.scheme == "" and parsed_conn.path == "iceberg+glue":
            catalog_type = "glue"

        elif parsed_conn.scheme == "iceberg+s3":
            catalog_type = "hadoop-s3"
            if warehouse is not None:
                warnings.warn(
                    "The `warehouse` property in the connection string will be ignored when accessing a Hadoop S3 catalog. Instead, the `warehouse` will be inferred from the connection string path itself.",
                    IcebergWarning,
                )
            warehouse = f"s3://{parsed_conn.netloc}{parsed_conn.path}"

        elif parsed_conn.scheme in ("iceberg+abfs", "iceberg+abfss"):
            catalog_type = "hadoop-abfs"
            if warehouse is not None:
                warnings.warn(
                    "The `warehouse` property in the connection string will be ignored when accessing a Hadoop ABFS catalog. Instead, the `warehouse` will be inferred from the connection string path itself.",
                    IcebergWarning,
                )
            warehouse = f"{parsed_conn.scheme}://{parsed_conn.netloc}{parsed_conn.path}"

        elif parsed_conn.scheme == "iceberg" or parsed_conn.scheme == "iceberg+file":
            catalog_type = "hadoop"
            if warehouse is not None:
                warnings.warn(
                    "The `warehouse` property in the connection string will be ignored when accessing a local Hadoop catalog. Instead, the `warehouse` will be inferred from the connection string path itself.",
                    IcebergWarning,
                )
            warehouse = f"{parsed_conn.netloc}{parsed_conn.path}"
        elif parsed_conn.scheme == "iceberg+snowflake":
            catalog_type = "snowflake"
        elif parsed_conn.scheme in {"iceberg+http", "iceberg+https"}:
            catalog_type = "rest"
        elif parsed_conn.scheme == "iceberg+arn" and "aws:s3tables" in parsed_conn.path:
            catalog_type = "s3tables"

        else:
            types = ", ".join(
                [
                    "hadoop-s3",
                    "hadoop",
                    "hive",
                    "nessie",
                    "glue",
                    "snowflake",
                    "rest",
                    "s3tables",
                ]
            )
            raise IcebergError(
                f"Cannot detect Iceberg catalog type from connection string:\n  {conn_str}\nIn the connection string, set the URL parameter `type` to one of the following:\n  {types}"
            )

    assert catalog_type in [
        "hadoop-s3",
        "hadoop-abfs",
        "hadoop",
        "hive",
        "nessie",
        "glue",
        "snowflake",
        "rest",
        "s3tables",
    ]

    # Get Warehouse Location
    if catalog_type not in ("snowflake", "s3tables") and warehouse is None:
        warnings.warn(
            "It is recommended that the `warehouse` property is included in the connection string for this type of catalog. Bodo can automatically infer what kind of FileIO to use from the warehouse location. It is also highly recommended to include with Glue and Nessie catalogs.",
            IcebergWarning,
        )

    # TODO: Pass more parsed results to use in java
    return catalog_type, warehouse


def gen_table_loc(
    catalog_type: CatalogType,
    warehouse: str,
    db_name: str,
    table_name: str,
) -> str:
    """
    Construct Table Location from Warehouse, Database Schema, and Table Name
    Note that this should only be used when creating a new table,
    as we have seen problems with guessing the location in the past

    TODO: Replace once we add PyIceberg
    """
    inner_name = (
        db_name + ".db" if catalog_type == "glue" or catalog_type == "hive" else db_name
    )

    return os.path.join(warehouse, inner_name, table_name)


def gen_file_loc(table_loc: str, db_name: str, table_name: str, file_name: str) -> str:
    """Construct Valid Paths for Files Written to Iceberg"""
    # S3 warehouse requires absolute file paths
    if (
        table_loc.startswith("s3a://")
        or table_loc.startswith("s3://")
        or table_loc.startswith("abfs://")
        or table_loc.startswith("abfss://")
    ):
        return os.path.join(table_loc, file_name)
    else:
        # TODO: Not sure if this is the best approach
        # How can we use the table or data location instead?
        return os.path.join(db_name, table_name, "data", file_name)


def normalize_loc(loc: str) -> str:
    return loc.replace("s3a://", "s3://").removeprefix("file:")


def normalize_data_loc(loc: str):
    loc = normalize_loc(loc)
    return os.path.join(loc, "data")
