"""
Helper code for:
- Constructing Iceberg catalogs from connection strings
- Additional Iceberg catalogs supported on the Java side
  but currently have no PyIceberg equivalent
"""

from __future__ import annotations

import typing as pt
from urllib.parse import parse_qs, urlparse

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.catalog import Catalog


def validate_conn_str(conn_str: str) -> None:
    parse_res = urlparse(conn_str)
    if not conn_str.startswith("iceberg+glue") and parse_res.scheme not in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
        "iceberg+snowflake",
        "iceberg+abfs",
        "iceberg+abfss",
        "iceberg+rest",
        "iceberg+arn",
    ):
        raise ValueError(
            "Iceberg connection strings must start with one of the following: \n"
            "  Hadoop / Directory Catalog: 'iceberg://', 'iceberg+file://', 'iceberg+s3://', 'iceberg+abfs://', 'iceberg+abfss://'\n"
            "  REST Catalog: 'iceberg+http://', 'iceberg+https://', 'iceberg+rest://'\n"
            "  Glue Catalog: 'iceberg+glue'\n"
            "  Hive Catalog: 'iceberg+thrift://'\n"
            "  Snowflake Catalog: 'iceberg+snowflake://'\n"
            "  S3 Tables Catalog: 'iceberg+arn'\n"
            f"Checking '{conn_str}' ('{parse_res.scheme}')"
        )


def conn_str_to_catalog(conn_str: str) -> Catalog:
    """TODO"""
    from pyiceberg.catalog import URI, WAREHOUSE_LOCATION

    validate_conn_str(conn_str)
    parse_res = urlparse(conn_str)

    # Property Parsing
    parsed_props = parse_qs(parse_res.query)
    if any(len(x) > 1 for x in parsed_props.values()):
        raise ValueError("Multiple values for a single property are not supported")
    properties = {key: val[0] for key, val in parsed_props.items()}

    # Constructing the base url (without properties or the iceberg+ prefix)
    # Useful for most catalogs
    base_url = (
        f"{parse_res.netloc}{parse_res.path}"
        if parse_res.scheme == "iceberg"
        else f"{parse_res.scheme.removeprefix('iceberg+')}://{parse_res.netloc}{parse_res.path}"
    )

    catalog: type[Catalog]
    if conn_str.startswith("iceberg+glue"):
        from pyiceberg.catalog.glue import GlueCatalog

        catalog = GlueCatalog

    match parse_res.scheme:
        case (
            "iceberg"
            | "iceberg+file"
            | "iceberg+s3"
            | "iceberg+abfs"
            | "iceberg+abfss"
        ):
            from .dir import DirCatalog

            catalog = DirCatalog
            properties[WAREHOUSE_LOCATION] = base_url
        case "iceberg+http" | "iceberg+https" | "iceberg+rest":
            from pyiceberg.catalog.rest import RestCatalog

            catalog = RestCatalog
            properties[URI] = base_url
        case "iceberg+thrift":
            from pyiceberg.catalog.hive import HiveCatalog

            catalog = HiveCatalog
            properties[URI] = base_url
        case "iceberg+arn":
            from .s3_tables import S3TablesCatalog

            catalog = S3TablesCatalog
        case "iceberg+snowflake":
            from .snowflake import SnowflakeCatalog

            catalog = SnowflakeCatalog
            properties[URI] = base_url

    return catalog("catalog", **properties)
