"""
Helper code for:
- Constructing Iceberg catalogs from connection strings
- Additional Iceberg catalogs supported on the Java side
  but currently have no PyIceberg equivalent
"""

from __future__ import annotations

import re
import typing as pt
from typing import cast
from urllib.parse import parse_qs, urlparse

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.catalog import Catalog

# iceberg+arn:aws:s3tables:<region>:<account_number>:bucket/<bucket>
S3_TABLES_PAT = re.compile(
    r"^iceberg\+arn:aws:s3tables:([a-z0-9-]+):([0-9]+):bucket/([a-z0-9-]+)$"
)

CATALOG_CACHE: dict[str, Catalog] = {}


def conn_str_to_catalog(conn_str: str) -> Catalog:
    """
    Construct a PyIceberg catalog from a connection string
    """

    import pyiceberg.utils.config
    from pyiceberg.catalog import URI, WAREHOUSE_LOCATION
    from pyiceberg.catalog.glue import GLUE_ID, GLUE_REGION
    from pyiceberg.catalog.rest import OAUTH2_SERVER_URI
    from pyiceberg.io import AWS_REGION
    from pyiceberg.typedef import RecursiveDict

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

        glue_id = properties.get(GLUE_ID)
        region = properties.get(GLUE_REGION, properties.get(AWS_REGION))
        cache_key = "glue"
        if glue_id:
            cache_key += f"_{glue_id}"
        if region:
            cache_key += f"_{region}"

    else:
        if parse_res.scheme in (
            "iceberg",
            "iceberg+file",
            "iceberg+s3",
            "iceberg+abfs",
            "iceberg+abfss",
            "iceberg+gs",
        ):
            from .dir import DirCatalog

            catalog = DirCatalog
            properties[WAREHOUSE_LOCATION] = base_url.removeprefix("file://")
            cache_key = base_url

        elif parse_res.scheme in ("iceberg+http", "iceberg+https", "iceberg+rest"):
            from pyiceberg.catalog.rest import RestCatalog

            catalog = RestCatalog
            properties[URI] = base_url
            properties[OAUTH2_SERVER_URI] = (
                base_url + ("/" if base_url[-1] != "/" else "") + "v1/oauth/tokens"
            )
            cache_key = (
                base_url
                + properties[OAUTH2_SERVER_URI]
                + properties[WAREHOUSE_LOCATION]
                + properties.get("scope", "")
            )

        elif parse_res.scheme == "iceberg+thrift":
            from pyiceberg.catalog.hive import HiveCatalog

            catalog = HiveCatalog
            properties[URI] = base_url
            cache_key = base_url

        elif parse_res.scheme == "iceberg+arn":
            from .s3_tables import (
                S3TABLES_REGION,
                S3TABLES_TABLE_BUCKET_ARN,
                S3TablesCatalog,
            )

            catalog = S3TablesCatalog
            properties[S3TABLES_TABLE_BUCKET_ARN] = (
                f"arn:{parse_res.netloc}{parse_res.path}"
            )
            parsed = re.match(S3_TABLES_PAT, conn_str)
            if not parsed:
                raise ValueError(f"Invalid S3 Tables ARN: {conn_str}")
            properties[S3TABLES_REGION] = parsed.group(1)
            cache_key = properties[S3TABLES_TABLE_BUCKET_ARN]

        elif parse_res.scheme == "iceberg+snowflake":
            from bodo.io.utils import parse_snowflake_conn_str

            from .snowflake import SnowflakeCatalog

            catalog = SnowflakeCatalog
            properties[URI] = base_url
            # Need to extract properties from the connection string and add them
            comps = parse_snowflake_conn_str(base_url)
            properties = {**properties, **comps}
            cache_key = comps["account"]

        else:
            raise ValueError(
                "Iceberg connection strings must start with one of the following: \n"
                "  Hadoop / Directory Catalog: 'iceberg://', 'iceberg+file://', 'iceberg+s3://', 'iceberg+abfs://', 'iceberg+abfss://', 'iceberg+gs://'\n"
                "  REST Catalog: 'iceberg+http://', 'iceberg+https://', 'iceberg+rest://'\n"
                "  Glue Catalog: 'iceberg+glue'\n"
                "  Hive Catalog: 'iceberg+thrift://'\n"
                "  Snowflake-Managed Iceberg Tables: 'iceberg+snowflake://'\n"
                "  S3 Tables Catalog: 'iceberg+arn'\n"
                f"Checking '{conn_str}' ('{parse_res.scheme}')"
            )

    if cache_key in CATALOG_CACHE:
        return CATALOG_CACHE[cache_key]
    else:
        pyiceberg_config = pyiceberg.utils.config.Config()
        catalog_name = properties.get(
            WAREHOUSE_LOCATION, pyiceberg_config.get_default_catalog_name()
        )
        merged_conf = pyiceberg.utils.config.merge_config(
            pyiceberg_config.get_catalog_config(catalog_name) or {},
            cast(RecursiveDict, properties),
        )
        cat_inst = catalog(catalog_name, **cast(dict[str, str], merged_conf))
        CATALOG_CACHE[cache_key] = cat_inst
        return cat_inst
