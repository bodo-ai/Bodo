import json
import os

from bodo_iceberg_connector.py4j_support import (
    catalog_dict,
    get_bodo_iceberg_handler_class,
    get_core_site_path,
)


def prefetch_sf_tables(
    conn_str: str, table_paths: list[str], verbose_level: int = 0
) -> None:
    """
    Prefetch the metadata path for a list of Snowflake-managed Iceberg tables
    Used for internal BodoSQL code generation

    Args:
        conn_str (str): Snowflake connection string
        table_paths (list[str]): List of fully qualified table paths to prefetch
        verbose_level (int, optional): Print logs for debugging
    """
    reader_class = get_bodo_iceberg_handler_class()
    # If the catalog object is not in the cache, create a new one
    # We can assume that this occurs when executing SQL from cache
    # When executing after compilation, cache should be populated
    if conn_str not in catalog_dict:
        created_core_site = get_core_site_path()
        # Use the defaults if the user didn't override the core site.
        core_site = created_core_site if os.path.exists(created_core_site) else ""
        catalog_dict[conn_str] = reader_class.buildPrefetcher(
            conn_str,
            "snowflake",
            core_site,
            # Py4J has trouble passing list of strings, so jsonify between Python and Java
            json.dumps(table_paths),
            verbose_level,
        )
