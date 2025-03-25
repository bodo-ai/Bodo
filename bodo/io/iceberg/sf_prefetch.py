"""
Helper code related to Snowflake-managed Iceberg tables
In particular, special optimizations to reduce the communication overhead
of fetching metadata for Snowflake-managed Iceberg tables from Snowflake.
"""

from __future__ import annotations

from numba.extending import overload

import bodo
from bodo.utils.utils import run_rank0


@run_rank0
def prefetch_sf_tables(conn_str: str, table_paths: list[str]):
    "Helper function for the Python contents of prefetch_sf_tables_njit."
    from bodo.io.iceberg.catalog import conn_str_to_catalog
    from bodo.io.iceberg.catalog.snowflake import SnowflakeCatalog

    sf_catalog = conn_str_to_catalog(conn_str)
    assert isinstance(sf_catalog, SnowflakeCatalog)

    sf_catalog.prefetch_metadata_paths(table_paths)


def prefetch_sf_tables_njit(
    conn_str: str, table_paths: list[str], verbose_level: int
) -> None:
    """
    Prefetch the metadata path for a list of Snowflake-managed Iceberg tables.
    This function is called in parallel across all ranks. It is mainly used
    for SQL code generation.

    Args:
        conn_str (str): Snowflake connection string to connect to.
        table_paths (list[str]): List of table paths to prefetch paths for.
    """
    pass


@overload(prefetch_sf_tables_njit)
def overload_prefetch_sf_tables_njit(conn_str, table_paths):
    def impl(conn_str, table_paths):
        with bodo.no_warning_objmode():
            prefetch_sf_tables(conn_str, table_paths)

    return impl
