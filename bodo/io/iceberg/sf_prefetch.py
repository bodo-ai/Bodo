"""
Helper code related to Snowflake-managed Iceberg tables
In particular, special optimizations to reduce the communication overhead
of fetching metadata for Snowflake-managed Iceberg tables from Snowflake.
"""

from __future__ import annotations

from numba.extending import overload

import bodo
from bodo.io.iceberg.common import format_iceberg_conn
from bodo.mpi4py import MPI
from bodo.utils.utils import BodoError


def prefetch_sf_tables(
    conn_str: str, table_paths: list[str], verbose_level: int
) -> None:
    "Helper function for the Python contents of prefetch_sf_tables_njit."
    import bodo_iceberg_connector as bic

    comm = MPI.COMM_WORLD
    exc = None
    conn_str = format_iceberg_conn(conn_str)
    if bodo.get_rank() == 0:
        try:
            bic.prefetch_sf_tables(conn_str, table_paths, verbose_level)
        except bic.IcebergError as e:
            exc = BodoError(
                f"Failed to prefetch Snowflake-managed Iceberg table paths: {e.message}"
            )
    exc = comm.bcast(exc)
    if exc is not None:
        raise exc


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
def overload_prefetch_sf_tables_njit(conn_str, table_paths, verbose_level):
    def impl(conn_str, table_paths, verbose_level):
        with bodo.no_warning_objmode():
            prefetch_sf_tables(conn_str, table_paths, verbose_level)

    return impl
