from pathlib import Path
from typing import Any

import duckdb
from queries.common_utils import (
    check_query_result_pl,
    get_table_path,
    run_query_generic,
)
from settings import Settings

settings = Settings()
_connection = None


def _scan_ds(table_name: str) -> str:
    path_str = get_table_path(table_name)

    if settings.run.io_type == "skip":
        name = path_str.replace("/", "_").replace(".", "_").replace("-", "_")
        duckdb.sql(
            f"create temp table if not exists {name} as select * from read_parquet('{path_str}');"
        )
        return name
    elif settings.run.io_type == "duckdb":
        return table_name
    elif settings.run.io_type == "parquet" or settings.run.io_type == "csv":
        # Bodo Change: read parquet
        return f"read_parquet('{path_str}')"
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        raise ValueError(msg)


def get_line_item_ds() -> str:
    return _scan_ds("lineitem")


def get_orders_ds() -> str:
    return _scan_ds("orders")


def get_customer_ds() -> str:
    return _scan_ds("customer")


def get_region_ds() -> str:
    return _scan_ds("region")


def get_nation_ds() -> str:
    return _scan_ds("nation")


def get_supplier_ds() -> str:
    return _scan_ds("supplier")


def get_part_ds() -> str:
    return _scan_ds("part")


def get_part_supp_ds() -> str:
    return _scan_ds("partsupp")


def get_persistent_path() -> str:
    return str(Path(get_table_path("lineitem")).parent / Path("tpch.db"))


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        if settings.run.io_type == "duckdb":
            # connect to persistent db
            _connection = duckdb.connect(get_persistent_path())
        elif settings.run.io_type == "skip":
            _connection = duckdb.connect(":default:")
        else:
            # connect to in-memory db
            _connection = duckdb.connect()
    return _connection


def run_query(query_number: int, query: str) -> None:
    conn = get_connection()
    if settings.run.show_results:

        def execute() -> Any:
            print(conn.sql(query))
    elif settings.run.check_results:

        def execute() -> Any:
            return conn.sql(query).pl()
    else:

        def execute() -> Any:
            conn.sql(query).fetchall()

    run_query_generic(
        execute, query_number, "duckdb", query_checker=check_query_result_pl
    )
