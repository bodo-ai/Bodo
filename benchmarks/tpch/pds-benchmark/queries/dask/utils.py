from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
from dask.distributed import Client
from queries.common_utils import (
    check_query_result_pd,
    get_table_path,
    run_query_generic,
)
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.dataframe import DataFrame

settings = Settings()


def read_ds(table_name: str) -> DataFrame:
    if settings.run.io_type == "skip":
        # TODO: Load into memory before returning the Dask DataFrame.
        # Code below is tripped up by date types
        # df = pd.read_parquet(path, dtype_backend="pyarrow")
        # return dd.from_pandas(df, npartitions=os.cpu_count())
        msg = "cannot run Dask starting from an in-memory representation"
        raise RuntimeError(msg)

    path = get_table_path(table_name)

    if settings.run.io_type == "parquet":
        # Bodo Change: use default dtype backend
        df = dd.read_parquet(path).rename(columns=str.lower)
    elif settings.run.io_type == "csv":
        df = dd.read_csv(path, dtype_backend="pyarrow")
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        raise ValueError(msg)

    for c in df.columns:
        if c.endswith("date"):
            df[c] = df[c].astype("date32[day][pyarrow]")
    return df  # type: ignore[no-any-return]


# Bodo Change: Removed decorators (always include IO)
def get_line_item_ds() -> DataFrame:
    return read_ds("lineitem")


def get_orders_ds() -> DataFrame:
    return read_ds("orders")


def get_customer_ds() -> DataFrame:
    return read_ds("customer")


def get_region_ds() -> DataFrame:
    return read_ds("region")


def get_nation_ds() -> DataFrame:
    return read_ds("nation")


def get_supplier_ds() -> DataFrame:
    return read_ds("supplier")


def get_part_ds() -> DataFrame:
    return read_ds("part")


def get_part_supp_ds() -> DataFrame:
    return read_ds("partsupp")


def run_query(query_number: int, query: Callable[..., Any]) -> None:
    with Client():  # Use default LocalCluster settings
        run_query_generic(
            query, query_number, "dask", query_checker=check_query_result_pd
        )
