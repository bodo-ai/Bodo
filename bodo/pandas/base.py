import typing as pt

import pandas as pd
import pyarrow as pa
from pandas._libs import lib

from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import (
    BODO_NONE_DUMMY,
    LazyPlan,
    LazyPlanDistributedArg,
    arrow_to_empty_df,
    check_args_fallback,
    ensure_datetime64ns,
    make_col_ref_exprs,
    wrap_plan,
)


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""

    import bodo

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Avoid datetime64[us] that is commonly used in Pandas but not supported in Bodo.
    df = ensure_datetime64ns(df)

    # Make sure empty_df has proper dtypes since used in the plan output schema.
    # Using sampling to avoid large memory usage.
    sample_size = 100

    # TODO [BSE-4788]: Refactor with convert_to_arrow_dtypes util
    pa_schema = pa.Schema.from_pandas(df.iloc[:sample_size])
    empty_df = arrow_to_empty_df(pa_schema)
    n_rows = len(df)

    res_id = None
    if bodo.dataframe_library_run_parallel:
        nrows = len(df)
        res_id = bodo.spawn.utils.scatter_data(df)
        plan = LazyPlan(
            "LogicalGetPandasReadParallel",
            empty_df,
            nrows,
            LazyPlanDistributedArg(None, res_id),
        )
    else:
        plan = LazyPlan("LogicalGetPandasReadSeq", empty_df, df)

    return wrap_plan(plan=plan, nrows=n_rows, res_id=res_id)


@check_args_fallback("all")
def read_parquet(
    path,
    engine="auto",
    columns=None,
    storage_options=None,
    use_nullable_dtypes=lib.no_default,
    dtype_backend=lib.no_default,
    filesystem=None,
    filters=None,
    **kwargs,
):
    from bodo.io.parquet_pio import get_dataset_unify_nulls

    if storage_options is None:
        storage_options = {}

    # Read Parquet schema
    use_hive = True
    pq_dataset = get_dataset_unify_nulls(
        path,
        storage_options,
        "hive" if use_hive else None,
    )
    arrow_schema = pq_dataset.schema

    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LazyPlan(
        "LogicalGetParquetRead",
        empty_df,
        path,
        storage_options,
    )
    return wrap_plan(plan=plan)


def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)


def _empty_like(val):
    """Create an empty Pandas DataFrame or Series having the same schema as
    the given BodoDataFrame or BodoSeries
    """
    import pyarrow as pa

    if not isinstance(val, (BodoDataFrame, BodoSeries)):
        raise TypeError(f"val must be a BodoDataFrame or BodoSeries, got {type(val)}")

    is_series = isinstance(val, BodoSeries)
    # Avoid triggering data collection
    val = val.head(0)

    if is_series:
        val = val.to_frame(name=BODO_NONE_DUMMY if val.name is None else val.name)

    # Reuse arrow_to_empty_df to make sure details like Index handling are correct
    out = arrow_to_empty_df(pa.Schema.from_pandas(val))

    if is_series:
        out = out.iloc[:, 0]

    return out


@check_args_fallback(
    supported=[
        "catalog_name",
        "catalog_properties",
        "selected_fields",
        "limit",
        "row_filter",
        "snapshot_id",
    ]
)
def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None,
    catalog_properties: dict[str, pt.Any] | None = None,
    row_filter: str | None = None,
    selected_fields: tuple[str] | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    limit: int | None = None,
    scan_properties: dict[str, pt.Any] | None = None,
) -> BodoDataFrame:
    import pyiceberg.catalog
    import pyiceberg.expressions
    import pyiceberg.table

    if catalog_properties is None:
        catalog_properties = {}
    catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)

    # Get the output schema
    table = catalog.load_table(table_identifier)
    pyiceberg_schema = table.schema()
    arrow_schema = pyiceberg_schema.as_arrow()
    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LazyPlan(
        "LogicalGetIcebergRead",
        empty_df,
        table_identifier,
        catalog_name,
        catalog_properties,
        pyiceberg.table._parse_row_filter(row_filter)
        if row_filter
        else pyiceberg.expressions.AlwaysTrue(),
        # We need to pass the pyiceberg schema so we can bind the iceberg filter to it
        # during filter conversion. See bodo/io/iceberg/common.py::pyiceberg_filter_to_pyarrow_format_str_and_scalars
        pyiceberg_schema,
        snapshot_id if snapshot_id is not None else -1,
        __pa_schema=arrow_schema,
    )

    if selected_fields is not None:
        col_idxs = {
            arrow_schema.get_field_index(field_name) for field_name in selected_fields
        }
        exprs = make_col_ref_exprs(col_idxs, plan)
        empty_df = empty_df[list(selected_fields)]
        plan = LazyPlan(
            "LogicalProjection",
            empty_df,
            plan,
            exprs,
        )

    if limit is not None:
        plan = LazyPlan(
            "LogicalLimit",
            empty_df,
            plan,
            limit,
        )

    return wrap_plan(plan=plan)
