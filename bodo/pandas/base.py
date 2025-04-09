import pandas as pd
from pandas._libs import lib

from bodo.ext import plan_optimizer
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import LazyPlan, check_args_fallback


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""
    import pyarrow as pa

    import bodo

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # TODO: Add support for Index
    if (
        not isinstance(df.index, pd.RangeIndex)
        or df.index.start != 0
        or df.index.step != 1
    ):
        raise ValueError("Only RangeIndex with start=0 and step=1 is supported")

    empty_df = df.iloc[:0]
    n_rows = len(df)
    arrow_schema = pa.Schema.from_pandas(df)

    if bodo.dataframe_library_run_parallel:
        res_id = bodo.spawn.utils.scatter_data(df)
        plan = LazyPlan("LogicalGetPandasReadParallel", res_id, arrow_schema)
    else:
        plan = LazyPlan("LogicalGetPandasReadSeq", df, arrow_schema)

    return plan_optimizer.wrap_plan(empty_df, plan=plan, nrows=n_rows)


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
    import pyarrow as pa

    from bodo.io.parquet_pio import get_parquet_dataset

    # Read Parquet schema and row count
    # TODO: Make this more robust (e.g. handle Index, etc.)
    use_hive = True
    pq_dataset = get_parquet_dataset(
        path,
        get_row_counts=False,
        storage_options=storage_options,
        read_categories=True,
        partitioning="hive" if use_hive else None,
    )
    arrow_schema = pq_dataset.schema

    empty_df = pa.Table.from_pydict(
        {k: [] for k in arrow_schema.names}, schema=arrow_schema
    ).to_pandas()
    empty_df.index = pd.RangeIndex(0)

    # TODO: add args here
    if storage_options is None:
        storage_options = {}
    storage_options.update({"bodo_dummy": "dummy"})

    plan = LazyPlan(
        "LogicalGetParquetRead", path.encode(), arrow_schema, storage_options
    )
    return plan_optimizer.wrap_plan(empty_df, plan=plan, nrows=None)


def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)


def _empty_like(val):
    """Create an empty Pandas DataFrame or Series having the same schema as
    the given BodoDataFrame or BodoSeries
    """
    if isinstance(val, BodoDataFrame):
        return pd.DataFrame(
            {col: pd.Series(dtype=dt) for col, dt in val.dtypes.items()}
        )
    elif isinstance(val, BodoSeries):
        return pd.Series(dtype=val.dtype)
    else:
        assert False & f"_empty_like cannot create empty object like type {type(val)}"
