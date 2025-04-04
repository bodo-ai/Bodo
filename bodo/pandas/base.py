import pandas as pd
from pandas._libs import lib

from bodo.ext import plan_optimizer
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""
    import pyarrow as pa

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    empty_df = df.iloc[:0]
    n_rows = len(df)
    arrow_schema = pa.Schema.from_pandas(df)

    # TODO: distribute to workers and get result_id
    plan = plan_optimizer.LazyPlan(
        plan_optimizer.LogicalGetPandasRead, df, arrow_schema
    )
    # TODO: Add support for Index

    return plan_optimizer.wrap_plan(empty_df, plan=plan, nrows=n_rows)


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
    from bodo.pandas import BODO_PANDAS_FALLBACK

    if (
        engine != "auto"
        or columns != None
        or storage_options != None
        or use_nullable_dtypes != lib.no_default
        or dtype_backend != lib.no_default
        or filesystem != None
        or filters != None
        or len(kwargs) > 0
    ):
        if BODO_PANDAS_FALLBACK != 0:
            return pd.read_parquet(
                path,
                engine=engine,
                columns=columns,
                storage_options=storage_options,
                use_nullable_dtypes=use_nullable_dtypes,
                dtype_backend=dtype_backend,
                filesystem=filesystem,
                filters=filters,
                **kwargs,
            )
        else:
            assert False and "Unsupported option to read_parquet"

    # Read Parquet schema and row count
    # TODO: Make this more robust (e.g. handle Index, etc.)
    use_hive = True
    pq_dataset = get_parquet_dataset(
        path,
        get_row_counts=True,
        storage_options=storage_options,
        read_categories=True,
        partitioning="hive" if use_hive else None,
    )
    arrow_schema = pq_dataset.schema
    nrows = pq_dataset._bodo_total_rows

    empty_df = pa.Table.from_pydict(
        {k: [] for k in arrow_schema.names}, schema=arrow_schema
    ).to_pandas()
    empty_df.index = pd.RangeIndex(0)

    plan = plan_optimizer.LazyPlan(
        plan_optimizer.LogicalGetParquetRead, path.encode(), arrow_schema
    )
    return plan_optimizer.wrap_plan(empty_df, plan=plan, nrows=nrows)


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
