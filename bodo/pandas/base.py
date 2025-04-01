import pandas as pd
from pandas._libs import lib

from bodo.ext import plan_optimizer
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.series import BodoSeries


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

    metadata = LazyMetadata("dummy", empty_df, nrows, None)
    return BodoDataFrame.from_lazy_metadata(
        metadata, plan=plan_optimizer.LogicalGetParquetRead(path.encode(), arrow_schema)
    )


def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)


def empty_like(val):
    """Create an empty DataFrame or Series from the given DataFrame or Series"""
    if isinstance(val, BodoDataFrame):
        return pd.DataFrame(
            {col: pd.Series(dtype=dt) for col, dt in val.dtypes.items()}
        )
    elif isinstance(val, BodoSeries):
        return pd.Series(dtype=val.dtype)
    else:
        assert False & f"empty_like cannot create empty object like type {type(val)}"
