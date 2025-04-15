import pandas as pd
from pandas._libs import lib

from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import (
    LazyPlan,
    arrow_to_empty_df,
    check_args_fallback,
    wrap_plan,
)


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""

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

    # Make sure empty_df has proper dtypes since used in the plan output schema.
    # Using sampling to avoid large memory usage.
    sample_size = 100
    empty_df = df.iloc[:sample_size].convert_dtypes(dtype_backend="pyarrow").iloc[:0]
    n_rows = len(df)

    res_id = None
    if bodo.dataframe_library_run_parallel:
        res_id = bodo.spawn.utils.scatter_data(df)
        plan = LazyPlan("LogicalGetPandasReadParallel", res_id)
    else:
        plan = LazyPlan("LogicalGetPandasReadSeq", df)

    return wrap_plan(empty_df, plan=plan, nrows=n_rows, res_id=res_id)


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
    from bodo.io.parquet_pio import get_pandas_metadata, get_parquet_dataset

    if storage_options is None:
        storage_options = {}

    # Read Parquet schema
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
    partition_names = pq_dataset.partition_names

    if len(partition_names) > 0:
        raise NotImplementedError(
            "bd.read_parquet: Reading parquet with partition column not supported yet."
        )

    index_cols, _ = get_pandas_metadata(arrow_schema)
    is_supported_index = False

    if len(index_cols) == 0:
        is_supported_index = True
    elif len(index_cols) == 1:
        index_col = index_cols[0]
        # RangeIndex case
        if (
            isinstance(index_col, dict)
            and index_col["start"] == 0
            and index_col["step"] == 1
        ):
            is_supported_index = True

    if not is_supported_index:
        raise NotImplementedError(
            "bd.read_parquet: Reading parquet files with index columns is not supported yet."
        )

    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LazyPlan("LogicalGetParquetRead", path.encode(), storage_options)
    return wrap_plan(empty_df, plan=plan)


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
