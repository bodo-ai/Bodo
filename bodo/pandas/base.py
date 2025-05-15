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
    wrap_plan,
)


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""

    import bodo

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Make sure empty_df has proper dtypes since used in the plan output schema.
    # Using sampling to avoid large memory usage.
    sample_size = 100

    # TODO [BSE-4788]: Refactor with convert_to_arrow_dtypes util
    empty_df = arrow_to_empty_df(pa.Schema.from_pandas(df.iloc[:sample_size]))
    n_rows = len(df)

    res_id = None
    if bodo.dataframe_library_run_parallel:
        res_id = bodo.spawn.utils.scatter_data(df)
        plan = LazyPlan(
            "LogicalGetPandasReadParallel",
            empty_df,
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
    from bodo.io.parquet_pio import get_parquet_dataset

    if storage_options is None:
        storage_options = {}

    # Read Parquet schema
    use_hive = True
    pq_dataset = get_parquet_dataset(
        path,
        get_row_counts=False,
        storage_options=storage_options,
        read_categories=True,
        partitioning="hive" if use_hive else None,
    )
    arrow_schema = pq_dataset.schema

    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LazyPlan("LogicalGetParquetRead", empty_df, path, storage_options)
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
