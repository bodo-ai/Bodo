import pandas as pd
from pandas._libs import lib

from bodo.pandas import plan_optimizer
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.parquet import get_pandas_schema
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
    pr = plan_optimizer.LogicalGetParquetRead(path.encode())
    return plan_optimizer.wrap_plan(get_pandas_schema(path), pr)


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
