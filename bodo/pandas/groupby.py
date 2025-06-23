"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

import typing as pt
import warnings
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.inference import is_dict_like, is_list_like

from bodo.pandas.utils import (
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    LazyPlan,
    check_args_fallback,
    make_col_ref_exprs,
    wrap_plan,
)

if pt.TYPE_CHECKING:
    from bodo.pandas import BodoDataFrame, BodoSeries


class DataFrameGroupBy:
    """
    Similar to pandas DataFrameGroupBy. See Pandas code for reference:
    https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/groupby/generic.py#L1329
    """

    def __init__(
        self,
        obj: pd.DataFrame,
        keys: list[str],
        as_index: bool = True,
        dropna: bool = True,
        selection: list[str] | None = None,
    ):
        self._obj = obj
        self._keys = keys
        self._as_index = as_index
        self._dropna = dropna
        self._selection = (
            selection
            if selection is not None
            else list(filter(lambda col: col not in keys, obj.columns))
        )

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        """
        Return a DataFrameGroupBy or SeriesGroupBy for the selected data columns.
        """
        if isinstance(key, str):
            return SeriesGroupBy(
                self._obj, self._keys, [key], self._as_index, self._dropna
            )
        elif isinstance(key, list) and all(isinstance(key_, str) for key_ in key):
            return DataFrameGroupBy(
                self._obj, self._keys, self._as_index, self._dropna, selection=key
            )
        else:
            raise BodoLibNotImplementedException(
                f"DataFrameGroupBy: Invalid key type: {type(key)}"
            )

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"DataFrameGroupBy.{name} is not "
                "implemented in Bodo dataframe library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            warnings.warn(BodoLibFallbackWarning(msg))
            gb = pd.DataFrame(self._obj).groupby(
                self._keys, as_index=self._as_index, dropna=self._dropna
            )
            if self._selection is not None:
                gb = gb[self._selection]
            return object.__getattribute__(gb, name)

    @check_args_fallback(supported="func")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        return _groupby_agg_plan(self, func, *args, **kwargs)

    agg = aggregate

    def _normalize_agg_func(self, func, kwargs: dict) -> list[tuple[str, str]]:
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        # list of (input column name, function) pairs
        normalized_func: list[tuple[str, str]] = []

        if func is None and kwargs:
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for col, func_ in kwargs.values()
            ]
        elif is_dict_like(func):
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for col, func_ in func.items()
            ]
        elif is_list_like(func):
            normalized_func = [
                (col, _get_aggfunc_str(func_))
                for col in self._selection
                for func_ in func
            ]
        else:
            func = _get_aggfunc_str(func)
            normalized_func = [(col, func) for col in self._selection]

        return normalized_func

    @check_args_fallback(supported="none")
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute the sum of each group.
        """
        return _groupby_agg_plan(self, "sum")

    @check_args_fallback(supported="none")
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute the sum of each group.
        """
        return _groupby_agg_plan(self, "mean")

    @check_args_fallback(supported="none")
    def count(self):
        """
        Compute the sum of each group.
        """
        return _groupby_agg_plan(self, "count")


class SeriesGroupBy:
    """
    Similar to pandas SeriesGroupBy.
    """

    def __init__(
        self,
        obj: pd.DataFrame,
        keys: list[str],
        selection: list[str],
        as_index: bool,
        dropna: bool,
    ):
        self._obj = obj
        self._keys = keys
        self._selection = selection
        self._as_index = as_index
        self._dropna = dropna

    @check_args_fallback(supported="none")
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute the sum of each group.
        """
        assert len(self._selection) == 1, (
            "SeriesGroupBy.sum() should only be called on a single column selection."
        )

        return _groupby_agg_plan(self, "sum")

    @check_args_fallback(supported="none")
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute the sum of each group.
        """
        assert len(self._selection) == 1, (
            "SeriesGroupBy.sum() should only be called on a single column selection."
        )

        return _groupby_agg_plan(self, "mean")

    @check_args_fallback(supported="none")
    def count(self):
        """
        Compute the sum of each group.
        """
        assert len(self._selection) == 1, (
            "SeriesGroupBy.sum() should only be called on a single column selection."
        )

        return _groupby_agg_plan(self, "count")

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"SeriesGroupBy.{name} is not "
                "implemented in Bodo dataframe library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            warnings.warn(BodoLibFallbackWarning(msg))
            gb = pd.DataFrame(self._obj).groupby(self._keys)[self._selection[0]]
            return object.__getattribute__(gb, name)

    @check_args_fallback(supported="func")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        return _groupby_agg_plan(self, func, *args, **kwargs)

    agg = aggregate

    def _normalize_agg_func(self, func, kwargs):
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        col = self._selection[0]

        # list of (input column name, function) pairs
        normalized_func: list[tuple[str, str]] = []
        if func is None and kwargs:
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for func_ in kwargs.values()
            ]
        elif is_dict_like(func):
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for func_ in func.values()
            ]
        elif is_list_like(func):
            normalized_func = [(col, _get_aggfunc_str(func_)) for func_ in func]
        else:
            normalized_func = [(col, _get_aggfunc_str(func))]

        return normalized_func


def _groupby_agg_plan(
    grouped: SeriesGroupBy | DataFrameGroupBy, func, *args, **kwargs
) -> BodoSeries | BodoDataFrame:
    """Compute groupby.func() on the Series or DataFrame GroupBy object."""
    from bodo.pandas.base import _empty_like

    zero_size_df = _empty_like(grouped._obj)
    empty_data_pandas = zero_size_df.groupby(grouped._keys, as_index=grouped._as_index)[
        grouped._selection[0]
        if isinstance(grouped, SeriesGroupBy)
        else grouped._selection
    ].agg(func, *args, **kwargs)

    func = grouped._normalize_agg_func(func, kwargs)

    n_key_cols = 0 if grouped._as_index else len(grouped._keys)
    empty_data = _cast_groupby_agg_columns(
        func, zero_size_df, empty_data_pandas, n_key_cols
    )

    key_indices = [grouped._obj.columns.get_loc(c) for c in grouped._keys]

    exprs = [
        LazyPlan(
            "AggregateExpression",
            empty_data.iloc[:, i]
            if isinstance(empty_data, pd.DataFrame)
            else empty_data,
            grouped._obj._plan,
            func_,
            [grouped._obj.columns.get_loc(col)],
            grouped._dropna,
        )
        for i, (
            col,
            func_,
        ) in enumerate(func)
    ]

    plan = LazyPlan(
        "LogicalAggregate",
        empty_data,
        grouped._obj._plan,
        key_indices,
        exprs,
    )

    # Add the data column then the keys since they become Index columns in output.
    # DuckDB generates keys first in output so we need to reverse the order.
    if grouped._as_index:
        col_indices = list(range(len(grouped._keys), len(grouped._keys) + len(func)))
        col_indices += list(range(len(grouped._keys)))

        exprs = make_col_ref_exprs(col_indices, plan)
        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            plan,
            exprs,
        )

    return wrap_plan(plan)


def _get_aggfunc_str(func):
    """Gets the name of a callable func"""
    from pandas.core.common import get_callable_name

    if isinstance(func, str):
        return func
    elif callable(func):
        return get_callable_name(func)

    raise TypeError(
        "GroupBy.agg(): expected func to be callable or string, got: ", type(func)
    )


def _get_agg_output_type(func: str, pa_type: pa.DataType, col_name: str) -> pa.DataType:
    """Gets the output type of an aggregation or raise either TypeError or BodoLibNotImplementedException
    for unsupported pa_type/func combinations. Should closely match
    https://github.com/bodo-ai/Bodo/blob/d1133e257662348cc7b9ef52cf445633036133d2/bodo/libs/groupby/_groupby_common.cpp#L562
    """
    new_type = pa_type
    if func == "sum":
        if pa.types.is_signed_integer(pa_type) or pa.types.is_boolean(pa_type):
            new_type = pa.int64()
        elif pa.types.is_unsigned_integer(pa_type):
            new_type = pa.uint64()
        elif pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_string(pa_type):
            new_type = pa_type
        elif pa.types.is_decimal(pa_type):
            # TODO Support Decimal columns for sum()
            raise BodoLibNotImplementedException(
                f"GroupBy.sum() on decimal column '{col_name}' not supported yet."
            )
        else:
            raise TypeError(
                f"GroupBy.sum(): Unsupported dtype in column '{col_name}': {pa_type}."
            )
        return new_type
    elif func == "mean":
        if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_boolean(pa_type):
            # TODO Support bool columns for mean()
            raise BodoLibNotImplementedException(
                f"GroupBy.mean() on boolean column '{col_name}' not supported yet."
            )
        elif pa.types.is_decimal(pa_type):
            # TODO Support decimal columns for mean()
            raise BodoLibNotImplementedException(
                f"GroupBy.mean() on boolean column '{col_name}' not supported yet."
            )
        else:
            raise TypeError(
                f"GroupBy.mean(): Unsupported dtype in column '{col_name}': {pa_type}."
            )
        return new_type
    elif func == "count":
        return pa.int64()

    raise BodoLibNotImplementedException("Unsupported aggregate function: ", func)


def _cast_groupby_agg_columns(
    func: list[tuple[str, str]] | str,
    in_data: pd.Series | pd.DataFrame,
    out_data: pd.Series | pd.DataFrame,
    n_key_cols: int,
) -> pd.Series | pd.DataFrame:
    """
    Casts dtypes in the output of GroupBy.agg() to the correct type for aggregation.

    Args:
        func : A list of (col, func) pairs where col is the name of the column in the
            input DataFrame to which func is applied.
        out_data : An empty DataFrame/Series with the same shape as the aggregate
            output
        in_data : An empty DataFrame/Series with the same shape as the input to the
            aggregation.
        n_key_cols : Number of grouping keys in the output.

    Returns:
        pd.Series | pd.DataFrame: A DataFrame or Series with the dtypes casted depending
            on the aggregate functions.
    """

    if isinstance(out_data, pd.Series):
        col, func = func[0]
        in_data = in_data[col]
        new_type = _get_agg_output_type(
            func, in_data.dtype.pyarrow_dtype, out_data.name
        )
        out_data = out_data.astype(pd.ArrowDtype(new_type))
        return out_data

    for i, (in_col_name, func_) in enumerate(func):
        out_col_name = out_data.columns[i + n_key_cols]

        if not isinstance(out_data[out_col_name], pd.Series):
            raise BodoLibNotImplementedException(
                f"GroupBy.agg(): detected duplicate output column name in output columns: '{out_col_name}'"
            )

        in_col = in_data[in_col_name]
        if not isinstance(out_data[out_col_name], pd.Series):
            raise BodoLibNotImplementedException(
                f"GroupBy.agg(): detected duplicate column name in input column: '{in_col_name}'"
            )

        new_type = _get_agg_output_type(func_, in_col.dtype.pyarrow_dtype, in_col_name)
        out_data[out_col_name] = out_data[out_col_name].astype(pd.ArrowDtype(new_type))

    return out_data
