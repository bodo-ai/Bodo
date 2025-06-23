"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

import typing as pt
import warnings
from typing import Any, Literal

import pandas as pd
import pyarrow as pa

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
        from pandas.core.apply import reconstruct_func
        from pandas.core.dtypes.inference import is_dict_like, is_list_like

        from bodo.pandas.base import _empty_like

        zero_size_df = _empty_like(self._obj)
        empty_data = zero_size_df.groupby(self._keys, as_index=self._as_index)[
            self._selection
        ].agg(func, *args, **kwargs)

        # Reconstruct func from keyword args, if no keyword args, leaves
        # func unchanged
        _, func, _, order = reconstruct_func(func, **kwargs)

        # assuming func is strings or dictlike of listlike with inner funcs that are strings
        # list of (function, column)
        normalized_func: list[tuple[str, str]] = []
        if is_dict_like(func):
            if order is not None:
                func_flattened = [
                    (col, func_) for (col, funcs) in func.items() for func_ in funcs
                ]
                for idx in order:
                    col, func_ = func_flattened[idx]
                    normalized_func.append((func_, col))
            else:
                normalized_func = [(func_, col) for (col, func_) in func.items()]
        elif is_list_like(func):
            normalized_func = [
                (func_, col) for col in self._selection for func_ in func
            ]
        elif isinstance(func, str):
            normalized_func = [(func, col) for col in self._selection]
        else:
            raise TypeError(
                "DataFrameGroupby.agg(): Unsupported type for func:", {func}
            )

        key_indices = [self._obj.columns.get_loc(c) for c in self._keys]

        n_key_cols = 0 if self._as_index else len(self._keys)
        empty_data = _cast_groupby_agg_columns2(normalized_func, empty_data, n_key_cols)

        exprs = [
            LazyPlan(
                "AggregateExpression",
                empty_data.iloc[:, i],  # needs to be the output type
                self._obj._plan,
                func_,
                [self._obj.columns.get_loc(col)],
                self._dropna,
            )
            for i, (func_, col) in enumerate(normalized_func)
        ]

        plan = LazyPlan(
            "LogicalAggregate",
            empty_data,
            self._obj._plan,
            key_indices,
            exprs,
        )

        # Add the data column then the keys since they become Index columns in output.
        # DuckDB generates keys first in output so we need to reverse the order.
        if self._as_index:
            col_indices = list(
                range(len(self._keys), len(self._keys) + len(normalized_func))
            )
            col_indices += list(range(len(self._keys)))

            exprs = make_col_ref_exprs(col_indices, plan)
            plan = LazyPlan(
                "LogicalProjection",
                empty_data,
                plan,
                exprs,
            )

        return wrap_plan(plan)

    agg = aggregate

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


def _groupby_agg_plan(
    grouped: SeriesGroupBy | DataFrameGroupBy, func: str | dict[str, str]
) -> BodoSeries | BodoDataFrame:
    """Compute groupby.func() on the Series or DataFrame GroupBy object."""
    from bodo.pandas.base import _empty_like

    zero_size_df = _empty_like(grouped._obj)
    empty_data_pandas = zero_size_df.groupby(grouped._keys, as_index=grouped._as_index)[
        grouped._selection[0]
        if isinstance(grouped, SeriesGroupBy)
        else grouped._selection
    ].agg(func)

    empty_data = _cast_groupby_agg_columns(func, empty_data_pandas, grouped._selection)

    key_indices = [grouped._obj.columns.get_loc(c) for c in grouped._keys]

    exprs = [
        LazyPlan(
            "AggregateExpression",
            zero_size_df[c],
            grouped._obj._plan,
            func,
            [grouped._obj.columns.get_loc(c)],
            grouped._dropna,
        )
        for c in grouped._selection
        if c not in grouped._keys
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
        col_indices = list(
            range(len(grouped._keys), len(grouped._keys) + len(grouped._selection))
        )
        col_indices += list(range(len(grouped._keys)))

        exprs = make_col_ref_exprs(col_indices, plan)
        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            plan,
            exprs,
        )

    return wrap_plan(plan)


def _get_agg_output_type(func: str, pa_type: pa.DataType, col_name: str) -> pa.DataType:
    """Gets the output type of an aggregation or raise ValueError for
    unsupported pa_type/func combinations. Should closely match
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
        else:
            raise ValueError(
                f"GroupBy.sum(): Unsupported dtype in column '{col_name}': {pa_type}."
            )
        return new_type
    elif func == "mean":
        if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
            new_type = pa.float64()
        else:
            raise ValueError(
                f"GroupBy.mean(): Unsupported dtype in column '{col_name}': {pa_type}."
            )
        return new_type
    elif func == "count":
        return pa.int64()

    raise BodoLibNotImplementedException("Unsupported aggregate function: ", func)


def _cast_groupby_agg_columns(
    func: str, data: pd.Series | pd.DataFrame, value_cols: list[str]
) -> pd.Series | pd.DataFrame:
    """Upcast value columns for aggregation functions and check output dtypes
    are valid.
    """
    from bodo.pandas.utils import _empty_pd_array

    if isinstance(data, pd.Series):
        pa_types = [data.dtype.pyarrow_dtype]
    else:
        pa_types = [
            data.iloc[:, i].dtype.pyarrow_dtype
            for i, col in enumerate(data.columns)
            if col in value_cols
        ]

    new_types: dict[str, pa.DataType] = {}

    for col, pa_type in zip(value_cols, pa_types):
        new_types[col] = _get_agg_output_type(func, pa_type, col)

    if isinstance(data, pd.Series):
        casted_col = _empty_pd_array(new_types[value_cols[0]])
        return pd.Series(casted_col, index=data.index, name=data.name)
    else:
        for col, new_type in new_types.items():
            data[col] = _empty_pd_array(new_type)
        return data


def _cast_groupby_agg_columns2(
    func: tuple[str, str] | str, data: pd.Series | pd.DataFrame, n_key_cols: int
) -> pd.Series | pd.DataFrame:
    """
    Args:
        func (tuple[str, str]): _description_
        data (pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """

    if isinstance(data, pd.Series):
        # only one output column
        assert isinstance(func, str)
        new_type = _get_agg_output_type(func, data.dtype.pyarrow_dtype, data.name)
        data.astype(pd.ArrowDtype(new_type))

    for i, func_ in enumerate(func):
        col = data.columns[i + n_key_cols]
        out_type = data[col].dtype.pyarrow_dtype
        new_type = _get_agg_output_type(func_[0], out_type, col)
        data[col] = data[col].astype(pd.ArrowDtype(new_type))

    return data
