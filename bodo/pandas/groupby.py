"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

import typing as pt
import warnings
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas.core.dtypes.inference import is_dict_like, is_list_like

from bodo.pandas.plan import (
    AggregateExpression,
    LogicalAggregate,
    LogicalProjection,
    make_col_ref_exprs,
)
from bodo.pandas.utils import (
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    check_args_fallback,
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
        self._selection = selection

    @property
    def selection_for_plan(self):
        return (
            self._selection
            if self._selection is not None
            else list(filter(lambda col: col not in self._keys, self._obj.columns))
        )

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        """
        Return a DataFrameGroupBy or SeriesGroupBy for the selected data columns.
        """
        if isinstance(key, str):
            if key not in self._obj:
                raise KeyError(f"Column not found: {key}")
            return SeriesGroupBy(
                self._obj, self._keys, [key], self._as_index, self._dropna
            )
        elif isinstance(key, list) and all(isinstance(key_, str) for key_ in key):
            invalid_keys = []
            for k in key:
                if k not in self._obj:
                    invalid_keys.append(f"'{k}'")
            if invalid_keys:
                raise KeyError(f"Column not found: {', '.join(invalid_keys)}")
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
        except AttributeError as e:
            if hasattr(pd.core.groupby.generic.DataFrameGroupBy, name):
                msg = (
                    f"DataFrameGroupBy.{name} is not "
                    "implemented in Bodo dataframe library yet. "
                    "Falling back to Pandas (may be slow or run out of memory)."
                )
                gb = pd.DataFrame(self._obj).groupby(
                    self._keys, as_index=self._as_index, dropna=self._dropna
                )
                if self._selection is not None:
                    gb = gb[self._selection]
                warnings.warn(BodoLibFallbackWarning(msg))
                return object.__getattribute__(gb, name)

            if name in self._obj:
                return self.__getitem__(name)

            raise AttributeError(e)

    @check_args_fallback(supported="func")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        return _groupby_agg_plan(self, func, *args, **kwargs)

    agg = aggregate

    def _normalize_agg_func(
        self, func, selection, kwargs: dict
    ) -> list[tuple[str, str]]:
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        # list of (input column name, function) pairs
        normalized_func: list[tuple[str, str]] = []

        if func is None and kwargs:
            # Handle cases like agg(my_sum=("A", "sum")) -> creates column my_sum
            # that sums column A.
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for col, func_ in kwargs.values()
            ]
        elif is_dict_like(func):
            # Handle cases like {"A": "sum"} -> creates sum column over column A
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for col, func_ in func.items()
            ]
        elif is_list_like(func):
            # Handle cases like ["sum", "count"] -> creates a sum and count column
            # for each input column (column names are a multi-index) i.e.:
            # ("A", "sum"), ("A", "count"), ("B", "sum), ("B", "count")
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for col in selection for func_ in func
            ]
        else:
            func = _get_aggfunc_str(func)
            # Size is a special case that only produces 1 column, since it doesn't
            # depend on input column given.
            if func == "size":
                # Getting the size of each groups without any input column.
                # e.g. df.groupby("B")[[]].size()
                if len(selection) < 1:
                    raise BodoLibNotImplementedException(
                        "GroupBy.size(): Aggregating without selected columns not supported yet."
                    )
                normalized_func = [(selection[0], "size")]
            else:
                normalized_func = [(col, func) for col in selection]

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
        Compute the mean of each group.
        """
        return _groupby_agg_plan(self, "mean")

    @check_args_fallback(supported="none")
    def count(self):
        """
        Compute the count of each group.
        """
        return _groupby_agg_plan(self, "count")

    @check_args_fallback(supported="none")
    def min(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        """
        Compute the min of each group.
        """
        return _groupby_agg_plan(self, "min")

    @check_args_fallback(supported="none")
    def max(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        """
        Compute the max of each group.
        """
        return _groupby_agg_plan(self, "max")

    @check_args_fallback(supported="none")
    def median(self, numeric_only=False):
        """
        Compute the median of each group.
        """
        return _groupby_agg_plan(self, "median")

    @check_args_fallback(supported="none")
    def nunique(self, dropna=True):
        """
        Compute the nunique of each group.
        """
        return _groupby_agg_plan(self, "nunique")

    @check_args_fallback(supported="none")
    def size(self):
        """
        Compute the size of each group (including missing values).
        """
        return _groupby_agg_plan(self, "size")

    @check_args_fallback(supported="none")
    def skew(self, axis=lib.no_default, skipna=True, numeric_only=False, **kwargs):
        """
        Compute the skew of each group.
        """
        return _groupby_agg_plan(self, "skew")

    @check_args_fallback(supported="none")
    def std(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        """
        Compute the std of each group.
        """
        return _groupby_agg_plan(self, "std")

    @check_args_fallback(supported="none")
    def var(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        """
        Compute the var of each group.
        """
        return _groupby_agg_plan(self, "var")


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

    @property
    def selection_for_plan(self):
        return (
            self._selection
            if self._selection is not None
            else list(filter(lambda col: col not in self._keys, self._obj.columns))
        )  # pragma: no cover

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

    def _normalize_agg_func(self, func, selection, kwargs):
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        col = selection[0]

        # list of (input column name, function) pairs
        normalized_func: list[tuple[str, str]] = []
        if func is None and kwargs:
            # Handle case agg(A="mean") -> create mean column "A"
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for func_ in kwargs.values()
            ]
        elif is_dict_like(func):
            # (Deprecated) handle cases like {"A": "mean"} -> create mean column "A"
            normalized_func = [
                (col, _get_aggfunc_str(func_)) for func_ in func.values()
            ]
        elif is_list_like(func):
            normalized_func = [(col, _get_aggfunc_str(func_)) for func_ in func]
        else:
            normalized_func = [(col, _get_aggfunc_str(func))]

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
        Compute the mean of each group.
        """
        return _groupby_agg_plan(self, "mean")

    @check_args_fallback(supported="none")
    def count(self):
        """
        Compute the count of each group.
        """
        return _groupby_agg_plan(self, "count")

    @check_args_fallback(supported="none")
    def min(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        """
        Compute the min of each group.
        """
        return _groupby_agg_plan(self, "min")

    @check_args_fallback(supported="none")
    def max(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        """
        Compute the max of each group.
        """
        return _groupby_agg_plan(self, "max")

    @check_args_fallback(supported="none")
    def median(self, numeric_only=False):
        """
        Compute the median of each group.
        """
        return _groupby_agg_plan(self, "median")

    @check_args_fallback(supported="none")
    def nunique(self, dropna=True):
        """
        Compute the nunique of each group.
        """
        return _groupby_agg_plan(self, "nunique")

    @check_args_fallback(supported="none")
    def size(self):
        """
        Compute the size of each group (including missing values).
        """
        return _groupby_agg_plan(self, "size")

    @check_args_fallback(supported="none")
    def skew(self, axis=lib.no_default, skipna=True, numeric_only=False, **kwargs):
        """
        Compute the skew of each group.
        """
        return _groupby_agg_plan(self, "skew")

    @check_args_fallback(supported="none")
    def std(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        """
        Compute the std of each group.
        """
        return _groupby_agg_plan(self, "std")

    @check_args_fallback(supported="none")
    def var(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        """
        Compute the var of each group.
        """
        return _groupby_agg_plan(self, "var")


def _groupby_agg_plan(
    grouped: SeriesGroupBy | DataFrameGroupBy, func, *args, **kwargs
) -> BodoSeries | BodoDataFrame:
    """Compute groupby.func() on the Series or DataFrame GroupBy object."""
    from bodo.pandas.base import _empty_like

    grouped_selection = grouped.selection_for_plan

    zero_size_df = _empty_like(grouped._obj)
    empty_data_pandas = zero_size_df.groupby(grouped._keys, as_index=grouped._as_index)[
        grouped_selection[0]
        if isinstance(grouped, SeriesGroupBy)
        else grouped_selection
    ].agg(func, *args, **kwargs)

    func = grouped._normalize_agg_func(func, grouped_selection, kwargs)

    # NOTE: assumes no key columns are being aggregated e.g:
    # df1.groupby("C", as_index=False)[["C"]].agg("sum")
    if set(grouped._keys) & set(grouped_selection):
        raise BodoLibNotImplementedException(
            "GroupBy.agg(): Aggregation on key columns not supported yet."
        )

    n_key_cols = 0 if grouped._as_index else len(grouped._keys)
    empty_data = _cast_groupby_agg_columns(
        func, zero_size_df, empty_data_pandas, n_key_cols
    )

    key_indices = [grouped._obj.columns.get_loc(c) for c in grouped._keys]

    exprs = [
        AggregateExpression(
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

    plan = LogicalAggregate(
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
        plan = LogicalProjection(
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
        f"GroupBy.agg(): expected func to be callable or string, got: {type(func)}."
    )


def _get_agg_output_type(func: str, pa_type: pa.DataType, col_name: str) -> pa.DataType:
    """Cast the input type to the correct output type depending on func or raise if
    the specific combination of func + input type is not supported.

    Args:
        func (str): The function to apply.
        pa_type (pa.DataType): The input type of the function.
        col_name (str): The name of the column in the input.

    Raises:
        BodoLibNotImplementedException: If the operation is not supported in Bodo
            but is supported in Pandas.
        TypeError: If the operation is not supported in Bodo or Pandas (due to gaps
            in Pandas' handling of Arrow Types)

    Returns:
        pa.DataType: The output type from applying func to col_name.
    """
    new_type = None
    fallback = False

    # TODO: Enable more fallbacks where the operation is supported in Pandas and not in Bodo
    if func in ("sum",):
        if pa.types.is_signed_integer(pa_type) or pa.types.is_boolean(pa_type):
            new_type = pa.int64()
        elif pa.types.is_unsigned_integer(pa_type):
            new_type = pa.uint64()
        elif pa.types.is_duration(pa_type):
            new_type = pa_type
        elif pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_string(pa_type):
            new_type = pa_type
        elif pa.types.is_decimal(pa_type):
            # TODO: Decimal sum
            fallback = True
    elif func in ("mean", "std", "var", "skew"):
        if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type):
            # TODO Support bool/decimal columns
            fallback = True
    elif func in ("count", "size", "nunique"):
        new_type = pa.int64()
    elif func in ("min", "max"):
        if (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_boolean(pa_type)
            or pa.types.is_string(pa_type)
            or pa.types.is_duration(pa_type)
            or pa.types.is_date(pa_type)
            or pa.types.is_timestamp(pa_type)
        ):
            new_type = pa_type
        elif pa.types.is_decimal(pa_type):
            fallback = True
    elif func == "median":
        if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
            new_type = pa_type
        elif (
            pa.types.is_boolean(pa_type)
            or pa.types.is_decimal(pa_type)
            or pa.types.is_timestamp(pa_type)
            or pa.types.is_duration(pa_type)
        ):
            # TODO: bool/decimal median
            fallback = True
    else:
        raise BodoLibNotImplementedException("Unsupported aggregate function: ", func)

    if new_type is not None:
        return new_type
    elif fallback:
        # For cases where Pandas supports the func+type combo but Bodo does not.
        raise BodoLibNotImplementedException(
            f"GroupBy.{func}() on input column '{col_name}' with type: {pa_type} not supported yet."
        )
    else:
        # For gaps in Pandas where a specific function is not implemented for arrow or was somehow
        # falling back to Pandas would also fail, so failing earlier is better.
        raise TypeError(
            f"GroupBy.{func}(): Unsupported dtype in column '{col_name}': {pa_type}."
        )


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

        # Checks for cases like bdf.groupby("C")[["A", "A"]].agg(["sum"]).
        if not isinstance(out_data[out_col_name], pd.Series):
            raise BodoLibNotImplementedException(
                f"GroupBy.agg(): detected duplicate output column name in output columns: '{out_col_name}'"
            )

        in_col = in_data[in_col_name]
        # Should've been handled in the check above, but just to be safe.
        if not isinstance(in_col, pd.Series):
            raise BodoLibNotImplementedException(
                f"GroupBy.agg(): detected duplicate column name in input column: '{in_col_name}'"
            )

        new_type = _get_agg_output_type(func_, in_col.dtype.pyarrow_dtype, in_col_name)
        out_data[out_col_name] = out_data[out_col_name].astype(pd.ArrowDtype(new_type))

    return out_data
