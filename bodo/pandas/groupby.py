"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

import ctypes
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
    BODO_NONE_DUMMY,
    BodoLibNotImplementedException,
    _empty_pd_array,
    check_args_fallback,
    convert_to_pandas_types,
    fallback_warn,
    wrap_plan,
)

if pt.TYPE_CHECKING:
    from bodo.pandas import BodoDataFrame, BodoSeries


BUILTIN_AGG_FUNCS = {
    "sum",
    "min",
    "max",
    "idxmin",
    "idxmax",
    "median",
    "mean",
    "std",
    "var",
    "skew",
    "count",
    "size",
    "nunique",
    "first",
    "last",
}


class GroupbyAggFunc:
    """Stores data for computing aggfuncs."""

    in_col: str | None  # Input column name, or None if func is size.
    func: pt.Callable | str  # The actual function
    func_name: str  # The function name as it will appear in the output.

    def __init__(self, in_col, func):
        self.in_col = in_col
        self.func = func
        self.func_name = _get_aggfunc_str(func)

    @property
    def is_custom_aggfunc(self):
        """False if self is a builtin agg func e.g. sum, mean,..."""
        # TODO: support custom implementations of builtin agg funcs
        return callable(self.func) and self.func_name not in BUILTIN_AGG_FUNCS


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
                    "implemented in Bodo Dataframes yet. "
                    "Falling back to Pandas (may be slow or run out of memory)."
                )
                gb = pd.DataFrame(self._obj).groupby(
                    self._keys, as_index=self._as_index, dropna=self._dropna
                )
                if self._selection is not None:
                    gb = gb[self._selection]
                fallback_warn(msg)
                return object.__getattribute__(gb, name)

            if name in self._obj:
                return self.__getitem__(name)

            raise AttributeError(e)

    @check_args_fallback(supported=["func"])
    def apply(self, func, *args, include_groups=False, **kwargs):
        """Apply a function group-wise and combine results together."""
        return _groupby_apply_plan(self, func, *args, **kwargs)

    @check_args_fallback(supported="func")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        return _groupby_agg_plan(self, func, *args, **kwargs)

    agg = aggregate

    def _normalize_agg_func(
        self, func, selection: list[str], kwargs: dict
    ) -> list[GroupbyAggFunc]:
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        # list of (input column name, function) pairs
        normalized_func: list[GroupbyAggFunc] = []

        if func is None and kwargs:
            # Handle cases like agg(my_sum=("A", "sum")) -> creates column my_sum
            # that sums column A.
            normalized_func = [
                GroupbyAggFunc(named_agg.column, named_agg.aggfunc)
                for named_agg in kwargs.values()
            ]
        elif is_dict_like(func):
            # Handle cases like {"A": "sum"} -> creates sum column over column A
            normalized_func = [
                GroupbyAggFunc(col, func_) for col, func_ in func.items()
            ]
        elif is_list_like(func):
            # Handle cases like ["sum", "count"] -> creates a sum and count column
            # for each input column (column names are a multi-index) i.e.:
            # ("A", "sum"), ("A", "count"), ("B", "sum), ("B", "count")
            normalized_func = [
                GroupbyAggFunc(col, func_) for col in selection for func_ in func
            ]
        else:
            # Size is a special case that only produces 1 column, since it doesn't
            # depend on input column given.
            if func == "size":
                normalized_func = [GroupbyAggFunc(None, func)]
            else:
                normalized_func = [GroupbyAggFunc(col, func) for col in selection]

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

    @check_args_fallback(supported="none")
    def first(self):
        """
        Get the first entry for each group.
        """
        return _groupby_agg_plan(self, "first")

    @check_args_fallback(supported="none")
    def last(self):
        """
        Get the last entry for each group.
        """
        return _groupby_agg_plan(self, "last")


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
                "implemented in Bodo DataFrames yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            fallback_warn(msg)
            gb = pd.DataFrame(self._obj).groupby(self._keys)[self._selection[0]]
            return object.__getattribute__(gb, name)

    @check_args_fallback(supported=["func"])
    def apply(self, func, *args, include_groups=False, **kwargs):
        """Apply a function group-wise and combine results together."""
        return _groupby_apply_plan(self, func, *args, **kwargs)

    @check_args_fallback(supported="func")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        return _groupby_agg_plan(self, func, *args, **kwargs)

    agg = aggregate

    def _normalize_agg_func(self, func, selection: list[str], kwargs):
        """
        Convert func and kwargs into a list of (column, function) tuples.
        """
        col = selection[0]

        # list of (input column name, function) pairs
        normalized_func: list[tuple[str, str]] = []
        if func is None and kwargs:
            # Handle case agg(A="mean") -> create mean column "A"
            normalized_func = [GroupbyAggFunc(col, func_) for func_ in kwargs.values()]
        elif is_dict_like(func):
            # (Deprecated) handle cases like {"A": "mean"} -> create mean column "A"
            normalized_func = [GroupbyAggFunc(col, func_) for func_ in func.values()]
        elif is_list_like(func):
            normalized_func = [GroupbyAggFunc(col, func_) for func_ in func]
        else:
            normalized_func = [GroupbyAggFunc(col, func)]

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


def _groupby_apply_plan(
    grouped: SeriesGroupBy | DataFrameGroupBy, func, *args, **kwargs
) -> BodoSeries | BodoDataFrame:
    """Implementation of SeriesGroupby/DataFrameGroupby.apply."""
    from bodo.pandas.base import _empty_like

    # Import compiler
    import bodo.decorators  # isort:skip # noqa

    bodo.spawn.utils.import_compiler_on_workers()

    if not callable(func):
        raise BodoLibNotImplementedException(
            "Groupby.apply() only supports callable values for func."
        )

    # TODO: support passing kwargs and args
    if args or kwargs:
        raise BodoLibNotImplementedException(
            "Groupby.apply(): passing positional or keyword arguments "
            "to func is not supported yet."
        )

    # NOTE: assumes no key columns are being aggregated e.g:
    # df1.groupby("C", as_index=False)[["C"]].apply(...)
    if set(grouped._keys) & set(grouped.selection_for_plan):
        raise BodoLibNotImplementedException(
            "Applying a function on key columns not supported yet."
        )

    zero_size_self = _empty_like(grouped._obj)

    selected_self = (
        zero_size_self[grouped.selection_for_plan]
        if isinstance(grouped, DataFrameGroupBy)
        else zero_size_self[grouped.selection_for_plan[0]]
    )

    out_type = _get_scalar_udf_out_type(func, selected_self)

    if isinstance(out_type, (bodo.types.DataFrameType, bodo.types.SeriesType)):
        raise BodoLibNotImplementedException(
            "Groupby.apply(): functions returning Series or DataFrame not implemented yet."
        )

    out_arrow_type = _numba_type_to_pyarrow_type(out_type)
    if out_arrow_type is None:
        raise BodoLibNotImplementedException(
            "Groupby.apply(): Unsupported UDF output type:", out_type
        )

    selected_cols = [
        grouped._obj.columns.get_loc(col) for col in grouped.selection_for_plan
    ]
    empty_out_col = pd.Series(_empty_pd_array(out_arrow_type))

    exprs = [
        AggregateExpression(
            empty_out_col,
            grouped._obj._plan,
            "udf",
            _get_cfunc_wrapper(func, selected_self, empty_out_col),
            selected_cols,
            grouped._dropna,
        )
    ]

    empty_data = zero_size_self[grouped._keys]

    out_name = (
        BODO_NONE_DUMMY if isinstance(grouped, DataFrameGroupBy) else selected_self.name
    )
    empty_data[out_name] = empty_out_col

    if grouped._as_index:
        # Convert output to series
        empty_data = empty_data.set_index(grouped._keys)
        empty_data = empty_data[out_name]
    elif out_name == BODO_NONE_DUMMY:
        # TODO: support as_index=False case output column is None
        empty_data = empty_data.rename(columns={out_name: "None"})

    return _make_logical_agg_plan(grouped, exprs, empty_data)


def _groupby_agg_plan(
    grouped: SeriesGroupBy | DataFrameGroupBy, func, *args, **kwargs
) -> BodoSeries | BodoDataFrame:
    """Compute groupby.func() on the Series or DataFrame GroupBy object."""
    from bodo.pandas.base import _empty_like

    grouped_selection = grouped.selection_for_plan

    zero_size_df = _empty_like(grouped._obj)

    # Convert to Pandas types to avoid gaps in Arrow conversion
    zero_size_df_pandas = convert_to_pandas_types(zero_size_df)
    empty_data_pandas = zero_size_df_pandas.groupby(
        grouped._keys, as_index=grouped._as_index
    )[
        grouped_selection[0]
        if isinstance(grouped, SeriesGroupBy)
        else grouped_selection
    ].agg(func, *args, **kwargs)

    normalized_func = grouped._normalize_agg_func(func, grouped_selection, kwargs)

    # NOTE: assumes no key columns are being aggregated e.g:
    # df1.groupby("C", as_index=False)[["C"]].agg("sum")
    if set(grouped._keys) & set(grouped_selection):
        raise BodoLibNotImplementedException(
            "GroupBy.agg(): Aggregation on key columns not supported yet."
        )

    n_key_cols = 0 if grouped._as_index else len(grouped._keys)
    empty_data = _cast_groupby_agg_columns(
        normalized_func, zero_size_df, empty_data_pandas, n_key_cols
    )

    out_types = empty_data
    if isinstance(empty_data, pd.DataFrame) and not grouped._as_index:
        out_types = empty_data.iloc[:, n_key_cols:]

    exprs = []
    for i, func in enumerate(normalized_func):
        out_type = (
            out_types.iloc[:, i] if isinstance(out_types, pd.DataFrame) else out_types
        )
        func_name = f"udf_{i}" if func.is_custom_aggfunc else func.func_name
        cfunc_wrapper = (
            _get_cfunc_wrapper(func.func, zero_size_df[func.in_col], out_type)
            if func.is_custom_aggfunc
            else None
        )
        col_idx = (
            grouped._obj.columns.get_loc(func.in_col) if func.func_name != "size" else 0
        )
        exprs.append(
            AggregateExpression(
                out_type,
                grouped._obj._plan,
                func_name,
                cfunc_wrapper,
                [col_idx],
                grouped._dropna,
            )
        )

    return _make_logical_agg_plan(grouped, exprs, empty_data)


def _make_logical_agg_plan(
    grouped: DataFrameGroupBy | SeriesGroupBy,
    exprs: list[AggregateExpression],
    empty_out_data: pd.DataFrame | pd.Series,
) -> BodoDataFrame | BodoSeries:
    """Wrap exprs in a LogicalAggregate lazy plan and do additional column reshuffling
    if necessary.
    """
    key_indices = [grouped._obj.columns.get_loc(c) for c in grouped._keys]

    plan = LogicalAggregate(
        empty_out_data,
        grouped._obj._plan,
        key_indices,
        exprs,
    )

    # Add the data column then the keys since they become Index columns in output.
    # DuckDB generates keys first in output so we need to reverse the order.
    if grouped._as_index:
        col_indices = list(range(len(grouped._keys), len(grouped._keys) + len(exprs)))
        col_indices += list(range(len(grouped._keys)))

        exprs = make_col_ref_exprs(col_indices, plan)
        plan = LogicalProjection(
            empty_out_data,
            plan,
            exprs,
        )

    return wrap_plan(plan)


def _get_cfunc_wrapper(
    func: pt.Callable, empty_data: pd.DataFrame | pd.Series, out_type: pd.Series
) -> pt.Callable[[], int]:
    """Create a wrapper around compiling a cfunc, which computes a UDF result for a
    single group. Called once on each worker.

    Args:
        func (pt.Callable): The UDF
        empty_data (pd.DataFrame | pd.Series): Input column or DataFrame. Will be a
          DataFrame in the case of DataFrameGroupby.apply().
        out_type (pd.Series): Empty Series of the same type as the output column.

    Returns:
        Callable: A function that takes no arguments and compiles the cfunc and returns
          the address.
    """
    import numba
    import numpy as np

    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.decorators import _cfunc
    from bodo.hiframes.table import TableType
    from bodo.libs.array import (
        array_info_type,
        array_to_info,
        table_type,
    )
    from bodo.pandas.utils_jit import cpp_table_to_df_jit, cpp_table_to_series_jit
    from bodo.utils.conversion import coerce_scalar_to_array, coerce_to_array
    from bodo.utils.typing import BodoWarning

    jitted_func = bodo.jit(cache=False, spawn=False, distributed=False)(func)

    # Ignore warning "Empty object array passed to Bodo"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BodoWarning)
        in_type = bodo.typeof(empty_data)
        out_col_type = bodo.typeof(out_type).data

    if isinstance(in_type, bodo.types.DataFrameType):
        # Input table does not have an index
        index_type = bodo.typeof(pd.RangeIndex(0))
        py_table_type = TableType(in_type.data)

        cols = tuple(empty_data.columns)
        out_cols_arr = np.array(range(len(cols)), dtype=np.int64)
        column_names = bodo.utils.typing.ColNamesMetaType(cols)

        @numba.njit
        def cpp_table_to_py_impl(in_cpp_table):  # pragma: no cover
            return cpp_table_to_df_jit(
                in_cpp_table, out_cols_arr, column_names, py_table_type, index_type
            )

    else:
        assert isinstance(in_type, bodo.types.SeriesType), (
            "Expected in_type to be either SeriesType or DataFrameType."
        )

        in_col_type = in_type.data

        @numba.njit
        def cpp_table_to_py_impl(in_cpp_table):  # pragma: no cover
            return cpp_table_to_series_jit(in_cpp_table, in_col_type)

    if isinstance(out_col_type, bodo.types.ArrayItemArrayType):

        @numba.njit
        def coerce_to_array_impl(scalar):  # pragma: no cover
            # ArrayItemArrayType expects an Array Dtype, so we first have to
            # coerce the type to Array before converting the Array to a nested array.
            out_arr = coerce_to_array(scalar, use_nullable_array=True)
            out_arr = coerce_scalar_to_array(
                out_arr, 1, out_col_type, dict_encode=False
            )
            return out_arr
    else:

        @numba.njit
        def coerce_to_array_impl(scalar):  # pragma: no cover
            out_arr = coerce_scalar_to_array(scalar, 1, out_col_type, dict_encode=False)
            # coerce_scalar_to_array doesn't respect nullable types, so using coerce_to_array
            # to cast output to nullable if it isn't already.
            out_arr = coerce_to_array(out_arr, use_nullable_array=True)
            return out_arr

    def wrapper():
        def apply_func_impl(in_cpp_table):  # pragma: no cover
            py_in = cpp_table_to_py_impl(in_cpp_table)
            result = jitted_func(py_in)
            out_arr = coerce_to_array_impl(result)
            return array_to_info(out_arr)

        c_sig = array_info_type(table_type)
        cfunc = _cfunc(c_sig, cache=False)(apply_func_impl)
        return ctypes.c_void_p(cfunc.address).value

    return wrapper


def _numba_type_to_pyarrow_type(typ):
    """Convert the given type to the corresponding Pyarrow type or
    return None.

    Similar to io/helpers.py::_numba_type_to_pyarrow_type.
    """
    from numba import types

    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
    from bodo.libs.binary_arr_ext import bytes_type
    from bodo.utils.typing import get_array_getitem_scalar_type
    from bodo.utils.utils import is_array_typ

    numba_to_arrow_map = {
        types.bool_: pa.bool_(),
        # Signed Int Types
        types.int8: pa.int8(),
        types.int16: pa.int16(),
        types.int32: pa.int32(),
        types.int64: pa.int64(),
        # Unsigned Int Types
        types.uint8: pa.uint8(),
        types.uint16: pa.uint16(),
        types.uint32: pa.uint32(),
        types.uint64: pa.uint64(),
        # Float Types
        types.float16: pa.float16(),
        types.float32: pa.float32(),
        types.float64: pa.float64(),
        # Date and Time
        types.NPDatetime("ns"): pa.date64(),
        pd_timedelta_type: pa.duration("ns"),
        # String / Binary
        types.unicode_type: pa.large_string(),
        bytes_type: pa.large_binary(),
    }

    if isinstance(typ, types.Literal):
        return _numba_type_to_pyarrow_type(typ.literal_type)

    elif isinstance(typ, types.Optional):
        return _numba_type_to_pyarrow_type(typ.type)

    elif isinstance(typ, types.List):
        inner_type = _numba_type_to_pyarrow_type(typ.dtype)
        return pa.large_list(inner_type)

    elif is_array_typ(typ, include_index_series=False):
        inner_type = _numba_type_to_pyarrow_type(get_array_getitem_scalar_type(typ))
        return pa.large_list(inner_type)

    elif isinstance(typ, bodo.types.PandasTimestampType):
        return pa.timestamp("ns", tz=typ.tz)

    elif isinstance(typ, bodo.types.StructType):
        inner_types = [_numba_type_to_pyarrow_type(typ_) for typ_ in typ.data]
        fields = [pa.field(name, typ_) for name, typ_ in zip(typ.names, inner_types)]
        return pa.struct(fields)

    # TODO: expand to more cases.

    return numba_to_arrow_map.get(typ, None)


def _get_scalar_udf_out_type(func: pt.Callable, empty_input: pd.DataFrame | pd.Series):
    """Use compiler utilities to determine the output type of func given it's input types."""
    import numba

    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from numba.core.target_extension import dispatcher_registry

    from bodo.utils.transform import get_const_func_output_type

    jitted_func = bodo.jit(cache=False, spawn=False, distributed=False)(func)

    disp = dispatcher_registry[numba.core.target_extension.CPU]
    typing_ctx = disp.targetdescr.typing_context
    # Refresh typing_ctx in case any new declarations were added.
    typing_ctx.refresh()

    target_ctx = (numba.core.registry.cpu_target.target_context,)

    in_type = bodo.typeof(empty_input)

    try:
        return_type = get_const_func_output_type(
            jitted_func, (in_type,), (), typing_ctx, target_ctx
        )
    except Exception as e:
        raise BodoLibNotImplementedException(
            f"An error occured while compiling user defined function '{func.__name__}': {e}"
        )

    return return_type


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


def _get_agg_output_type(
    func: GroupbyAggFunc, pa_type: pa.DataType, col_name: str
) -> pa.DataType:
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
    func_name = func.func_name

    # TODO: Enable more fallbacks where the operation is supported in Pandas and not in Bodo
    if func_name in ("sum",):
        if pa.types.is_signed_integer(pa_type) or pa.types.is_boolean(pa_type):
            new_type = pa.int64()
        elif pa.types.is_unsigned_integer(pa_type):
            new_type = pa.uint64()
        elif pa.types.is_duration(pa_type):
            new_type = pa_type
        elif pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            new_type = pa_type
        elif pa.types.is_decimal(pa_type):
            # TODO: Decimal sum
            fallback = True
    elif func_name in ("mean", "std", "var", "skew"):
        if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
            new_type = pa.float64()
        elif pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type):
            # TODO Support bool/decimal columns
            fallback = True
    elif func_name in ("count", "size", "nunique"):
        new_type = pa.int64()
    elif func_name in ("min", "max"):
        if (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_boolean(pa_type)
            or pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
            or pa.types.is_duration(pa_type)
            or pa.types.is_date(pa_type)
            or pa.types.is_timestamp(pa_type)
        ):
            new_type = pa_type
        elif pa.types.is_decimal(pa_type):
            fallback = True
    elif func_name == "median":
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
    elif func_name in ("first", "last"):
        new_type = pa_type
    elif callable(func.func):
        # Import compiler
        import bodo.decorators  # isort:skip # noqa
        from bodo.utils.utils import is_array_typ

        # UDF case
        empty_in_col = pd.Series(_empty_pd_array(pa_type))
        out_numba_type = _get_scalar_udf_out_type(func.func, empty_in_col)

        # Matches Pandas error without the need to fall back.
        if (
            isinstance(
                out_numba_type, (bodo.types.SeriesType, bodo.types.DataFrameType)
            )
            or is_array_typ(out_numba_type)
        ) and not (pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type)):
            raise ValueError(
                "Groupby.agg(): User defined function must produce aggregated value."
            )

        # If result is None (could not convert output type to pyarrow),
        # fall back to Pandas.
        fallback = True
        new_type = _numba_type_to_pyarrow_type(out_numba_type)
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
    func: list[GroupbyAggFunc],
    in_data: pd.DataFrame,
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
        in_data : An empty DataFrame with the same shape as the input to the
            aggregation.
        n_key_cols : Number of grouping keys in the output.

    Returns:
        pd.Series | pd.DataFrame: A DataFrame or Series with the dtypes casted depending
            on the aggregate functions.
    """

    if in_data.columns.has_duplicates:
        raise BodoLibNotImplementedException(
            "GroupBy.agg(): duplicate column names in input not supported yet."
        )

    # Checks for cases like bdf.groupby("C")[["A", "A"]].agg(["sum"]).
    if isinstance(out_data, pd.DataFrame) and out_data.columns.has_duplicates:
        raise BodoLibNotImplementedException(
            "GroupBy.agg(): duplicate column names in output not supported yet."
        )

    for i, func_ in enumerate(func):
        if func_.func_name == "size":
            new_type = pa.int64()
        else:
            in_col = in_data[func_.in_col]
            new_type = _get_agg_output_type(
                func_, in_col.dtype.pyarrow_dtype, in_col.name
            )

        if isinstance(out_data, pd.Series):
            return pd.Series(
                [],
                dtype=pd.ArrowDtype(new_type),
                name=out_data.name,
                index=out_data.index,
            )

        out_col_name = out_data.columns[i + n_key_cols]
        out_data[out_col_name] = pd.Series([], dtype=pd.ArrowDtype(new_type))

    return out_data
