from __future__ import annotations

import inspect
import itertools
import numbers
import typing as pt
import warnings
from collections.abc import Callable, Hashable

import numpy
import pandas as pd
import pyarrow as pa
from pandas._typing import (
    Axis,
    SortKind,
    ValueKeyFunc,
)

import bodo
from bodo.ext import plan_optimizer
from bodo.pandas.array_manager import LazySingleArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper, ExecState
from bodo.pandas.managers import LazyMetadataMixin, LazySingleBlockManager
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ColRefExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    LazyPlan,
    LazyPlanDistributedArg,
    LogicalAggregate,
    LogicalFilter,
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    LogicalLimit,
    LogicalOperator,
    LogicalOrder,
    LogicalProjection,
    PythonScalarFuncExpression,
    UnaryOpExpression,
    _get_df_python_func_plan,
    execute_plan,
    get_proj_expr_single,
    get_single_proj_source_if_present,
    is_arith_expr,
    is_col_ref,
    is_scalar_func,
    is_single_colref_projection,
    is_single_projection,
    make_col_ref_exprs,
)
from bodo.pandas.utils import (
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    arrow_to_empty_df,
    check_args_fallback,
    fallback_wrapper,
    get_lazy_single_manager_class,
    get_n_index_arrays,
    get_scalar_udf_result_type,
    wrap_plan,
)
from bodo.utils.typing import BodoError


class BodoSeries(pd.Series, BodoLazyWrapper):
    # We need to store the head_s to avoid data pull when head is called.
    # Since BlockManagers are in Cython it's tricky to override all methods
    # so some methods like head will still trigger data pull if we don't store head_s and
    # use it directly when available.
    _head_s: pd.Series | None = None
    _name: Hashable = None

    def __new__(cls, *args, **kwargs):
        """Support bodo.pandas.Series() constructor by creating a pandas Series
        and then converting it to a BodoSeries.
        """
        # Handle Pandas internal use which creates an empty object and then assigns the
        # manager:
        # https://github.com/pandas-dev/pandas/blob/1da0d022057862f4352113d884648606efd60099/pandas/core/generic.py#L309
        if not args and not kwargs:
            return super().__new__(cls, *args, **kwargs)

        S = pd.Series(*args, **kwargs)
        df = pd.DataFrame({"A": S})
        bodo_S = bodo.pandas.base.from_pandas(df)["A"]
        bodo_S._name = S.name
        return bodo_S

    def __init__(self, *args, **kwargs):
        # No-op since already initialized by __new__
        pass

    @property
    def _plan(self):
        if hasattr(self._mgr, "_plan"):
            if self.is_lazy_plan():
                return self._mgr._plan
            else:
                """We can't create a new LazyPlan each time that _plan is called
                   because filtering checks that the projections that are part of
                   the filter all come from the same source and if you create a
                   new LazyPlan here each time then they will appear as different
                   sources.  We sometimes use a pandas manager which doesn't have
                   _source_plan so we have to do getattr check.
                """
                if getattr(self, "_source_plan", None) is not None:
                    return self._source_plan

                from bodo.pandas.base import _empty_like

                empty_data = _empty_like(self)
                if bodo.dataframe_library_run_parallel:
                    nrows = len(self)
                    self._source_plan = LogicalGetPandasReadParallel(
                        empty_data,
                        nrows,
                        LazyPlanDistributedArg(self),
                    )
                else:
                    self._source_plan = LogicalGetPandasReadSeq(
                        empty_data,
                        self,
                    )

                return self._source_plan

        raise NotImplementedError(
            "Plan not available for this manager, recreate this series with from_pandas"
        )

    def __getattribute__(self, name: str):
        """Custom attribute access that triggers a fallback warning for unsupported attributes."""

        ignore_fallback_attrs = ["dtype", "name", "to_string", "attrs", "flags"]

        cls = object.__getattribute__(self, "__class__")
        base = cls.__mro__[0]

        if (
            name not in base.__dict__
            and name not in ignore_fallback_attrs
            and not name.startswith("_")
        ):
            msg = (
                f"{name} is not implemented in Bodo Dataframe Library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            warnings.warn(BodoLibFallbackWarning(msg))
            return fallback_wrapper(object.__getattribute__(self, name))

        return object.__getattribute__(self, name)

    @check_args_fallback("all")
    def _cmp_method(self, other, op):
        """Called when a BodoSeries is compared with a different entity (other)
        with the given operator "op".
        """
        from bodo.pandas.base import _empty_like

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = _empty_like(other) if isinstance(other, BodoSeries) else other
        # This is effectively a check for a dataframe or series.
        if hasattr(other, "_plan"):
            other = other._plan

        # Compute schema of new series.
        empty_data = zero_size_self._cmp_method(zero_size_other, op)
        assert isinstance(empty_data, pd.Series), "_cmp_method: Series expected"

        # Extract argument expressions
        lhs = get_proj_expr_single(self._plan)
        rhs = get_proj_expr_single(other) if isinstance(other, LazyPlan) else other
        expr = ComparisonOpExpression(
            empty_data,
            lhs,
            rhs,
            op,
        )

        key_indices = [i + 1 for i in range(get_n_index_arrays(empty_data.index))]
        plan_keys = get_single_proj_source_if_present(self._plan)
        key_exprs = tuple(make_col_ref_exprs(key_indices, plan_keys))

        plan = LogicalProjection(
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,) + key_exprs,
        )
        return wrap_plan(plan=plan)

    def _conjunction_binop(self, other, op):
        """Called when a BodoSeries is element-wise boolean combined with a different entity (other)"""
        from bodo.pandas.base import _empty_like

        if not (
            (
                isinstance(other, BodoSeries)
                and isinstance(other.dtype, pd.ArrowDtype)
                and other.dtype.type is bool
            )
            or isinstance(other, bool)
        ):
            raise BodoLibNotImplementedException(
                "'other' should be boolean BodoSeries or a bool. "
                f"Got {type(other).__name__} instead."
            )

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = _empty_like(other) if isinstance(other, BodoSeries) else other
        # This is effectively a check for a dataframe or series.
        if hasattr(other, "_plan"):
            other = other._plan

        # Compute schema of new series.
        empty_data = getattr(zero_size_self, op)(zero_size_other)
        assert isinstance(empty_data, pd.Series), (
            "_conjunction_binop: empty_data is not a Series"
        )

        # Extract argument expressions
        lhs = get_proj_expr_single(self._plan)
        rhs = get_proj_expr_single(other) if isinstance(other, LazyPlan) else other
        expr = ConjunctionOpExpression(
            empty_data,
            lhs,
            rhs,
            op,
        )

        key_indices = [i + 1 for i in range(get_n_index_arrays(empty_data.index))]
        plan_keys = get_single_proj_source_if_present(self._plan)
        key_exprs = tuple(make_col_ref_exprs(key_indices, plan_keys))

        plan = LogicalProjection(
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,) + key_exprs,
        )
        return wrap_plan(plan=plan)

    @check_args_fallback("all")
    def __and__(self, other):
        """Called when a BodoSeries is element-wise and'ed with a different entity (other)"""
        return self._conjunction_binop(other, "__and__")

    @check_args_fallback("all")
    def __or__(self, other):
        """Called when a BodoSeries is element-wise or'ed with a different entity (other)"""
        return self._conjunction_binop(other, "__or__")

    @check_args_fallback("all")
    def __xor__(self, other):
        """Called when a BodoSeries is element-wise xor'ed with a different
        entity (other). xor is not supported in duckdb so convert to
        (A or B) and not (A and B).
        """
        return self.__or__(other).__and__(self.__and__(other).__invert__())

    @check_args_fallback("all")
    def __invert__(self):
        """Called when a BodoSeries is element-wise not'ed with a different entity (other)"""
        from bodo.pandas.base import _empty_like

        # Get empty Pandas objects for self and other with same schema.
        empty_data = _empty_like(self)

        assert isinstance(empty_data, pd.Series), "Series expected"
        source_expr = get_proj_expr_single(self._plan)
        expr = UnaryOpExpression(
            empty_data,
            source_expr,
            "__invert__",
        )

        key_indices = [i + 1 for i in range(get_n_index_arrays(empty_data.index))]
        plan_keys = get_single_proj_source_if_present(self._plan)
        key_exprs = tuple(make_col_ref_exprs(key_indices, plan_keys))

        plan = LogicalProjection(
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,) + key_exprs,
        )
        return wrap_plan(plan=plan)

    def _arith_binop(self, other, op, reverse):
        """Called when a BodoSeries is element-wise arithmetically combined with a different entity (other)"""
        from bodo.pandas.base import _empty_like

        if not (
            (
                isinstance(other, BodoSeries)
                and isinstance(other.dtype, pd.ArrowDtype)
                and pd.api.types.is_numeric_dtype(other.dtype)
            )
            or isinstance(other, numbers.Number)
        ):
            raise BodoLibNotImplementedException(
                "'other' should be numeric BodoSeries or a numeric. "
                f"Got {type(other).__name__} instead."
            )

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = _empty_like(other) if isinstance(other, BodoSeries) else other
        # This is effectively a check for a dataframe or series.
        if hasattr(other, "_plan"):
            other = other._plan

        # Compute schema of new series.
        empty_data = getattr(zero_size_self, op)(zero_size_other)
        assert isinstance(empty_data, pd.Series), (
            "_arith_binop: empty_data is not a Series"
        )

        # Extract argument expressions
        lhs = get_proj_expr_single(self._plan)
        rhs = get_proj_expr_single(other) if isinstance(other, LazyPlan) else other
        if reverse:
            lhs, rhs = rhs, lhs

        expr = ArithOpExpression(empty_data, lhs, rhs, op)

        key_indices = [i + 1 for i in range(get_n_index_arrays(empty_data.index))]
        plan_keys = get_single_proj_source_if_present(self._plan)
        key_exprs = tuple(make_col_ref_exprs(key_indices, plan_keys))

        plan = LogicalProjection(
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,) + key_exprs,
        )
        return wrap_plan(plan=plan)

    @check_args_fallback("all")
    def __add__(self, other):
        return self._arith_binop(other, "__add__", False)

    @check_args_fallback("all")
    def __radd__(self, other):
        return self._arith_binop(other, "__radd__", True)

    @check_args_fallback("all")
    def __sub__(self, other):
        return self._arith_binop(other, "__sub__", False)

    @check_args_fallback("all")
    def __rsub__(self, other):
        return self._arith_binop(other, "__rsub__", True)

    @check_args_fallback("all")
    def __mul__(self, other):
        return self._arith_binop(other, "__mul__", False)

    @check_args_fallback("all")
    def __rmul__(self, other):
        return self._arith_binop(other, "__rmul__", True)

    @check_args_fallback("all")
    def __truediv__(self, other):
        return self._arith_binop(other, "__truediv__", False)

    @check_args_fallback("all")
    def __rtruediv__(self, other):
        return self._arith_binop(other, "__rtruediv__", True)

    @check_args_fallback("all")
    def __floordiv__(self, other):
        return self._arith_binop(other, "__floordiv__", False)

    @check_args_fallback("all")
    def __rfloordiv__(self, other):
        return self._arith_binop(other, "__rfloordiv__", True)

    @check_args_fallback("all")
    def __getitem__(self, key):
        """Called when df[key] is used."""

        from bodo.pandas.base import _empty_like

        # Only selecting columns or filtering with BodoSeries is supported
        if not isinstance(key, BodoSeries):
            raise BodoLibNotImplementedException("only BodoSeries keys are supported")

        zero_size_self = _empty_like(self)

        key_plan = (
            # TODO: error checking for key to be a projection on the same dataframe
            # with a binary operator
            get_proj_expr_single(key._plan)
            if key._plan is not None
            else plan_optimizer.LogicalGetSeriesRead(key._mgr._md_result_id)
        )
        zero_size_key = _empty_like(key)
        zero_size_index = zero_size_key.index
        empty_data = zero_size_self.__getitem__(zero_size_key)
        empty_data_index = empty_data.index
        if isinstance(zero_size_index, pd.RangeIndex) and not isinstance(
            empty_data_index, pd.RangeIndex
        ):
            # Drop the explicit integer Index generated from filtering RangeIndex (TODO: support RangeIndex properly).
            empty_data.reset_index(drop=True, inplace=True)
        return wrap_plan(
            plan=LogicalFilter(empty_data, self._plan, key_plan),
        )

    @staticmethod
    def from_lazy_mgr(
        lazy_mgr: LazySingleArrayManager | LazySingleBlockManager,
        head_s: pd.Series | None,
    ):
        """
        Create a BodoSeries from a lazy manager and possibly a head_s.
        If you want to create a BodoSeries from a pandas manager use _from_mgr
        """
        series = BodoSeries._from_mgr(lazy_mgr, [])
        series._name = head_s._name
        series._head_s = head_s
        return series

    @classmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        plan: LogicalOperator | None = None,
    ) -> BodoSeries:
        """
        Create a BodoSeries from a lazy metadata object.
        """
        assert isinstance(lazy_metadata.head, pd.Series)
        lazy_mgr = get_lazy_single_manager_class()(
            None,
            None,
            result_id=lazy_metadata.result_id,
            nrows=lazy_metadata.nrows,
            head=lazy_metadata.head._mgr,
            collect_func=collect_func,
            del_func=del_func,
            index_data=lazy_metadata.index_data,
            plan=plan,
        )
        return cls.from_lazy_mgr(lazy_mgr, lazy_metadata.head)

    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        """
        Update the series with new metadata.
        """
        assert self._lazy
        assert isinstance(lazy_metadata.head, pd.Series)
        # Call delfunc to delete the old data.
        self._mgr._del_func(self._mgr._md_result_id)
        self._head_s = lazy_metadata.head
        self._mgr._md_nrows = lazy_metadata.nrows
        self._mgr._md_result_id = lazy_metadata.result_id
        self._mgr._md_head = lazy_metadata.head._mgr

    def is_lazy_plan(self):
        """Returns whether the BodoSeries is represented by a plan."""
        return getattr(self._mgr, "_plan", None) is not None

    def execute_plan(self):
        if self.is_lazy_plan():
            return self._mgr.execute_plan()

    @property
    def shape(self):
        """
        Get the shape of the series. Data is fetched from metadata if present, otherwise the data fetched from workers is used.
        """
        from bodo.pandas.plan import count_plan

        if self._exec_state == ExecState.PLAN:
            return (count_plan(self),)
        if self._exec_state == ExecState.DISTRIBUTED:
            return (self._mgr._md_nrows,)
        if self._exec_state == ExecState.COLLECTED:
            return super().shape

    def head(self, n: int = 5):
        """
        Get the first n rows of the series. If head_s is present and n < len(head_s) we call head on head_s.
        Otherwise we use the data fetched from the workers.
        """
        if n == 0 and self._head_s is not None:
            if self._exec_state == ExecState.COLLECTED:
                return self.iloc[:0].copy()
            else:
                assert self._head_s is not None
                return self._head_s.head(0).copy()

        if (self._head_s is None) or (n > self._head_s.shape[0]):
            if bodo.dataframe_library_enabled and isinstance(
                self._mgr, LazyMetadataMixin
            ):
                from bodo.pandas.base import _empty_like

                planLimit = LogicalLimit(
                    _empty_like(self),
                    self._plan,
                    n,
                )

                return wrap_plan(planLimit)
            else:
                return super().head(n)
        else:
            # If head_s is available and larger than n, then use it directly.
            return self._head_s.head(n)

    def __len__(self):
        from bodo.pandas.plan import count_plan

        if self._exec_state == ExecState.PLAN:
            return count_plan(self)
        if self._exec_state == ExecState.DISTRIBUTED:
            return self._mgr._md_nrows
        if self._exec_state == ExecState.COLLECTED:
            return super().__len__()

    def __repr__(self):
        # Pandas repr implementation calls len() first which will execute an extra
        # count query before the actual plan which is unnecessary.
        if self._exec_state == ExecState.PLAN:
            self.execute_plan()
        return super().__repr__()

    @property
    def index(self):
        self.execute_plan()
        return super().index

    @index.setter
    def index(self, value):
        self.execute_plan()
        super()._set_axis(0, value)

    def _get_result_id(self) -> str | None:
        if isinstance(self._mgr, LazyMetadataMixin):
            return self._mgr._md_result_id
        return None

    @property
    def empty(self):
        return len(self) == 0

    @property
    def str(self):
        return BodoStringMethods(self)

    @property
    def dt(self):
        return BodoDatetimeProperties(self)

    @check_args_fallback(unsupported="none")
    def map(self, arg, na_action=None):
        """
        Map values of Series according to an input mapping or function.
        """

        # Get output data type by running the UDF on a sample of the data.
        empty_series = get_scalar_udf_result_type(self, "map", arg, na_action=na_action)

        return _get_series_python_func_plan(
            self._plan, empty_series, "map", (arg, na_action), {}
        )

    @check_args_fallback(supported=["ascending", "na_position", "kind"])
    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: bool = True,
        inplace: bool = False,
        kind: SortKind | None = None,
        na_position: str = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> BodoSeries | None:
        from bodo.pandas.base import _empty_like

        # Validate ascending argument.
        if not isinstance(ascending, bool):
            raise BodoError(
                "DataFrame.sort_values(): argument ascending iterable does not contain only boolean"
            )

        # Validate na_position argument.
        if not isinstance(na_position, str):
            raise BodoError("Series.sort_values(): argument na_position not a string")

        if na_position not in ["first", "last"]:
            raise BodoError(
                "Series.sort_values(): argument na_position does not contain only 'first' or 'last'"
            )

        if kind is not None:
            warnings.warn("sort_values() kind argument ignored")

        ascending = [ascending]
        na_position = [True if na_position == "first" else False]
        cols = [0]

        """ Create 0 length versions of the dataframe as sorted dataframe
            has the same structure. """
        zero_size_self = _empty_like(self)

        return wrap_plan(
            plan=LogicalOrder(
                zero_size_self,
                self._plan,
                ascending,
                na_position,
                cols,
                self._plan.pa_schema,
            ),
        )

    @check_args_fallback(unsupported="all")
    def min(
        self, axis: Axis | None = 0, skipna: bool = True, numeric_only: bool = False
    ):
        return _compute_series_reduce(self, ["min"])[0]

    @check_args_fallback(unsupported="all")
    def max(
        self, axis: Axis | None = 0, skipna: bool = True, numeric_only: bool = False
    ):
        return _compute_series_reduce(self, ["max"])[0]

    @check_args_fallback(unsupported="all")
    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count=0,
        **kwargs,
    ):
        return _compute_series_reduce(self, ["sum"])[0]

    @check_args_fallback(unsupported="all")
    def prod(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count=0,
        **kwargs,
    ):
        return _compute_series_reduce(self, ["product"])[0]

    product = prod

    @check_args_fallback(unsupported="all")
    def count(self):
        return _compute_series_reduce(self, ["count"])[0]

    @check_args_fallback(unsupported="all")
    def mean(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        """Returns sample mean."""
        reduced = _compute_series_reduce(self, ["count", "sum"])
        count, sum = reduced[0], reduced[1]
        if count <= 0:
            return pd.NA
        return sum / count

    @check_args_fallback(supported=["ddof"])
    def std(self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs):
        """Returns sample standard deviation."""
        reduced_self = _compute_series_reduce(self, ["count", "sum"])
        count, sum = reduced_self[0], reduced_self[1]
        if count <= 0 or count <= ddof:
            return pd.NA
        squared = self.map(lambda x: x * x)
        squared_sum = _compute_series_reduce(squared, ["sum"])[0]
        return ((squared_sum - (sum**2) / count) / (count - ddof)) ** 0.5

    @check_args_fallback(unsupported="all")
    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generates descriptive statistics.
        Descriptive statistics include those that summarize the central tendency, dispersion and
        shape of a dataset's distribution, excluding NaN values.
        """
        if not isinstance(self.dtype, pd.ArrowDtype):
            raise BodoLibNotImplementedException(
                "BodoSeries.describe() is not supported for non-Arrow dtypes."
            )

        pa_type = self.dtype.pyarrow_dtype

        if pa.types.is_null(pa_type):
            return pd.Series(
                [0, 0, pd.NA, pd.NA],
                index=["count", "unique", "top", "freq"],
                name=self.name,
            )

        # TODO: Support describe() for non-numeric Series
        if not (
            pa.types.is_unsigned_integer(pa_type)
            or pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
        ):
            raise BodoLibNotImplementedException(
                "Series.describe() is not supported for non-numeric Series yet."
            )

        reduced_self = _compute_series_reduce(self, ["count", "min", "max", "sum"])
        count, min, max, sum = (reduced_self[i] for i in range(4))

        if count == 0:
            return pd.Series(
                [0] + [pd.NA] * 7,
                index=[
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ],
                name=self.name,
                dtype=pd.ArrowDtype(pa.float64()),
            )

        # Evaluate mean
        mean_val = sum / count

        # Evaluate std
        squared = self.map(lambda x: x * x)
        squared_sum = _compute_series_reduce(squared, ["sum"])[0]
        std_val = ((squared_sum - (sum**2) / count) / (count - 1)) ** 0.5

        # TODO [BSE-4970]: implement Series.quantile
        quantile = self.quantile([0.25, 0.5, 0.75])
        result = [
            count,
            mean_val,
            std_val,
            min,
            quantile[0.25],
            quantile[0.5],
            quantile[0.75],
            max,
        ]

        return pd.Series(
            result,
            index=[
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ],
            name=self.name,
        )

    @property
    def ndim(self) -> int:
        return super().ndim

    @check_args_fallback(supported=["func"])
    def aggregate(self, func=None, axis=0, *args, **kwargs):
        """Aggregate using one or more operations."""
        if isinstance(func, list):
            reduced = _compute_series_reduce(self, func)
            return BodoSeries(reduced, index=func, name=self._name)

        elif isinstance(func, str):
            return _compute_series_reduce(self, [func])[0]

        else:
            raise BodoLibNotImplementedException(
                "Series.aggregate() is not supported for the provided arguments yet."
            )

    agg = aggregate


class BodoStringMethods:
    """Support Series.str string processing methods same as Pandas."""

    def __init__(self, series):
        # Validate input series
        allowed_types = allowed_types_map["str_default"]
        if not (
            isinstance(series, BodoSeries)
            and isinstance(series.dtype, pd.ArrowDtype)
            and series.dtype in allowed_types
        ):
            raise AttributeError("Can only use .str accessor with string values!")

        self._series = series
        self._dtype = series.dtype
        self._is_string = series.dtype in (
            pd.ArrowDtype(pa.string()),
            pd.ArrowDtype(pa.large_string()),
        )

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> pt.Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"StringMethods.{name} is not "
                "implemented in Bodo dataframe library for the specified arguments yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            if not name.startswith("_"):
                warnings.warn(BodoLibFallbackWarning(msg))
            return object.__getattribute__(pd.Series(self._series).str, name)

    @check_args_fallback("none")
    def cat(self, others=None, sep=None, na_rep=None, join="left"):
        """
        If others is specified, concatenates the Series and elements of others
        element-wise and returns a Series. If others is not passed, then falls back to
        Pandas, and all values in the Series are concatenated into a single string with a given sep.
        """
        # Validates others is provided, falls back to Pandas otherwise
        if others is None:
            raise BodoLibNotImplementedException(
                "str.cat(): others is not provided: falling back to Pandas"
            )

        # Validates others is a lazy BodoSeries, falls back to Pandas otherwise
        if not isinstance(others, BodoSeries):
            raise BodoLibNotImplementedException(
                "str.cat(): others is not a BodoSeries instance: falling back to Pandas"
            )

        # Validates input series and others series are from same df, falls back to Pandas otherwise
        base_plan, arg_inds = zip_series_plan(self._series, others)
        index = base_plan.empty_data.index

        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(pa.large_string()),
            name=self._series.name,
            index=index,
        )

        return _get_df_python_func_plan(
            base_plan,
            new_metadata,
            "bodo.pandas.series._str_cat_helper",
            (sep, na_rep, *arg_inds),
            {},
            is_method=False,
        )

    @check_args_fallback(unsupported="none")
    def join(self, sep):
        """
        Join lists contained as elements in the Series/Index with passed delimiter.
        If the elements of a Series are lists themselves, join the content of these lists using
        the delimiter passed to the function.
        """

        def join_list(l):
            """Performs String join with sep=sep if list.dtype == String, returns None otherwise."""
            try:
                return sep.join(l)
            except Exception:
                return pd.NA

        validate_dtype("str.join", self)
        series = self._series
        dtype = pd.ArrowDtype(pa.large_string())

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        # If input Series is a series of lists, creates plan that maps 'join_list'.
        if not self._is_string:
            return _get_series_python_func_plan(
                series._plan, new_metadata, "map", (join_list, None), {}
            )

        return _get_series_python_func_plan(
            series._plan, new_metadata, "str.join", (sep,), {}
        )

    def extract(self, pat, flags=0, expand=True):
        """
        Extract capture groups in the regex pat as columns in a DataFrame.
        For each subject string in the Series, extract groups from the first
        match of regular expression pat.
        """
        import re

        pattern = re.compile(pat, flags=flags)
        n_cols = pattern.groups

        # Like Pandas' implementation, raises ValueError when there are no capture groups.
        if n_cols == 0:
            raise ValueError("pattern contains no capture groups")

        group_names = pattern.groupindex
        is_series_output = not expand and n_cols == 1  # In this case, returns a series.

        series = self._series

        if is_series_output:
            dtype = pd.ArrowDtype(pa.large_string())
        else:
            dtype = pd.ArrowDtype(pa.large_list(pa.large_string()))

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        series_out = _get_series_python_func_plan(
            series._plan,
            new_metadata,
            "bodo.pandas.series._str_extract_helper",
            (
                pat,
                expand,
                n_cols,
                flags,
            ),
            {},
            is_method=False,
        )

        # expand=False and n_cols=1: returns series
        if is_series_output:
            return series_out

        n_index_arrays = get_n_index_arrays(index)
        index_cols = tuple(range(1, 1 + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

        assert series_out.is_lazy_plan()

        # Create schema for output DataFrame with n_cols columns
        if not group_names:
            field_list = [
                pa.field(f"{idx}", pa.large_string()) for idx in range(n_cols)
            ]
        else:
            field_list = [
                pa.field(f"{name}", pa.large_string()) for name in group_names.keys()
            ]

        arrow_schema = pa.schema(field_list)
        empty_data = arrow_to_empty_df(arrow_schema)
        empty_data.index = index

        expr = tuple(
            get_col_as_series_expr(idx, empty_data, series_out, index_cols)
            for idx in range(n_cols)
        )

        # Creates DataFrame with n_cols columns
        df_plan = LogicalProjection(
            empty_data,
            series_out._plan,
            expr + index_col_refs,
        )

        return wrap_plan(plan=df_plan)

    @check_args_fallback(unsupported="none")
    def split(self, pat=None, *, n=-1, expand=False, regex=None):
        """
        Split strings around given separator/delimiter.
        Splits the string in the Series/Index from the beginning, at the specified delimiter string.
        """
        return _split_internal(self, "split", pat, n, expand, regex=regex)

    @check_args_fallback(unsupported="none")
    def rsplit(self, pat=None, *, n=-1, expand=False):
        """
        Split strings around given separator/delimiter.
        Splits the string in the Series/Index from the end, at the specified delimiter string.
        """
        return _split_internal(self, "rsplit", pat, n, expand)


class BodoDatetimeProperties:
    """Support Series.dt datetime accessors same as Pandas."""

    def __init__(self, series):
        allowed_types = allowed_types_map["dt_default"]
        # Validates series type
        # Allows duration[ns] type, timestamp any precision without timezone.
        # TODO: timestamp with timezone, other duration types.
        if not (
            isinstance(series, BodoSeries)
            and (
                series.dtype in allowed_types or _is_pd_pa_timestamp_no_tz(series.dtype)
            )
        ):
            raise AttributeError("Can only use .dt accessor with datetimelike values")
        self._series = series
        self._dtype = series.dtype

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> pt.Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"Series.dt.{name} is not "
                "implemented in Bodo dataframe library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            if not name.startswith("_"):
                warnings.warn(BodoLibFallbackWarning(msg))
            return object.__getattribute__(pd.Series(self._series).dt, name)

    @check_args_fallback(unsupported="none")
    def isocalendar(self):
        """Calculate year, week, and day according to the ISO 8601 standard, returns a BodoDataFrame"""
        series = self._series
        dtype = pd.ArrowDtype(
            pa.list_(pa.uint32())
        )  # Match output type of Pandas: UInt32

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        series_out = _get_series_python_func_plan(
            series._plan,
            new_metadata,
            "bodo.pandas.series._isocalendar_helper",
            (),
            {},
            is_method=False,
        )

        n_index_arrays = get_n_index_arrays(index)
        index_cols = tuple(range(1, 1 + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

        # Create schema for output DataFrame with 3 columns
        arrow_schema = pa.schema(
            [pa.field(f"{label}", pa.uint32()) for label in ["year", "week", "day"]]
        )
        empty_data = arrow_to_empty_df(arrow_schema)
        empty_data.index = index

        expr = tuple(
            get_col_as_series_expr(idx, empty_data, series_out, index_cols)
            for idx in range(3)
        )

        assert series_out.is_lazy_plan()

        # Creates DataFrame with 3 columns
        df_plan = LogicalProjection(
            empty_data,
            series_out._plan,
            expr + index_col_refs,
        )

        return wrap_plan(plan=df_plan)

    @property
    def components(self):
        """Calculate year, week, and day according to the ISO 8601 standard, returns a BodoDataFrame"""
        series = self._series
        dtype = pd.ArrowDtype(pa.list_(pa.int64()))

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        series_out = _get_series_python_func_plan(
            series._plan,
            new_metadata,
            "bodo.pandas.series._components_helper",
            (),
            {},
            is_method=False,
        )

        n_index_arrays = get_n_index_arrays(index)
        index_cols = tuple(range(1, 1 + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

        # Create schema for output DataFrame with 3 columns
        arrow_schema = pa.schema(
            [
                pa.field(f"{label}", pa.int64())
                for label in [
                    "days",
                    "hours",
                    "minutes",
                    "seconds",
                    "milliseconds",
                    "microseconds",
                    "nanoseconds",
                ]
            ]
        )
        empty_data = arrow_to_empty_df(arrow_schema)
        empty_data.index = index

        expr = tuple(
            get_col_as_series_expr(idx, empty_data, series_out, index_cols)
            for idx in range(7)
        )

        assert series_out.is_lazy_plan()

        # Creates DataFrame with 3 columns
        df_plan = LogicalProjection(
            empty_data,
            series_out._plan,
            expr + index_col_refs,
        )

        return wrap_plan(plan=df_plan)

    @check_args_fallback(unsupported="none")
    def tz_localize(self, tz=None, ambiguous="NaT", nonexistent="NaT"):
        """Localize tz-naive Datetime Series to tz-aware Datetime Series."""

        if (
            ambiguous != "NaT"
            or nonexistent not in ("shift_forward", "shift_backward", "NaT")
            and not isinstance(nonexistent, pd.Timedelta)
        ):
            raise BodoLibNotImplementedException(
                "BodoDatetimeProperties.tz_localize is unsupported for the given arguments, falling back to Pandas"
            )

        series = self._series
        dtype = pd.ArrowDtype(pa.timestamp("ns", tz))

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        return _get_series_python_func_plan(
            series._plan,
            new_metadata,
            "bodo.pandas.series._tz_localize_helper",
            (
                tz,
                nonexistent,
            ),
            {},
            is_method=False,
        )


def func_name_to_str(func_name):
    """Converts built-in functions to string."""
    if func_name in ("min", "max", "sum", "product", "prod", "count"):
        return func_name
    if func_name == sum:
        return "sum"
    if func_name == max:
        return "max"
    if func_name == min:
        return "min"
    raise BodoLibNotImplementedException(
        f"{func_name}() not supported for BodoSeries reduction."
    )


def map_validate_reduce(func_names, pa_type):
    """Maps validate_reduce to func_names list, returns resulting pyarrow schema."""
    res = []
    for idx in range(len(func_names)):
        func_names[idx] = func_name_to_str(func_names[idx])
        assigned_type = validate_reduce(func_names[idx], pa_type)
        res.append(pa.field(f"{idx}", assigned_type))
    return pa.schema(res)


def validate_reduce(func_name, pa_type):
    """Validates individual function name, returns upcast input type if necessary, otherwise original type."""

    if func_name in (
        "max",
        "min",
    ):
        if isinstance(
            pa_type,
            (pa.DurationType, pa.ListType, pa.LargeListType, pa.StructType, pa.MapType),
        ):
            raise BodoLibNotImplementedException(
                f"{func_name}() not implemented for {pa_type} type."
            )
        return pa_type

    elif func_name in (
        "sum",
        "product",
    ):
        if pa.types.is_unsigned_integer(pa_type):
            return pa.uint64()
        elif pa.types.is_integer(pa_type):
            return pa.int64()
        elif pa.types.is_floating(pa_type):
            return pa.float64()
        else:
            raise BodoLibNotImplementedException(
                f"{func_name}() not implemented for BodoSeries reduction."
            )

    elif func_name in ("count",):
        return pa.int64()
    else:
        raise BodoLibNotImplementedException(
            f"{func_name}() not implemented for {pa_type} type."
        )


def generate_null_reduce(func_names):
    """Generates a list that maps reduction operations to their default values."""
    res = []
    for func_name in func_names:
        if func_name in ("max", "min"):
            res.append(pd.NA)
        elif func_name in ("sum", "count"):
            res.append(0)
        elif func_name == "product":
            res.append(1)
        else:
            raise BodoLibNotImplementedException(f"{func_name}() not implemented.")
    return res


def _compute_series_reduce(bodo_series: BodoSeries, func_names: list[str]):
    """
    Computes a list of reduction functions like ["min", "max"] on a BodoSeries.
    Returns a list of equal length that stores reduction values of each function.
    """

    if not isinstance(bodo_series.dtype, pd.ArrowDtype):
        raise BodoLibNotImplementedException()

    # Drop Index columns since not necessary for reduction output.
    pa_type = bodo_series.dtype.pyarrow_dtype

    if pa.types.is_null(pa_type):
        return generate_null_reduce(func_names)

    new_arrow_schema = map_validate_reduce(func_names, pa_type)
    zero_size_self = arrow_to_empty_df(new_arrow_schema)

    exprs = [
        AggregateExpression(
            zero_size_self,
            bodo_series._plan,
            func_name,
            [0],
            True,  # dropna
        )
        for func_name in func_names
    ]

    plan = LogicalAggregate(
        zero_size_self,
        bodo_series._plan,
        [],
        exprs,
    )
    out_rank = execute_plan(plan)

    df = pd.DataFrame(out_rank)
    res = []
    # TODO: use parallel reduction for slight improvement in very large scales
    for i in range(len(df.columns)):
        func_name = func_names[i]
        reduced_val = getattr(
            df[str(i)], "sum" if func_name == "count" else func_name
        )()
        res.append(reduced_val)
    assert len(res) == len(func_names)
    return res


def _tz_localize_helper(s, tz, nonexistent):
    """Apply tz_localize on individual elements with ambiguous set to 'raise', fill with None."""

    def _tz_localize(d):
        try:
            return d.tz_localize(tz, ambiguous="raise", nonexistent=nonexistent)
        except Exception:
            return None

    return s.map(_tz_localize)


def _isocalendar_helper(s):
    """Maps pandas.Timestamp.isocalendar() to non-null elements, otherwise fills with None."""

    def get_iso(ts):
        if isinstance(ts, pd.Timestamp):
            return list(ts.isocalendar())
        return None

    return s.map(get_iso)


def _components_helper(s):
    """Applies Series.dt.components to input series, maps tolist() to create series."""
    df = s.dt.components
    return pd.Series([df.iloc[i, :].tolist() for i in range(len(s))])


def _str_cat_helper(df, sep, na_rep, left_idx=0, right_idx=1):
    """Concatenates df[idx] for idx in idx_pair, separated by sep."""
    if sep is None:
        sep = ""

    # df is a two-column DataFrame created in zip_series_plan().
    lhs_col = df.iloc[:, left_idx]
    rhs_col = df.iloc[:, right_idx]

    return lhs_col.str.cat(rhs_col, sep, na_rep)


def _get_col_as_series(s, col):
    """Extracts column col from list series and returns as Pandas series."""
    series = pd.Series(
        [
            None
            if (not isinstance(s.iloc[i], list) or len(s.iloc[i]) <= col)
            else s.iloc[i][col]
            for i in range(len(s))
        ]
    )
    return series


def _str_extract_helper(s, pat, expand, n_cols, flags):
    """Performs row-wise pattern matching, returns a series of match lists."""
    is_series_output = not expand and n_cols == 1
    # Type conversion is necessary to prevent ArrowExtensionArray routing
    string_s = s.astype(str)
    extracted = string_s.str.extract(pat, flags=flags, expand=expand)

    if is_series_output:
        return extracted

    def to_extended_list(s):
        """Extends list in each row to match length to n_cols"""
        list_s = s.tolist()
        list_s.extend([pd.NA] * (n_cols - len(s)))
        return list_s

    # Map tolist() to convert DataFrame to Series of lists
    extended_s = extracted.apply(to_extended_list, axis=1)
    return extended_s


def _get_split_len(s, is_split=True, pat=None, n=-1, regex=None):
    """Runs str.split per element in s and returns length of resulting match group for each index."""
    if is_split:
        split_s = s.str.split(pat=pat, n=n, expand=False, regex=regex)
    else:
        split_s = s.str.rsplit(pat=pat, n=n, expand=False)

    def get_len(x):
        """Get length if output of str.split() is numpy array, otherwise 1."""
        return len(x) if isinstance(x, numpy.ndarray) else 1

    return split_s.map(get_len)


def validate_str_cat(lhs, rhs):
    """
    Checks if lhs and rhs are from the same DataFrame.
    Extracts and returns list projections from each plan.
    """

    lhs_list = get_list_projections(lhs._plan)
    rhs_list = get_list_projections(rhs._plan)

    if lhs_list[0] != rhs_list[0]:
        raise BodoLibNotImplementedException(
            "str.cat(): self and others are from distinct DataFrames: falling back to Pandas"
        )

    # Ensures that at least 1 additional layer is present: single ColRefExpression at the least.
    if not (len(lhs_list) > 1 and len(rhs_list) > 1):
        raise BodoLibNotImplementedException(
            "str.cat(): plans should be longer than length 1: falling back to Pandas"
        )

    return lhs_list, rhs_list


def get_list_projections(plan):
    """Returns list projections of plan."""
    if is_single_projection(plan):
        return get_list_projections(plan.args[0]) + [plan]
    else:
        return [plan]


def get_new_idx(idx, first, side):
    """For first layer of expression, uses idx of itself. Otherwise, left=0 and right=1."""
    if first:
        return idx
    elif side == "right":
        return 1
    else:
        return 0


def make_expr(expr, plan, first, schema, index_cols, side="right"):
    """Creates expression lazyplan with new index depending on lhs/rhs."""
    # if expr=None, expr is a dummy padded onto shorter plan. Create a simple ColRefExpression.
    if expr is None:
        idx = 1 if side == "right" else 0
        empty_data = arrow_to_empty_df(pa.schema([schema[idx]]))
        return ColRefExpression(empty_data, plan, idx)
    elif is_col_ref(expr):
        idx = expr.args[1]
        idx = get_new_idx(idx, first, side)
        empty_data = arrow_to_empty_df(pa.schema([expr.pa_schema[0]]))
        return ColRefExpression(empty_data, plan, idx)
    elif is_scalar_func(expr):
        idx = expr.args[2][0]
        idx = get_new_idx(idx, first, side)
        empty_data = arrow_to_empty_df(pa.schema([expr.pa_schema[0]]))
        return PythonScalarFuncExpression(
            empty_data,
            plan,
            expr.args[1],
            (idx,) + tuple(index_cols),
        )
    elif is_arith_expr(expr):
        # TODO: recursively traverse arithmetic expr tree to update col idx.
        raise BodoLibNotImplementedException(
            "Arithmetic expression unsupported yet, falling back to pandas."
        )
    else:
        raise BodoLibNotImplementedException("Unsupported expr type:", expr.plan_class)


def zip_series_plan(lhs, rhs) -> BodoSeries:
    """Takes in two series plan from the same dataframe, zips into single plan."""

    # Validation runs get_list_projections() and ensures length of lists are >1.
    lhs_list, rhs_list = validate_str_cat(lhs, rhs)
    result = lhs_list[0]
    schema, empty_data, first = [], None, True

    # Initializes index columns info.
    columns = lhs_list[0].empty_data.columns
    index = lhs_list[0].empty_data.index
    n_index_arrays = get_n_index_arrays(index)
    n_cols = len(columns)

    default_schema = pa.field("default", pa.large_string())
    left_schema, right_schema = default_schema, default_schema
    left_empty_data, right_empty_data = None, None
    arg_inds = (0, 1)

    # Shortcut for columns of same dataframe cases like df.A.str.cat(df.B) to avoid
    # creating an extra projection (which causes issues in df setitem).
    if (
        len(lhs_list) == 2
        and len(rhs_list) == 2
        and isinstance(lhs_list[1].exprs[0], ColRefExpression)
        and isinstance(rhs_list[1].exprs[0], ColRefExpression)
    ):
        arg_inds = (lhs_list[1].exprs[0].col_index, rhs_list[1].exprs[0].col_index)
        return result, arg_inds

    # Pads shorter list with None values.
    for lhs_part, rhs_part in itertools.zip_longest(
        lhs_list[1:], rhs_list[1:], fillvalue=None
    ):
        # Create the plan for the shared part
        left_expr = None if not lhs_part else lhs_part.args[1][0]
        right_expr = None if not rhs_part else rhs_part.args[1][0]

        # Extracts schema and empty_data from first layer of expressions.
        default_schema = pa.field("default", pa.large_string())

        if left_expr is not None:
            left_schema = left_expr.pa_schema[0]

        if right_expr is not None:
            right_schema = right_expr.pa_schema[0]

        schema = [left_schema, right_schema]

        # Create index metadata.
        index_cols = tuple(range(n_cols, n_cols + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, result))

        left_expr = make_expr(left_expr, result, first, schema, index_cols, "left")
        right_expr = make_expr(right_expr, result, first, schema, index_cols)

        left_expr.empty_data.columns = ["lhs"]
        right_expr.empty_data.columns = ["rhs"]

        if left_expr is not None:
            left_empty_data = left_expr.empty_data

        if right_expr is not None:
            right_empty_data = right_expr.empty_data

        assert left_empty_data is not None and right_empty_data is not None

        empty_data = pd.concat([left_empty_data, right_empty_data])
        empty_data.index = index

        result = LogicalProjection(
            empty_data,
            result,
            (
                left_expr,
                right_expr,
            )
            + index_col_refs,
        )

        # Toggle 'first' off after first iteration.
        if first:
            first = False
            n_cols = 2

    return result, arg_inds


def get_col_as_series_expr(idx, empty_data, series_out, index_cols):
    """
    Extracts indexed column values from list series and
    returns resulting scalar expression.
    """
    return PythonScalarFuncExpression(
        empty_data,
        series_out._plan,
        (
            "bodo.pandas.series._get_col_as_series",
            True,  # is_series
            False,  # is_method
            (idx,),  # args
            {},  # kwargs
        ),
        (0,) + index_cols,
    )


def _get_series_python_func_plan(
    series_proj, empty_data, func_name, args, kwargs, is_method=True
):
    """Create a plan for calling a Series method in Python. Creates a proper
    PythonScalarFuncExpression with the correct arguments and a LogicalProjection.
    """
    # Optimize out trivial df["col"] projections to simplify plans
    if is_single_colref_projection(series_proj):
        source_data = series_proj.args[0]
        input_expr = series_proj.args[1][0]
        col_index = input_expr.args[1]
    else:
        source_data = series_proj
        col_index = 0

    n_cols = len(source_data.empty_data.columns)
    index_cols = range(
        n_cols, n_cols + get_n_index_arrays(source_data.empty_data.index)
    )
    expr = PythonScalarFuncExpression(
        empty_data,
        source_data,
        (
            func_name,
            True,  # is_series
            is_method,  # is_method
            args,  # args
            kwargs,  # kwargs
        ),
        (col_index,) + tuple(index_cols),
    )
    # Select Index columns explicitly for output
    index_col_refs = tuple(make_col_ref_exprs(index_cols, source_data))
    return wrap_plan(
        plan=LogicalProjection(
            empty_data,
            source_data,
            (expr,) + index_col_refs,
        ),
    )


def _split_internal(self, name, pat, n, expand, regex=None):
    """
    Internal template shared by split() and rsplit().
    name=split splits the string in the Series/Index from the beginning,
    at the specified delimiter string, whereas name=rsplit splits from the end.
    """
    if pat is not None and not isinstance(pat, str):
        raise BodoLibNotImplementedException(
            "BodoStringMethods.split() and rsplit() do not support non-string patterns, falling back to Pandas."
        )

    series = self._series
    index = series.head(0).index
    dtype = pd.ArrowDtype(pa.large_list(pa.large_string()))
    is_split = name == "split"

    # When pat is a string and regex=None, the given pat is compiled as a regex only if len(pat) != 1.
    if regex is None and pat is not None and len(pat) != 1:
        regex = True

    empty_series = pd.Series(
        dtype=dtype,
        name=series.name,
        index=index,
    )
    if is_split:
        kwargs = {"pat": pat, "n": n, "expand": False, "regex": regex}
    else:
        kwargs = {"pat": pat, "n": n, "expand": False}

    series_out = _get_series_python_func_plan(
        series._plan,
        empty_series,
        f"str.{name}",
        (),
        kwargs,
    )

    if not expand:
        return series_out

    cnt_empty_series = pd.Series(
        dtype=pd.ArrowDtype(pa.int32()),
        name=series.name,
        index=index,
    )

    length_series = _get_series_python_func_plan(
        series._plan,
        cnt_empty_series,
        "bodo.pandas.series._get_split_len",
        (),
        {"is_split": is_split, "pat": pat, "n": n, "regex": regex},
        is_method=False,
    )

    n_cols = length_series.max()

    n_index_arrays = get_n_index_arrays(index)
    index_cols = tuple(range(1, 1 + n_index_arrays))
    index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

    # Create schema for output DataFrame with n_cols columns
    arrow_schema = pa.schema(
        [pa.field(f"{idx}", pa.large_string()) for idx in range(n_cols)]
    )

    empty_data = arrow_to_empty_df(arrow_schema)
    empty_data.index = index

    expr = tuple(
        get_col_as_series_expr(idx, empty_data, series_out, index_cols)
        for idx in range(n_cols)
    )

    # Creates DataFrame with n_cols columns
    df_plan = LogicalProjection(
        empty_data,
        series_out._plan,
        expr + index_col_refs,
    )

    return wrap_plan(plan=df_plan)


def gen_partition(name):
    """Generates partition and rpartition using generalized template."""

    def partition(self, sep=" ", expand=True):
        """
        Splits string into 3 elements-before the separator, the separator itself,
        and the part after the separator.
        """
        validate_dtype(f"str.{name}", self)

        series = self._series
        dtype = pd.ArrowDtype(pa.list_(pa.large_string()))

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        series_out = _get_series_python_func_plan(
            series._plan,
            new_metadata,
            f"str.{name}",
            (),
            {"sep": sep, "expand": False},
        )
        # if expand=False, return Series of lists
        if not expand:
            return series_out

        n_index_arrays = get_n_index_arrays(index)
        index_cols = tuple(range(1, 1 + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

        # Create schema for output DataFrame with 3 columns
        arrow_schema = pa.schema(
            [pa.field(f"{idx}", pa.large_string()) for idx in range(3)]
        )
        empty_data = arrow_to_empty_df(arrow_schema)
        empty_data.index = index

        expr = tuple(
            get_col_as_series_expr(idx, empty_data, series_out, index_cols)
            for idx in range(3)
        )

        assert series_out.is_lazy_plan()

        # Creates DataFrame with 3 columns
        df_plan = LogicalProjection(
            empty_data,
            series_out._plan,
            expr + index_col_refs,
        )

        return wrap_plan(plan=df_plan)

    return partition


def sig_bind(name, accessor_type, *args, **kwargs):
    """
    Binds args and kwargs to method's signature for argument validation.
    Exception cases, in which methods take *args and **kwargs, are handled separately using sig_map.
    Signatures are manually created and mapped in sig_map, to which the provided arguments are bound.
    """
    accessor_names = {"str.": "BodoStringMethods.", "dt.": "BodoDatetimeProperties."}
    msg = ""
    try:
        if accessor_type + name in sig_map:
            params = [
                inspect.Parameter(param[0], param[1])
                if not param[2]
                else inspect.Parameter(param[0], param[1], default=param[2][0])
                for param in sig_map[accessor_type + name]
            ]
            signature = inspect.Signature(params)
        else:
            if not accessor_type:
                sample_series = pd.Series([])
            elif accessor_type == "str.":
                sample_series = pd.Series(["a"]).str
            elif accessor_type == "dt.":
                sample_series = pd.Series(pd.to_datetime(["2023-01-01"])).dt
            else:
                raise TypeError(
                    "BodoSeries accessors other than '.dt' and '.str' are not implemented yet."
                )

            func = getattr(sample_series, name)
            signature = inspect.signature(func)

        signature.bind(*args, **kwargs)
        return
    # Separated raising error from except statement to avoid nested errors
    except TypeError as e:
        msg = e
    raise TypeError(f"{accessor_names.get(accessor_type, '')}{name}() {msg}")


# Maps Series methods to signatures. Empty default parameter tuple means argument is required.
sig_map: dict[str, list[tuple[str, inspect._ParameterKind, tuple[pt.Any, ...]]]] = {
    "clip": [
        ("lower", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("upper", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("axis", inspect.Parameter.KEYWORD_ONLY, (None,)),
        ("inplace", inspect.Parameter.KEYWORD_ONLY, (False,)),
    ],
    "str.replace": [
        ("to_replace", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("value", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("regex", inspect.Parameter.KEYWORD_ONLY, (False,)),
        ("inplace", inspect.Parameter.KEYWORD_ONLY, (False,)),
    ],
    "str.wrap": [
        ("width", inspect.Parameter.POSITIONAL_OR_KEYWORD, ()),
        ("expand_tabs", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("replace_whitespace", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("drop_whitespace", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("break_long_words", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("break_on_hyphens", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
    "dt.normalize": [],
    "dt.strftime": [
        ("date_format", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
    ],
    "dt.month_name": [
        ("locale", inspect.Parameter.KEYWORD_ONLY, (None,)),
    ],
    "dt.day_name": [
        ("locale", inspect.Parameter.KEYWORD_ONLY, (None,)),
    ],
    "dt.floor": [
        ("freq", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("normalize", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
    "dt.ceil": [
        ("freq", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("normalize", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
    "dt.total_seconds": [],
}


def _is_pd_pa_timestamp_no_tz(dtype):
    """True when dtype is Arrow extension type timestamp (without timezone)"""
    return (
        isinstance(dtype, pd.ArrowDtype)
        and pa.types.is_timestamp(dtype.pyarrow_dtype)
        and dtype.pyarrow_dtype.tz is None
    )


def validate_dtype(name, obj):
    """Validates dtype of input series for Series.<name> methods."""
    if "." not in name:
        return

    dtype = obj._dtype
    accessor = name.split(".")[0]
    if accessor == "str":
        if dtype not in allowed_types_map.get(
            name, [pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.large_string())]
        ):
            raise AttributeError("Can only use .str accessor with string values!")
    if accessor == "dt":
        if dtype not in allowed_types_map.get(
            name, [pd.ArrowDtype(pa.duration("ns"))]
        ) and not _is_pd_pa_timestamp_no_tz(dtype):
            raise AttributeError("Can only use .dt accessor with datetimelike values!")


def gen_method(
    name, return_type, is_method=True, accessor_type="", allowed_types=[str]
):
    """Generates Series methods, supports optional/positional args."""

    def method(self, *args, **kwargs):
        """Generalized template for Series methods and argument validation using signature"""

        validate_dtype(accessor_type + name, self)

        if is_method:
            sig_bind(name, accessor_type, *args, **kwargs)  # Argument validation

        series = self._series if accessor_type else self
        dtype = series.dtype if not return_type else return_type

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        return _get_series_python_func_plan(
            series._plan, new_metadata, accessor_type + name, args, kwargs
        )

    method.__name__ = name
    return method


# Maps series_str_methods to return types
series_str_methods = [
    # idx = 0: Series(String)
    (
        [
            # no args
            "upper",
            "lower",
            "title",
            "swapcase",
            "capitalize",
            "casefold",
            # args
            "strip",
            "lstrip",
            "rstrip",
            "center",
            "get",
            "removeprefix",
            "removesuffix",
            "pad",
            "rjust",
            "ljust",
            "repeat",
            "slice",
            "slice_replace",
            "translate",
            "zfill",
            "replace",
            "wrap",
            "normalize",
            "decode",
        ],
        pd.ArrowDtype(pa.large_string()),
    ),
    # idx = 1: Series(Bool)
    (
        [
            # no args
            "isalpha",
            "isnumeric",
            "isalnum",
            "isdigit",
            "isdecimal",
            "isspace",
            "islower",
            "isupper",
            "istitle",
            # args
            "startswith",
            "endswith",
            "contains",
            "match",
            "fullmatch",
        ],
        pd.ArrowDtype(pa.bool_()),
    ),
    # idx = 2: Series(Int)
    (
        [
            "find",
            "index",
            "rindex",
            "count",
            "rfind",
            "len",
        ],
        pd.ArrowDtype(pa.int64()),
    ),
    # idx = 3: Series(List(String))
    (
        [
            "findall",
        ],
        pd.ArrowDtype(pa.large_list(pa.large_string())),
    ),
    (
        [
            "encode",
        ],
        pd.ArrowDtype(pa.binary()),
    ),
]


# Maps Series.dt accessors to return types
dt_accessors = [
    # idx = 0: Series(Int)
    (
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
            "dayofweek",
            "day_of_week",
            "weekday",
            "dayofyear",
            "day_of_year",
            "daysinmonth",
            "days_in_month",
            "quarter",
            "days",
            "seconds",
            "microseconds",
            "nanoseconds",
        ],
        pd.ArrowDtype(pa.int32()),
    ),
    # idx = 1: Series(Date)
    (
        [
            "date",
        ],
        pd.ArrowDtype(pa.date32()),
    ),
    # idx = 2: Series(Time)
    (
        [
            "time",
        ],
        pd.ArrowDtype(pa.time64("ns")),
    ),
    # idx = 3: Series(Boolean)
    (
        [
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
            "is_leap_year",
        ],
        pd.ArrowDtype(pa.bool_()),
    ),
]


# Maps Series.dt methods to return types
dt_methods = [
    # idx = 0: Series(Timestamp)
    (
        [
            "normalize",
            "floor",
            "ceil",
            "round",
            # TODO: implement end_time
        ],
        pd.ArrowDtype(pa.timestamp("ns")),
    ),
    # idx = 1: Series(Float)
    (
        [
            "total_seconds",
        ],
        pd.ArrowDtype(pa.float64()),
    ),
    # idx = 2: Series(String)
    (
        [
            "month_name",
            "day_name",
            # TODO [BSE-4880]: fix precision of seconds (%S by default prints up to nanoseconds)
            # "strftime",
        ],
        pd.ArrowDtype(pa.large_string()),
    ),
]

# Maps direct Series methods to return types
dir_methods = [
    # idx = 0: Series(Boolean)
    (
        [
            "isin",
            "notnull",
            "isnull",
            "isna",
            "notna",
        ],
        pd.ArrowDtype(pa.bool_()),
    ),
    (  # idx = 1: Series(Float)
        [
            # TODO: implement ffill, bfill,
        ],
        pd.ArrowDtype(pa.float64()),
    ),
    (
        # idx = 2: None(outputdtype == inputdtype)
        [
            "replace",
            "round",
            "clip",
            "abs",
        ],
        None,
    ),
]

allowed_types_map = {
    "str.decode": [
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.binary()),
        pd.ArrowDtype(pa.large_binary()),
    ],
    "str.join": [
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.list_(pa.string())),
        pd.ArrowDtype(pa.list_(pa.large_string())),
        pd.ArrowDtype(pa.large_list(pa.string())),
        pd.ArrowDtype(pa.large_list(pa.large_string())),
    ],
    "str_default": [
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_list(pa.large_string())),
        pd.ArrowDtype(pa.list_(pa.large_string())),
        pd.ArrowDtype(pa.list_(pa.string())),
        pd.ArrowDtype(pa.large_binary()),
        pd.ArrowDtype(pa.binary()),
    ],
    "dt.round": [
        pd.ArrowDtype(pa.timestamp("ns")),
    ],
    "dt_default": [
        pd.ArrowDtype(pa.timestamp("ns")),
        pd.ArrowDtype(pa.date64()),
        pd.ArrowDtype(pa.time64("ns")),
        pd.ArrowDtype(pa.duration("ns")),
    ],
}


def _install_series_str_methods():
    """Install Series.str.<method>() methods."""
    for str_pair in series_str_methods:
        for name in str_pair[0]:
            method = gen_method(name, str_pair[1], accessor_type="str.")
            setattr(BodoStringMethods, name, method)


def _install_series_dt_accessors():
    """Install Series.dt.<acc> accessors."""
    for dt_accessor_pair in dt_accessors:
        for name in dt_accessor_pair[0]:
            accessor = gen_method(
                name, dt_accessor_pair[1], is_method=False, accessor_type="dt."
            )
            setattr(BodoDatetimeProperties, name, property(accessor))


def _install_series_dt_methods():
    """Install Series.dt.<method>() methods."""
    for dt_method_pair in dt_methods:
        for name in dt_method_pair[0]:
            method = gen_method(name, dt_method_pair[1], accessor_type="dt.")
            setattr(BodoDatetimeProperties, name, method)


def _install_series_direct_methods():
    """Install direct Series.<method>() methods."""
    for dir_method_pair in dir_methods:
        for name in dir_method_pair[0]:
            method = gen_method(name, dir_method_pair[1])
            setattr(BodoSeries, name, method)


def _install_str_partitions():
    """Install Series.str.partition and Series.str.rpartition."""
    for name in ["partition", "rpartition"]:
        method = gen_partition(name)
        setattr(BodoStringMethods, name, method)


_install_series_direct_methods()
_install_series_dt_accessors()
_install_series_dt_methods()
_install_series_str_methods()
_install_str_partitions()
