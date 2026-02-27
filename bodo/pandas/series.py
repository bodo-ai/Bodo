from __future__ import annotations

import bisect
import datetime
import inspect
import itertools
import numbers
import sys
import typing as pt
import warnings
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor

import numpy
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas._typing import (
    Axis,
    Level,
    SortKind,
    ValueKeyFunc,
)

import bodo
import bodo.ai
from bodo.ai.backend import Backend
from bodo.ai.utils import (
    get_default_bedrock_request_formatter,
    get_default_bedrock_response_formatter,
)
from bodo.pandas.array_manager import LazySingleArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper, ExecState
from bodo.pandas.managers import LazyMetadataMixin, LazySingleBlockManager
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ArrowScalarFuncExpression,
    CaseExpression,
    ColRefExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    LazyPlan,
    LazyPlanDistributedArg,
    LogicalAggregate,
    LogicalComparisonJoin,
    LogicalDistinct,
    LogicalFilter,
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    LogicalLimit,
    LogicalOperator,
    LogicalOrder,
    LogicalProjection,
    NullExpression,
    PythonScalarFuncExpression,
    UnaryOpExpression,
    _get_df_python_func_plan,
    execute_plan,
    get_proj_expr_single,
    get_single_proj_source_if_present,
    is_arith_expr,
    is_arrow_scalar_func,
    is_col_ref,
    is_python_scalar_func,
    is_single_colref_projection,
    is_single_projection,
    make_col_ref_exprs,
    match_binop_expr_source_plans,
    maybe_make_list,
    reset_index,
)
from bodo.pandas.utils import (
    BodoCompilationFailedWarning,
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    _fix_multi_index_names,
    _get_empty_series_arrow,
    arrow_to_empty_df,
    check_args_fallback,
    fallback_warn,
    fallback_wrapper,
    get_lazy_single_manager_class,
    get_n_index_arrays,
    get_scalar_udf_result_type,
    insert_bodo_scalar,
    scalarOutputNACheck,
    wrap_module_functions_and_methods,
    wrap_plan,
)


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
        df = pd.DataFrame({f"{S.name}": S})
        bodo_S = bodo.pandas.base.from_pandas(df)[f"{S.name}"]
        bodo_S._name = S.name
        bodo_S._head_s.name = S.name
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
                    read_plan = LogicalGetPandasReadParallel(
                        empty_data.to_frame(),
                        nrows,
                        LazyPlanDistributedArg(self),
                    )
                else:
                    read_plan = LogicalGetPandasReadSeq(
                        empty_data.to_frame(),
                        self,
                    )

                # Make sure Series plans are always single expr projections for easier
                # matching later.
                self._source_plan = LogicalProjection(
                    empty_data,
                    read_plan,
                    tuple(
                        make_col_ref_exprs(
                            range(1 + get_n_index_arrays(empty_data.index)), read_plan
                        )
                    ),
                )

                return self._source_plan

        raise NotImplementedError(
            "Plan not available for this manager, recreate this series with from_pandas"
        )

    def __getattribute__(self, name: str):
        """Custom attribute access that triggers a fallback warning for unsupported attributes."""

        ignore_fallback_attrs = [
            "dtype",
            "dtypes",
            "name",
            "to_string",
            "attrs",
            "flags",
            "iloc",
        ]

        cls = object.__getattribute__(self, "__class__")
        base = cls.__mro__[0]

        if (
            name not in base.__dict__
            and name not in ignore_fallback_attrs
            and not name.startswith("_")
            and hasattr(pd.Series, name)
        ):
            msg = (
                f"Series.{name} is not implemented in Bodo DataFrames yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            return fallback_wrapper(
                self, object.__getattribute__(self, name), name, msg
            )

        return object.__getattribute__(self, name)

    @check_args_fallback("all")
    def _cmp_method(self, other, op):
        """Called when a BodoSeries is compared with a different entity (other)
        with the given operator "op".
        """
        from bodo.pandas.base import _empty_like
        from bodo.pandas.scalar import BodoScalar

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = (
            _empty_like(other) if type(other) in {BodoSeries, BodoScalar} else other
        )

        # Compute schema of new series.
        empty_data = zero_size_self._cmp_method(zero_size_other, op)
        assert isinstance(empty_data, pd.Series), "_cmp_method: Series expected"

        lhs_plan, lhs, rhs = _handle_series_binop_args(self._plan, other)

        # Match the source plans of lhs and rhs, if they don't match return None, None
        lhs, rhs = match_binop_expr_source_plans(lhs, rhs)
        if lhs is None and rhs is None:
            raise BodoLibNotImplementedException(
                "binary operation arguments must have the same dataframe source."
            )

        expr = ComparisonOpExpression(
            empty_data,
            lhs,
            rhs,
            op,
        )

        plan = _create_series_binop_plan(lhs_plan, empty_data, expr)
        return wrap_plan(plan=plan)

    def _conjunction_binop(self, other, op):
        """Called when a BodoSeries is element-wise boolean combined with a different entity (other)"""
        from bodo.pandas.base import _empty_like
        from bodo.pandas.scalar import BodoScalar

        if not (
            (type(other) is BodoScalar and other.wrapped_series.dtype.type is bool)
            or (
                isinstance(other, BodoSeries)
                and isinstance(other.dtype, pd.ArrowDtype)
                and other.dtype.type is bool
            )
            or isinstance(other, bool)
        ):
            raise BodoLibNotImplementedException(
                "'other' should be boolean BodoSeries, BodoScalar or a bool. "
                f"Got {type(other).__name__} instead."
            )

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = (
            _empty_like(other) if type(other) in {BodoSeries, BodoScalar} else other
        )

        # Compute schema of new series.
        empty_data = getattr(zero_size_self, op)(zero_size_other)
        assert isinstance(empty_data, pd.Series), (
            "_conjunction_binop: empty_data is not a Series"
        )

        lhs_plan, lhs, rhs = _handle_series_binop_args(self._plan, other)

        lhs, rhs = match_binop_expr_source_plans(lhs, rhs)
        if lhs is None and rhs is None:
            raise BodoLibNotImplementedException(
                "binary operation arguments should have the same dataframe source."
            )
        expr = ConjunctionOpExpression(
            empty_data,
            lhs,
            rhs,
            op,
        )

        plan = _create_series_binop_plan(lhs_plan, empty_data, expr)
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

        self_bool = is_bool(self)
        other_bool = is_bool(other)
        if (
            (is_numeric(self) or self_bool)
            and (is_numeric(other) or other_bool)
            and not (self_bool and other_bool)
        ):
            if self_bool:
                self = self.map({True: 1, False: 0})

            if other_bool:
                if isinstance(other, BodoSeries):
                    other = other.map({True: 1, False: 0})
                else:
                    other = 1 if other else 0
            return self._numeric_binop(other, op, reverse)

        return self._non_numeric_binop(other, op, reverse)

    def _numeric_binop(self, other, op, reverse):
        """Handles op(self, other) when other is a numeric BodoSeries or scalar."""
        from bodo.pandas.base import _empty_like
        from bodo.pandas.scalar import BodoScalar

        # Get empty Pandas objects for self and other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_other = (
            _empty_like(other) if type(other) in (BodoSeries, BodoScalar) else other
        )

        if op in ("__mod__", "__rmod__"):
            empty_data = zero_size_self
        else:
            # Compute schema of new series.
            empty_data = getattr(zero_size_self, op)(zero_size_other)
        assert isinstance(empty_data, pd.Series), (
            "_numeric_binop: empty_data is not a Series"
        )

        lhs_plan, lhs, rhs = _handle_series_binop_args(self._plan, other)

        lhs, rhs = match_binop_expr_source_plans(lhs, rhs)
        if lhs is None and rhs is None:
            raise BodoLibNotImplementedException(
                "binary operation arguments should have the same dataframe source."
            )

        if reverse:
            lhs, rhs = rhs, lhs

        expr = ArithOpExpression(empty_data, lhs, rhs, op)
        plan = _create_series_binop_plan(lhs_plan, empty_data, expr)
        return wrap_plan(plan=plan)

    def _non_numeric_binop(self, other, op, reverse):
        """Handles op(self, other) when other is non-numeric (e.g., pd.DateOffset, str, etc.)."""
        if (
            is_bodo_string_series(self)
            and is_bodo_string_series(other)
            and op in ("__add__", "__radd__")
        ):
            if op == "__add__":
                return self.str.cat(other)
            if op == "__radd__":
                return other.str.cat(self)

        # If other is an iterable, fall back to Pandas.
        elif pd.api.types.is_scalar(other):
            if op == "__add__":
                return self.add(other)
            if op == "__radd__":
                return self.radd(other)
            if op == "__sub__":
                return self.sub(other)
            if op == "__rsub__":
                return self.rsub(other)

        raise BodoLibNotImplementedException(
            f"BodoSeries.{op} is not supported between 'self' of dtype="
            f"{self.dtype} and 'other' of type {type(other).__name__}."
        )

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
    def __mod__(self, other):
        return self._arith_binop(other, "__mod__", False)

    @check_args_fallback("all")
    def __rmod__(self, other):
        return self._arith_binop(other, "__rmod__", True)

    @check_args_fallback("all")
    def __getitem__(self, key):
        """Called when series[key] is used."""
        from bodo.ext import plan_optimizer
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

    @check_args_fallback(unsupported="none")
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: pt.Any, **kwargs: pt.Any
    ):
        """Adds support for simple numpy ufuncs on BodoSeries like np.func(Series)."""
        from bodo.pandas.base import _empty_like

        if method != "__call__" or len(inputs) != 1 or inputs[0] is not self or kwargs:
            raise BodoLibNotImplementedException(
                "ufunc not implemented for BodoSeries yet"
            )

        new_metadata = _get_empty_series_arrow(ufunc(_empty_like(self)))

        return _get_series_func_plan(
            self._plan,
            new_metadata,
            ufunc,
            (),
            {},
            is_method=False,
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
        if n < 0:
            # Convert the negative number of the number not to include to a positive number so the rest of the
            # code can run normally.  Unfortunately, this will likely require a plan execution here.
            n = max(0, len(self) + n)

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

        # Avoid fallback warnings for prints
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
            return super().__repr__()

    @property
    def index(self):
        self.execute_plan()
        index = super().index

        if isinstance(index, pd.MultiIndex):
            index.names = _fix_multi_index_names(index.names)

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

    @property
    def ai(self):
        return BodoSeriesAiMethods(self)

    @property
    def T(self):
        return self

    @check_args_fallback(unsupported="none")
    def map(self, arg, na_action=None, engine="bodo"):
        """
        Map values of Series according to an input mapping or function.
        """
        import bodo

        if engine not in ("bodo", "python"):
            raise TypeError(
                f"Series.map() got unsupported engine: {engine}, expected one of ('bodo', 'python')."
            )

        if engine == "bodo":
            # Import compiler
            import bodo.decorators  # isort:skip # noqa
            from bodo.pandas.utils_jit import (
                cpp_table_to_series_jit,
                get_udf_cfunc_decorator,
                series_to_cpp_table_jit,
            )
            from bodo.utils.typing import BodoError

            empty_series = self.head(0)

            arr_type = bodo.typeof(empty_series).data

            @bodo.jit(cache=True, spawn=False, distributed=False)
            def map_wrapper_inner(series):
                return series.map(arg, na_action=na_action)

            def map_wrapper(in_cpp_table):
                series = cpp_table_to_series_jit(in_cpp_table, arr_type)
                out_series = map_wrapper_inner(series)
                out_cpp_table = series_to_cpp_table_jit(out_series)
                return out_cpp_table

            try:
                # Compile map inner wrapper, get the output type
                empty_series = _get_empty_series_arrow(map_wrapper_inner(empty_series))
            except BodoError as e:
                empty_series = None
                error_msg = str(e)

            assert empty_series is None or isinstance(empty_series.dtype, pd.ArrowDtype)

            # Jit failed to determine dtypes, likely from gaps in our Arrow support.
            if empty_series is not None and pa.types.is_null(
                empty_series.dtype.pyarrow_dtype
            ):
                empty_series = None
                error_msg = "Jit could not determine pyarrow return type from UDF."

            if empty_series is not None:
                bodo.spawn.utils.import_compiler_on_workers()
                # Compile the cfunc and get pointer
                return _get_series_func_plan(
                    self._plan,
                    empty_series,
                    map_wrapper,
                    (),
                    {},
                    cfunc_decorator=get_udf_cfunc_decorator(),
                )
            else:
                msg = (
                    "Series.map(): Compiling user defined function failed or "
                    "encountered an unsupported result type. Falling back to "
                    "Python engine. Add engine='python' to ignore this warning. "
                    "Original error: "
                    f"{error_msg}."
                )
                if bodo.dataframe_library_warn:
                    warnings.warn(BodoCompilationFailedWarning(msg))

        # engine == "python"
        # Get output data type by running the UDF on a sample of the data.
        empty_series = get_scalar_udf_result_type(self, "map", arg, na_action=na_action)

        return _get_series_func_plan(
            self._plan, empty_series, "map", (arg, na_action), {}
        )

    def map_with_state(self, init_state_fn, row_fn, na_action=None, output_type=None):
        """
        Map values of the Series by first initializaing state and then processing
        each row of the series using the given function.  This variant of map is useful
        where the initialization is potentially so expensive that doing it once per
        partition/batch is prohibitive.  This variant performs the initialization only
        once via the init_state_fn function.  That function returns the initiailized
        state which is then passed to each invocation of row_fn along with the given
        row to be processed.

        Args:
            init_state_fn : Callable returning state, which can have any type
            row_fn : Callable taking the state returned by init_state_fn and the
                     row to be processed and returning the row to be included in the
                     output series.
            output_type : if present, is an empty Pandas series specifying the output
                          dtype of the operation.

        Returns:
            A BodoSeries containing the result of running row_fn on each row of the
            current series.
        """
        if output_type is None:
            state = init_state_fn()
            # Get output data type by running the UDF on a sample of the data.
            empty_series = get_scalar_udf_result_type(
                self, "map_with_state", (state, row_fn), na_action=na_action
            )
        else:
            empty_series = output_type

        return _get_series_func_plan(
            self._plan,
            empty_series,
            "map_with_state",
            (init_state_fn, row_fn, na_action),
            {},
        )

    def map_partitions_with_state(
        self, init_state_fn, func, *args, output_type=None, **kwargs
    ):
        """
        Apply a function to each partition of the series with a one-time initialization.

        NOTE: this pickles the function and sends it to the workers, so globals are
        pickled. The use of lazy data structures as globals causes issues.

        Args:
            init_state_fn : Callable returning state, which can have any type
            func (Callable): A callable which takes in a Series as its first
                argument and returns a DataFrame or Series that has the same length
                its input.
            *args: Additional positional arguments to pass to func.
            **kwargs: Additional key-word arguments to pass to func.
            output_type : if present, is an empty Pandas series specifying the output
                          dtype of the operation.

        Returns:
            DataFrame or Series: The result of applying the func.
        """
        if output_type is None:
            state = init_state_fn()
            # Get output data type by running the UDF on a sample of the data.
            empty_series = get_scalar_udf_result_type(
                self, "map_partitions_with_state", (state, func), *args, **kwargs
            )
        else:
            empty_series = output_type

        return _get_series_func_plan(
            self._plan,
            empty_series,
            "map_partitions_with_state",
            (init_state_fn, func, *args),
            kwargs,
        )

    def map_partitions(self, func, *args, **kwargs):
        """
        Apply a function to each partition of the series.

        If self is a lazy plan, then the result will also be a lazy plan
        (assuming result is Series and the dtype can be infered). Otherwise, the supplied
        function will be sent to the workers and executed immediately.

        NOTE: this pickles the function and sends it to the workers, so globals are
        pickled. The use of lazy data structures as globals causes issues.

        Args:
            func (Callable): A callable which takes in a Series as its first
                argument and returns a DataFrame or Series that has the same length
                its input.
            *args: Additional positional arguments to pass to func.
            **kwargs: Additional key-word arguments to pass to func.

        Returns:
            DataFrame or Series: The result of applying the func.
        """
        import bodo.spawn.spawner

        if self._exec_state == ExecState.PLAN:
            required_fallback = False
            try:
                empty_series = get_scalar_udf_result_type(
                    self, "map_partitions", func, *args, **kwargs
                )
            except BodoLibNotImplementedException as e:
                required_fallback = True
                msg = (
                    f"map_partitions(): encountered exception: {e}, while trying to "
                    "build lazy plan. Executing plan and running map_partitions on "
                    "workers (may be slow or run out of memory)."
                )
                fallback_warn(msg)

                self_arg = self.execute_plan()

            if not required_fallback:
                return _get_series_func_plan(
                    self._plan, empty_series, func, args, kwargs
                )
        else:
            self_arg = self

        return bodo.spawn.spawner.submit_func_to_workers(
            func, [], self_arg, *args, **kwargs
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
            raise ValueError(
                "DataFrame.sort_values(): argument ascending iterable does not contain only boolean"
            )

        # Validate na_position argument.
        if not isinstance(na_position, str):
            raise ValueError("Series.sort_values(): argument na_position not a string")

        if na_position not in ["first", "last"]:
            raise ValueError(
                "Series.sort_values(): argument na_position does not contain only 'first' or 'last'"
            )

        if kind is not None:
            if bodo.dataframe_library_warn:
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
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["min"])
        if df.is_lazy_plan():
            return BodoScalar(df["0"])
        ser = df["0"]
        return scalarOutputNACheck(ser[0], ser.dtype)

    @check_args_fallback(unsupported="all")
    def max(
        self, axis: Axis | None = 0, skipna: bool = True, numeric_only: bool = False
    ):
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["max"])
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        ser = df["0"]
        return scalarOutputNACheck(ser[0], ser.dtype)

    @check_args_fallback(unsupported="all")
    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count=0,
        **kwargs,
    ):
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["sum"])
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        return df["0"][0]

    @check_args_fallback(unsupported="all")
    def prod(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count=0,
        **kwargs,
    ):
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["product"])
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        return df["0"][0]

    product = prod

    @check_args_fallback(unsupported="all")
    def count(self):
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["count"])
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        return df["0"][0]

    @check_args_fallback(unsupported="all")
    def mean(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        """Returns sample mean."""
        from bodo.pandas.scalar import BodoScalar

        df = _compute_series_reduce(self, ["mean"])
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        return df["0"][0]

    @check_args_fallback(supported=["ddof"])
    def std(self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs):
        """Returns sample standard deviation."""
        from bodo.pandas.scalar import BodoScalar

        if ddof == 1:
            df = _compute_series_reduce(self, ["std"])
        elif ddof == 0:
            df = _compute_series_reduce(self, ["std_pop"])
        else:
            n = self.count()
            smean = self.mean()
            squared_diffs = (self - smean) ** 2
            variance = squared_diffs.sum() / (n - ddof)
            return numpy.sqrt(float(variance))
        if hasattr(df, "_lazy") and df.is_lazy_plan():
            return BodoScalar(df["0"])
        return df["0"][0]

    @check_args_fallback(supported=["percentiles"])
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
            return BodoSeries(
                ["0", "0", None, None],
                index=["count", "unique", "top", "freq"],
                name=self.name,
            )

        if not (
            pa.types.is_unsigned_integer(pa_type)
            or pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
        ):
            return _nonnumeric_describe(self)

        quantile_qs = [0.25, 0.5, 0.75]

        if percentiles is not None:
            _, percentiles = validate_quantile(percentiles)
            if 0.5 not in percentiles:
                bisect.insort(percentiles, 0.5)
            quantile_qs = percentiles

        quantile_index = [f"{q * 100:g}%" for q in quantile_qs]
        index = ["count", "mean", "std", "min"] + quantile_index + ["max"]

        # Evaluate count and sum
        stats_df = _compute_series_reduce(self, ["count", "mean", "std"])
        stats_df.execute_plan()
        count = stats_df["0"][0]
        mean = stats_df["1"][0]
        std = stats_df["2"][0]
        if count == 0:
            return BodoSeries(
                [0] + [pd.NA] * (len(index) - 1),
                index=index,
                name=self.name,
                dtype=pd.ArrowDtype(pa.float64()),
            )
        count = float(count)  # Float cast to match Pandas behavior

        # Evaluate quantiles, min, and max altogether since KLL tracks exact min and max values
        min_q_max = [0.0] + quantile_qs + [1.0]
        new_arrow_schema = pa.schema(
            [pa.field(f"{val}", pa.float64()) for val in min_q_max]
        )
        zero_size_self = arrow_to_empty_df(new_arrow_schema)

        exprs = [
            AggregateExpression(
                zero_size_self,
                self._plan,
                func_name,
                None,  # udf_args
                [0],
                True,  # dropna
            )
            for func_name in [f"quantile_{val}" for val in min_q_max]
        ]

        plan = LogicalAggregate(
            zero_size_self,
            self._plan,
            [],
            exprs,
        )
        out_rank = execute_plan(plan)
        quantile_df = pd.DataFrame(out_rank)

        result = [count, mean, std] + [quantile_df[str(val)][0] for val in min_q_max]

        return BodoSeries(
            result,
            index=index,
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
            agg = reduced.iloc[0]
            agg.index = func
            agg.rename(self.name, inplace=True)
            return agg

        elif isinstance(func, str):
            from bodo.pandas.scalar import BodoScalar

            df = _compute_series_reduce(self, [func])
            if hasattr(df, "_lazy") and df.is_lazy_plan():
                return BodoScalar(df["0"])
            return df["0"][0]

        else:
            raise BodoLibNotImplementedException(
                "Series.aggregate() is not supported for the provided arguments yet."
            )

    agg = aggregate

    @check_args_fallback(supported=["other"])
    def add(self, other, level=None, fill_value=None, axis=0):
        """Return Addition of series and other, element-wise (binary operator add)."""
        return gen_arith(self, other, "add")

    @check_args_fallback(supported=["other"])
    def sub(self, other, level=None, fill_value=None, axis=0):
        """Return Addition of series and other, element-wise (binary operator radd)."""
        return gen_arith(self, other, "sub")

    @check_args_fallback(supported=["other"])
    def radd(self, other, level=None, fill_value=None, axis=0):
        """Return Subtraction of series and other, element-wise (binary operator sub)."""
        return gen_arith(self, other, "radd")

    @check_args_fallback(supported=["other"])
    def rsub(self, other, level=None, fill_value=None, axis=0):
        """Return Subtraction of series and other, element-wise (binary operator rsub)."""
        return gen_arith(self, other, "rsub")

    @check_args_fallback(unsupported="none")
    def isin(self, values):
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.
        """
        from bodo.ext import plan_optimizer
        from bodo.pandas.base import _empty_like

        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(pa.bool_()),
            name=self.name,
            index=self.head(0).index,
        )

        if isinstance(values, BodoSeries):
            # Drop duplicate values in 'values' to avoid unnecessary work
            zero_size_values = _empty_like(values)
            if not isinstance(zero_size_values.index, pd.RangeIndex):
                # Drop Index arrays since distinct backend does not support non-key
                # columns yet.
                zero_size_values = zero_size_values.reset_index(drop=True)
                exprs = make_col_ref_exprs([0], values._plan)
                distinct_input_plan = LogicalProjection(
                    zero_size_values,
                    values._plan,
                    exprs,
                )
            else:
                distinct_input_plan = values._plan
            exprs = make_col_ref_exprs([0], distinct_input_plan)
            values_plan = LogicalDistinct(
                zero_size_values,
                distinct_input_plan,
                exprs,
            )

            empty_left = _empty_like(self)
            empty_left.name = None
            # Mark column is after the left columns in DuckDB, see:
            # https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/planner/operator/logical_join.cpp#L20
            empty_join_out = pd.concat(
                [
                    empty_left,
                    pd.Series(
                        [], dtype=pd.ArrowDtype(pa.bool_()), index=empty_left.index
                    ),
                ],
                axis=1,
            )
            empty_join_out.index = empty_left.index
            planComparisonJoin = LogicalComparisonJoin(
                empty_join_out,
                self._plan,
                values_plan,
                plan_optimizer.CJoinType.MARK,
                [(0, 0)],
            )

            # Can't use make_col_ref_exprs since output type is not in input schema
            empty_col_data = arrow_to_empty_df(
                pa.schema([pa.field("mark", pa.bool_())])
            )
            n_indices = get_n_index_arrays(new_metadata.index)
            mark_col = ColRefExpression(
                empty_col_data, planComparisonJoin, n_indices + 1
            )

            # Ignore data column of left side, only Index columns and mark column
            col_indices = list(range(1, n_indices + 1))
            exprs = make_col_ref_exprs(col_indices, planComparisonJoin)
            proj_plan = LogicalProjection(
                new_metadata,
                planComparisonJoin,
                [mark_col] + exprs,
            )

            return wrap_plan(proj_plan)

        # It's just a map function if 'values' is not a BodoSeries
        return _get_series_func_plan(self._plan, new_metadata, "isin", (values,), {})

    @check_args_fallback(supported=["drop", "name", "level"])
    def reset_index(
        self,
        level=None,
        *,
        drop=False,
        name=lib.no_default,
        inplace=False,
        allow_duplicates=False,
    ):
        """
        Generate a new DataFrame or Series with the index reset.
        This is useful when the index needs to be treated as a column, or when the index is meaningless and
        needs to be reset to the default before another operation.
        """
        return reset_index(self, drop, level, name=name)

    @check_args_fallback(unsupported=["interpolation"])
    def quantile(self, q=0.5, interpolation=lib.no_default):
        """Return value at the given quantile."""

        if not isinstance(self.dtype, pd.ArrowDtype):
            raise BodoLibNotImplementedException()

        is_list, q = validate_quantile(q)
        index = [str(float(val)) for val in q] if is_list else []

        # Drop Index columns since not necessary for reduction output.
        pa_type = self.dtype.pyarrow_dtype

        if pa.types.is_null(pa_type):
            return (
                BodoSeries(
                    [pd.NA] * len(q), index=index, dtype=pd.ArrowDtype(pa.float64())
                )
                if is_list
                else pd.NA
            )

        if not (pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type)):
            raise BodoLibNotImplementedException(
                "BodoSeries.quantile() is not supported for non-numeric dtypes."
            )

        new_arrow_schema = pa.schema([pa.field(f"{val}", pa.float64()) for val in q])
        zero_size_self = arrow_to_empty_df(new_arrow_schema)

        exprs = [
            AggregateExpression(
                zero_size_self,
                self._plan,
                func_name,
                None,  # udf_args
                [0],
                True,  # dropna
            )
            for func_name in [f"quantile_{val}" for val in q]
        ]

        plan = LogicalAggregate(
            zero_size_self,
            self._plan,
            [],
            exprs,
        )
        out_rank = execute_plan(plan)

        df = pd.DataFrame(out_rank)
        res = []
        cols = df.columns

        # Return as scalar if q is a scalar value.
        if not is_list:
            return df[cols[0]][0]

        # Otherwise, return a BodoSeries with quantile values.
        for i in range(len(cols)):
            res.append(df[cols[i]][0])

        return BodoSeries(
            res, index=index, dtype=pd.ArrowDtype(pa.float64()), name=self.name
        )

    @check_args_fallback(supported=["cond", "other"])
    def where(
        self,
        cond,
        other=pd.NA,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> BodoSeries | None:
        """Replace values where the condition is False. Creates a case/when expression
        in the plan.
        """
        from bodo.pandas.base import _empty_like
        from bodo.pandas.scalar import BodoScalar

        # Check for BodoSeries condition and BodoSeries/scalar other.
        if not isinstance(cond, BodoSeries):  # pragma: no cover
            raise BodoLibNotImplementedException(
                "Series.where: cond must be a BodoSeries"
            )

        if not (
            type(other) in (BodoSeries, BodoScalar) or pd.api.types.is_scalar(other)
        ):  # pragma: no cover
            raise BodoLibNotImplementedException(
                "Series.where: other must be a scalar or a BodoSeries or BodoScalar"
            )

        # Get empty Pandas objects for self and cond/other with same schema.
        zero_size_self = _empty_like(self)
        zero_size_cond = _empty_like(cond)
        zero_size_other = (
            _empty_like(other) if type(other) in (BodoSeries, BodoScalar) else other
        )

        # Compute schema of new series.
        empty_data = zero_size_self.where(zero_size_cond, zero_size_other)
        assert isinstance(empty_data, pd.Series), (
            "Series.where: empty_data is not a Series"
        )

        orig_other = other
        lhs_plan, lhs, other = _handle_series_binop_args(self._plan, other)
        if other is pd.NA:
            other = NullExpression(zero_size_self, lhs_plan, 0)

        cond = get_proj_expr_single(cond._plan)

        # If BodoScalar changes the source plan, update cond source plan too.
        if (
            type(orig_other) is BodoScalar
            and orig_other.is_lazy_plan()
            and self._plan.source == cond.source
        ):
            cond = cond.with_new_source(lhs_plan)

        # Match source plans of arguments
        lhs, other = match_binop_expr_source_plans(lhs, other)
        if lhs is None and other is None:  # pragma: no cover
            raise BodoLibNotImplementedException(
                "Series.where operation arguments should have the same dataframe source."
            )
        lhs, cond = match_binop_expr_source_plans(lhs, cond)
        if lhs is None and cond is None:  # pragma: no cover
            raise BodoLibNotImplementedException(
                "Series.where operation arguments should have the same dataframe source."
            )
        lhs, other = match_binop_expr_source_plans(lhs, other)
        if lhs is None and other is None:  # pragma: no cover
            raise BodoLibNotImplementedException(
                "Series.where operation arguments should have the same dataframe source."
            )

        expr = CaseExpression(empty_data, cond, lhs, other)

        plan = _create_series_binop_plan(lhs_plan, empty_data, expr)
        return wrap_plan(plan=plan)

    @check_args_fallback(supported="none")
    def cumsum(self, axis: Axis | None = None, skipna: bool = True, *args, **kwargs):
        # cumsum not supported for pyarrow boolean so convert to int
        # Fix in pyarrow instead?
        if self.dtype == pd.ArrowDtype(pa.bool_()):
            self = self.map({True: 1, False: 0})
        msg = (
            "Series.cumsum is not implemented in Bodo DataFrames yet. "
            "Falling back to Pandas (may be slow or run out of memory)."
        )
        fallback_warn(msg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
            with bodo.pandas.utils.FallbackContext():
                py_res = super().cumsum(axis, skipna, *args, **kwargs)

        # Convert objects to Bodo before returning them to the user.
        if bodo.pandas.utils.FallbackContext.is_top_level():
            return bodo.pandas.utils.convert_to_bodo(py_res)
        return py_res

    @check_args_fallback(supported=["dtype_backend"])
    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: str = "numpy_nullable",
    ):
        if dtype_backend == "pyarrow":
            # Pandas is buggy for this case and drops timezone info from timestamps
            return self

        raise BodoLibNotImplementedException(
            "convert_dtypes() only supports dtype_backend='pyarrow' in Bodo Series."
        )


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
                "implemented in Bodo DataFrames for the specified arguments yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            if not name.startswith("_"):
                fallback_warn(msg)
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
            return _get_series_func_plan(
                series._plan, new_metadata, "map", (join_list, None), {}
            )

        return _get_series_func_plan(series._plan, new_metadata, "str.join", (sep,), {})

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

        series_out = _get_series_func_plan(
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


class BodoSeriesAiMethods:
    def __init__(self, series):
        self._series = series

    def tokenize(
        self,
        tokenizer: Callable[[], transformers.PreTrainedTokenizerBase]  # noqa: F821
        | transformers.PreTrainedTokenizerBase,  # noqa: F821
    ) -> BodoSeries:
        self._check_ai_input("tokenize")

        try:
            import transformers
        except ImportError:
            raise ImportError(
                "Series.ai.tokenize() requires the 'transformers' package to be installed. "
                "Please install it using 'pip install transformers'."
            )
        if isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            tokenizer_func = lambda: tokenizer
        else:
            tokenizer_func = tokenizer

        def per_row(tokenizer, row):
            return tokenizer.encode(row, add_special_tokens=True)

        list_of_int64 = pa.list_(pa.int64())
        return self._series.map_with_state(
            tokenizer_func,
            per_row,
            output_type=pd.Series(dtype=pd.ArrowDtype(list_of_int64)),
        )

    def llm_generate(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], str] | None = None,
        region: str | None = None,
        backend: Backend = Backend.OPENAI,
        **generation_kwargs,
    ) -> BodoSeries:
        self._check_ai_input("llm_generate")

        if backend == Backend.BEDROCK:
            if model is None:
                raise ValueError(
                    "Series.ai.llm_generate() requires a model ID when using the Bedrock backend."
                )
            if base_url is not None:
                raise ValueError(
                    "Series.ai.llm_generate() does not support base_url with the Bedrock backend."
                )
            return self._llm_generate_bedrock(
                modelId=model,
                request_formatter=request_formatter,
                response_formatter=response_formatter,
                region=region,
                **generation_kwargs,
            )

        # OpenAI backend
        if request_formatter is not None or response_formatter is not None:
            raise ValueError(
                "Series.ai.llm_generate() does not support request_formatter or response_formatter with the OpenAI backend."
            )
        if region is not None:
            raise ValueError(
                "Series.ai.llm_generate() does not support region with the OpenAI backend."
            )

        import importlib

        assert importlib.util.find_spec("openai") is not None, (
            "Series.ai.llm_generate() requires the 'openai' package to be installed. "
            "Please install it using 'pip install openai[aiohttp]'."
        )

        if model is not None:
            generation_kwargs["model"] = model

        def map_func(series, api_key, base_url, generation_kwargs):
            import asyncio

            import openai

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                # TODO: The below should have better performance but currently
                # pixi won't solve the dependencies.
                # http_client=openai.DefaultAioHttpClient(),
            )

            async def per_row(row, client, generation_kwargs):
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": row}],
                    **generation_kwargs,
                )
                return response.choices[0].message.content

            async def all_tasks(series, client, generation_kwargs):
                tasks = [per_row(row, client, generation_kwargs) for row in series]
                return await asyncio.gather(*tasks, return_exceptions=True)

            # Check if there is a running event loop to determine if we need to run in a thread pool
            # or if we can run the async function directly.
            run_in_threadpool = False
            try:
                asyncio.get_running_loop()
                run_in_threadpool = True
            except RuntimeError:
                pass
            if run_in_threadpool:
                with ThreadPoolExecutor(1) as pool:
                    return pool.submit(
                        lambda: pd.Series(
                            asyncio.run(all_tasks(series, client, generation_kwargs))
                        )
                    ).result()
            else:
                return pd.Series(
                    asyncio.run(all_tasks(series, client, generation_kwargs))
                )

        return self._series.map_partitions(
            map_func, api_key, base_url, generation_kwargs=generation_kwargs
        )

    def _check_ai_input(self, func: str):
        if self._series.dtype not in ("string[pyarrow]", "large_string[pyarrow]"):
            raise TypeError(
                f"Series.ai.{func}() got unsupported dtype: {self._series.dtype},"
                " expected either large_string[pyarrow] or string[pyarrow]."
            )

    def _llm_generate_bedrock(
        self,
        modelId: str,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], str] | None = None,
        region: str = None,
        **generation_kwargs,
    ) -> BodoSeries:
        """Generate text using Amazon Bedrock model.
        Args:
            modelId (str): The Bedrock model ID to use for text generation.
            request_formatter (Callable[[str], str], optional): A function that formats the input text
                into the required JSON format for the Bedrock model. Defaults to None, which uses a default formatter for Nova, Titan, Claude, and OpenAI models.
            response_formatter (Callable[[str], str], optional): A function that formats the response
                from the Bedrock model into a string. Defaults to None, which uses a default formatter
                for Nova, Titan, Claude, and OpenAI models.
            region (str, optional): The AWS region to use for the Bedrock model. Defaults to None, which uses the default region configured in AWS SDK.
            **generation_kwargs: Additional keyword arguments to pass to the Bedrock invoke_model API.
        """
        if request_formatter is None:
            request_formatter = get_default_bedrock_request_formatter(modelId)
        if response_formatter is None:
            response_formatter = get_default_bedrock_response_formatter(modelId)

        def map_func(series, modelId):
            import boto3
            import botocore.config

            client = boto3.client(
                "bedrock-runtime",
                config=botocore.config.Config(
                    connect_timeout=3600,  # 60 minutes
                    read_timeout=3600,  # 60 minutes
                    retries={"max_attempts": 1},
                    region_name=region,
                ),
            )

            def per_row(row):
                response = client.invoke_model(
                    modelId=modelId,
                    body=request_formatter(row),
                    **generation_kwargs,
                )
                return response_formatter(response["body"].read().decode("utf-8"))

            return pd.Series([per_row(row) for row in series])

        return self._series.map_partitions(map_func, modelId)

    def embed(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], list[float]] | None = None,
        region: str | None = None,
        backend: Backend = Backend.OPENAI,
        **embedding_kwargs,
    ) -> BodoSeries:
        self._check_ai_input("embed")

        if backend == Backend.BEDROCK:
            if model is None:
                raise ValueError(
                    "Series.ai.embed() requires a model ID when using the Bedrock backend."
                )
            if base_url is not None:
                raise ValueError(
                    "Series.ai.embed() does not support base_url with the Bedrock backend."
                )
            return self._embed_bedrock(
                modelId=model,
                request_formatter=request_formatter,
                response_formatter=response_formatter,
                region=region,
                **embedding_kwargs,
            )

        # OpenAI backend
        if request_formatter is not None or response_formatter is not None:
            raise ValueError(
                "Series.ai.embed() does not support request_formatter or response_formatter with the OpenAI backend."
            )
        if region is not None:
            raise ValueError(
                "Series.ai.embed() does not support region with the OpenAI backend."
            )

        import importlib

        assert importlib.util.find_spec("openai") is not None, (
            "Series.ai.embed() requires the 'openai' package to be installed. "
            "Please install it using 'pip install openai[aiohttp]'."
        )

        if model is not None:
            embedding_kwargs["model"] = model

        def map_func(series, api_key, base_url, embedding_kwargs):
            import asyncio

            import openai

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                # TODO: The below should have better performance but currently
                # pixi won't solve the dependencies.
                # http_client=openai.DefaultAioHttpClient(),
            )

            async def per_row(row, client, embedding_kwargs):
                response = await client.embeddings.create(
                    input=row,
                    **embedding_kwargs,
                )
                return response.data[0].embedding

            async def all_tasks(series, client, embedding_kwargs):
                tasks = [per_row(row, client, embedding_kwargs) for row in series]
                return await asyncio.gather(*tasks, return_exceptions=True)

            # Check if there is a running event loop to determine if we need to run in a thread pool
            # or if we can run the async function directly.
            run_in_threadpool = False
            try:
                asyncio.get_running_loop()
                run_in_threadpool = True
            except RuntimeError:
                pass
            if run_in_threadpool:
                with ThreadPoolExecutor(1) as pool:
                    return pool.submit(
                        lambda: pd.Series(
                            asyncio.run(all_tasks(series, client, embedding_kwargs))
                        )
                    ).result()
            else:
                return pd.Series(
                    asyncio.run(all_tasks(series, client, embedding_kwargs))
                )

        return self._series.map_partitions(
            map_func, api_key, base_url, embedding_kwargs=embedding_kwargs
        )

    def _embed_bedrock(
        self,
        modelId: str,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], list[float]] | None = None,
        region: str | None = None,
        **embedding_kwargs,
    ) -> BodoSeries:
        """Embed text using Amazon Bedrock model.
        Args:
            modelId (str): The Bedrock model ID to use for embedding.
            request_formatter (Callable[[str], str], optional): A function that formats the input text
                into the required JSON format for the Bedrock model. Defaults to None, which uses a default formatter for Titan embeddings models.
            response_formatter (Callable[[str], list[float]], optional): A function that formats the response
                from the Bedrock model into a list of floats. Defaults to None, which uses a default formatter for Titan embeddings models.
            region (str, optional): The AWS region to use for the Bedrock model. Defaults to None, which uses the default region configured in AWS SDK.
            **embedding_kwargs: Additional keyword arguments to pass to the Bedrock invoke_model API.
        """
        if request_formatter is None:
            request_formatter = get_default_bedrock_request_formatter(modelId)
        if response_formatter is None:
            response_formatter = get_default_bedrock_response_formatter(modelId)

        def map_func(series, modelId):
            import boto3
            import botocore.config

            client = boto3.client(
                "bedrock-runtime",
                config=botocore.config.Config(
                    connect_timeout=3600,  # 60 minutes
                    read_timeout=3600,  # 60 minutes
                    retries={"max_attempts": 1},
                ),
                region_name=region,
            )

            def per_row(row):
                response = client.invoke_model(
                    modelId=modelId,
                    body=request_formatter(row),
                    **embedding_kwargs,
                )
                return response_formatter(response["body"].read())

            return pd.Series([per_row(row) for row in series])

        return self._series.map_partitions(map_func, modelId)

    def query_s3_vectors(
        self,
        vector_bucket_name: str,
        index_name: str,
        topk: int,
        region: str = None,
        filter: dict = None,
        return_distance: bool = False,
        return_metadata: bool = False,
    ):
        """Query S3 vector index and return matching vector data as a BodoDataFrame."""
        series = self._series

        if series.dtype.pyarrow_dtype not in (
            pa.list_(pa.float32()),
            pa.large_list(pa.float32()),
            pa.list_(pa.float64()),
            pa.large_list(pa.float64()),
        ):
            raise TypeError(
                f"Series.ai.query_s3_vectors() got unsupported dtype: {series.dtype}, expected list[float32] or list[float64]."
            )

        index = series.head(0).index
        struct_schema = {"keys": pa.large_list(pa.large_string())}
        if return_distance:
            struct_schema["distances"] = pa.large_list(pa.float32())
        if return_metadata:
            struct_schema["metadata"] = pa.large_list(pa.large_string())

        # Output of projection function is a struct that is expanded into columns of a
        # DataFrame.
        struct_type = pa.struct(struct_schema)
        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(struct_type),
            name=series.name,
            index=index,
        )

        series_out = _get_series_func_plan(
            series._plan,
            new_metadata,
            "bodo.pandas.utils.query_s3_vectors_helper",
            (
                vector_bucket_name,
                index_name,
                region,
                topk,
                filter,
                return_distance,
                return_metadata,
            ),
            {},
            is_method=False,
        )

        n_index_arrays = get_n_index_arrays(index)
        index_cols = tuple(range(1, 1 + n_index_arrays))
        index_col_refs = tuple(make_col_ref_exprs(index_cols, series_out._plan))

        empty_data = arrow_to_empty_df(pa.schema(struct_type))
        empty_data.index = index

        exprs = tuple(
            get_col_as_series_expr(f.name, empty_data, series_out, index_cols)
            for f in struct_type
        )

        df_plan = LogicalProjection(
            empty_data,
            series_out._plan,
            exprs + index_col_refs,
        )
        return wrap_plan(plan=df_plan)


class BodoDatetimeProperties:
    """Support Series.dt datetime accessors same as Pandas."""

    def __init__(self, series):
        allowed_types = allowed_types_map["dt_default"]
        # Validates series type
        # Allows duration[ns] type, timestamp any precision without timezone.
        # TODO: other duration/time types.
        if not (
            isinstance(series, BodoSeries)
            and (series.dtype in allowed_types or _is_pd_pa_timestamp(series.dtype))
        ):
            raise AttributeError(
                f"Can only use .dt accessor with datetimelike values, got {series.dtype} {type(series.dtype)} instead"
            )
        self._series = series
        self._dtype = series.dtype

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> pt.Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"Series.dt.{name} is not "
                "implemented in Bodo DataFrames yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            if not name.startswith("_"):
                fallback_warn(msg)
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

        series_out = _get_series_func_plan(
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

        series_out = _get_series_func_plan(
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
        if _is_pd_pa_timestamp(series.dtype):
            unit = series.dtype.pyarrow_dtype.unit
        else:
            unit = "ns"
        dtype = pd.ArrowDtype(pa.timestamp(unit, tz))

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        return _get_series_func_plan(
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


def is_bool(other):
    from bodo.pandas.scalar import BodoScalar

    if type(other) is BodoScalar:
        dtype = other.wrapped_series.dtype
        return pd.api.types.is_bool_dtype(dtype)

    is_bool_bodoseries = (
        isinstance(other, BodoSeries)
        and isinstance(other.dtype, pd.ArrowDtype)
        and pd.api.types.is_bool_dtype(other.dtype)
    )
    is_bool_scalar = isinstance(other, bool)
    return is_bool_bodoseries or is_bool_scalar


def is_numeric(other):
    """Returns whether other is a numeric BodoSeries/scalar."""
    from bodo.pandas.scalar import BodoScalar

    if type(other) is BodoScalar:
        dtype = other.wrapped_series.dtype
        return pd.api.types.is_numeric_dtype(dtype)

    is_numeric_bodoseries = (
        isinstance(other, BodoSeries)
        and isinstance(other.dtype, pd.ArrowDtype)
        and pd.api.types.is_numeric_dtype(other.dtype)
    )
    is_numeric_scalar = isinstance(other, numbers.Number) and not isinstance(
        other, allowed_types_map["binop_dtlike"]
    )
    return is_numeric_bodoseries or is_numeric_scalar


def _handle_series_binop_args(series_plan: LazyPlan, other):
    """
    Handles the arguments for binary operations on Series and return updated plan
    and expressions.
    BodoScalar updates the plan since it needs to be inserted into the plan.
    """
    from bodo.pandas.scalar import BodoScalar

    # The plan of the parent table without the Series projection node.
    lhs_plan = series_plan.source

    # Extract argument expressions
    lhs = get_proj_expr_single(series_plan)

    # If other is a lazy BodoScalar we need to insert it into the plan.
    if type(other) is BodoScalar:
        if other.is_lazy_plan():
            lhs_plan, rhs = insert_bodo_scalar(lhs_plan, other)
            # Point lhs to the new plan, only the source of the expression changes.
            lhs = lhs.with_new_source(lhs_plan)
        else:
            rhs = other.get_value()

    # If other is a LazyPlan we need to extract the expression.
    elif hasattr(other, "_plan") and isinstance(other._plan, LazyPlan):
        rhs = get_proj_expr_single(other._plan)
    # If other is a Pandas Series or a scalar we can use it directly.
    else:
        rhs = other

    return lhs_plan, lhs, rhs


def _create_series_binop_plan(lhs_plan, empty_data, expr):
    """Create a projection plan for output of binary operations on Series.
    Handles Index columns properly.
    """
    ncols = lhs_plan.empty_data.shape[1]
    key_indices = [
        ncols + i for i in range(get_n_index_arrays(lhs_plan.empty_data.index))
    ]
    key_exprs = tuple(make_col_ref_exprs(key_indices, lhs_plan))

    plan = LogicalProjection(
        empty_data,
        lhs_plan,
        (expr,) + key_exprs,
    )
    return plan


def func_name_to_str(func_name):
    """Converts built-in functions to string."""
    if func_name in ("min", "max", "sum", "product", "count", "mean", "std", "std_pop"):
        return func_name
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
    elif func_name in ("mean", "std", "std_pop"):
        if pd.api.types.is_numeric_dtype(pd.ArrowDtype(pa_type)):
            return pa.float64()
        else:
            raise BodoLibNotImplementedException(
                f"{func_name}() not implemented for {pa_type} type."
            )
    else:
        raise BodoLibNotImplementedException(
            f"{func_name}() not implemented for {pa_type} type."
        )


def generate_null_reduce(func_names):
    """Generates a list that maps reduction operations to their default values."""
    from bodo.pandas.frame import BodoDataFrame

    res = []
    for func_name in func_names:
        if func_name in ("max", "min"):
            res.append(pd.NA)
        elif func_name in ("sum", "count"):
            res.append(0)
        elif func_name == "product":
            res.append(1)
        elif func_name in ("mean", "std"):
            res.append(pd.NA)
        else:
            raise BodoLibNotImplementedException(f"{func_name}() not implemented.")
    return BodoDataFrame({f"{i}": [res[i]] for i in range(len(res))}).execute_plan()


def _compute_series_reduce(bodo_series: BodoSeries, func_names: list[str]):
    """
    Computes a list of reduction functions like ["min", "max"] on a BodoSeries.
    Returns a BodoDataFrame that stores reduction values of each function.
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
            None,  # udf_args
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
    return wrap_plan(plan=plan)


def validate_quantile(q):
    """Validates that quantile input falls in the range [0, 1].
    Taken from Pandas validation code for percentiles to produce the same behavior as Pandas.
    https://github.com/pandas-dev/pandas/blob/d4ae6494f2c4489334be963e1bdc371af7379cd5/pandas/util/_validators.py#L311"""
    from pandas.api.types import is_list_like

    is_list = is_list_like(q)

    q_arr = numpy.asarray(q)
    msg = "percentiles should all be in the interval [0, 1]"
    if q_arr.ndim == 0:
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    else:
        if not all(0 <= qs <= 1 for qs in q_arr):
            raise ValueError(msg)

    return is_list, maybe_make_list(q)


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

    # Work around Pandas 3 bugs with ArrowDtype str.cat()
    out = lhs_col.astype(object).str.cat(rhs_col.astype(object), sep, na_rep)
    return out.astype("str").astype(pd.ArrowDtype(pa.large_string()))


def _get_col_as_series(s, col):
    """Extracts column col from list series and returns as Pandas series."""

    # Extract column from struct case
    if isinstance(col, str):
        return pd.Series(pa.table(s.array._pa_array).column(col))

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
        return extracted.astype(pd.ArrowDtype(pa.large_string()))

    def to_extended_list(s):
        """Extends list in each row to match length to n_cols"""
        list_s = s.astype(object).replace(np.nan, None).tolist()
        list_s.extend([None] * (n_cols - len(s)))
        return list_s

    # Map tolist() to convert DataFrame to Series of lists
    if len(extracted) == 0:
        return pd.Series([], dtype=pd.ArrowDtype(pa.large_list(pa.large_string())))

    extended_s = extracted.apply(to_extended_list, axis=1)
    return extended_s.astype(pd.ArrowDtype(pa.large_list(pa.large_string())))


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


def _nonnumeric_describe(series):
    """Computes non-numeric series.describe() using DataFrameGroupBy."""

    # Since Series groupby is unsupported, we toggle is_series to use DataFrameGroupBy.
    plan = series._plan
    plan.is_series = False
    plan.empty_data.columns = pd.Index(["A"])
    df = wrap_plan(plan)

    # size() aggregation is not supported with single-column DataFrames.
    # The workaround is setting a duplicate column.
    df.columns = pd.Index(["None"])
    df["B"] = df["None"]
    gb = df.groupby("None")

    gb_size = gb.agg("size")  # Plan execution
    count_val = gb_size.sum()  # Plan execution
    unique_val = len(gb_size.index)
    gb_sorted = gb_size.sort_values(ascending=False)
    top_val = gb_sorted.index[0]
    freq_val = gb_sorted.iloc[0]  # Plan execution

    return bodo.pandas.BodoSeries(
        [f"{count_val}", f"{unique_val}", f"{top_val}", f"{freq_val}"],
        name=series.name,
        index=pd.Index(["count", "unique", "top", "freq"]),
    )


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
    elif is_python_scalar_func(expr):
        idx = expr.input_column_indices[0]
        idx = get_new_idx(idx, first, side)
        empty_data = arrow_to_empty_df(pa.schema([expr.pa_schema[0]]))
        return PythonScalarFuncExpression(
            empty_data,
            plan,
            expr.func_args,
            (idx,) + tuple(index_cols),
            expr.is_cfunc,
            False,
        )
    elif is_arrow_scalar_func(expr):
        idx = expr.input_column_indices[0]
        idx = get_new_idx(idx, first, side)
        empty_data = arrow_to_empty_df(pa.schema([expr.pa_schema[0]]))
        return ArrowScalarFuncExpression(
            empty_data, plan, (idx,) + tuple(index_cols), expr.function_name, ()
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
            True,  # use_arrow_dtypes
        ),
        (0,) + index_cols,
        False,  # is_cfunc
        False,  # has_state
    )


def _get_series_func_plan(
    series_proj,
    empty_data,
    func,
    args,
    kwargs,
    is_method=True,
    cfunc_decorator=None,
    use_arrow_dtypes=None,
):
    """Create a plan for calling a Series method in Python. Creates a proper
    ScalarFuncExpression with the correct arguments and a LogicalProjection.
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

    # List of Series methods to be routed to Arrow Compute
    arrow_compute_list = (
        "dt.hour",
        "dt.month",
        "dt.dayofweek",
        "dt.day_of_week",
        "dt.quarter",
        "dt.year",
        "dt.day",
        "dt.minute",
        "dt.second",
        "dt.microsecond",
        "dt.nanosecond",
        "dt.weekday",
        "dt.dayofyear",
        "dt.day_of_year",
        # string methods that correspond to utf8_{name}
        "str.isalnum",
        "str.isalpha",
        "str.isdecimal",
        "str.isdigit",
        "str.isnumeric",
        "str.isupper",
        "str.isspace",
        "str.capitalize",
        "str.length",
        "str.lower",
        "str.upper",
        "str.swapcase",
        "str.title",
        "str.reverse",
        "str.match",
    )

    def get_arrow_func(name):
        """Maps method name to its corresponding Arrow Compute Function name."""
        if name in ("dt.dayofweek", "dt.weekday"):
            return "day_of_week"
        if name == "dt.dayofyear":
            return "day_of_year"
        if name.startswith("str.is"):
            body = name.split(".")[1]
            return "utf8_" + body[:2] + "_" + body[2:]
        if name == "str.match":
            return "match_substring_regex"
        if name.startswith("str."):
            return "utf8_" + name.split(".")[1]
        return name.split(".")[1]

    if func in arrow_compute_list and len(kwargs) == 0:
        func_name = get_arrow_func(func)
        func_args = tuple(args)
        is_cfunc = False
        has_state = False
        expr = ArrowScalarFuncExpression(
            empty_data,
            source_data,
            (col_index,) + tuple(index_cols),
            func_name,
            func_args,
        )
    else:
        # Empty func_name separates Python calls from Arrow calls.
        has_state = func in ("map_with_state", "map_partitions_with_state")
        if cfunc_decorator:
            func_args = (func, cfunc_decorator)
            is_cfunc = True
        else:
            func_args = (
                func,
                True,  # is_series
                is_method,  # is_method
                args,  # args
                kwargs,  # kwargs
                use_arrow_dtypes,
            )
            is_cfunc = False

        expr = PythonScalarFuncExpression(
            empty_data,
            source_data,
            func_args,
            (col_index,) + tuple(index_cols),
            is_cfunc,
            has_state,
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

    series_out = _get_series_func_plan(
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

    length_series = _get_series_func_plan(
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

        series_out = _get_series_func_plan(
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

        bound_sig = signature.bind(*args, **kwargs)
        return bound_sig
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
    "replace": [
        ("to_replace", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("value", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("regex", inspect.Parameter.KEYWORD_ONLY, (False,)),
        ("inplace", inspect.Parameter.KEYWORD_ONLY, (False,)),
    ],
    "str.replace": [
        ("pat", inspect.Parameter.POSITIONAL_OR_KEYWORD, ()),
        ("repl", inspect.Parameter.POSITIONAL_OR_KEYWORD, ()),
        ("n", inspect.Parameter.POSITIONAL_OR_KEYWORD, (-1,)),
        ("case", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("flags", inspect.Parameter.POSITIONAL_OR_KEYWORD, (0,)),
        ("regex", inspect.Parameter.POSITIONAL_OR_KEYWORD, (False,)),
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


def _is_pd_pa_timestamp(dtype):
    """True when dtype is Arrow extension type timestamp"""
    return isinstance(dtype, pd.ArrowDtype) and pa.types.is_timestamp(
        dtype.pyarrow_dtype
    )


def gen_arith(self, other, name):
    """Generates Series.add/radd/sub/rsub."""
    if isinstance(
        other,
        (
            BodoSeries,
            pd.Series,
        ),
    ):
        raise BodoLibNotImplementedException(
            f"Series.{name}() is not supported for other of type {type(other)} yet."
        )
    if (
        name
        in (
            "sub",
            "rsub",
        )
        and self.dtype in allowed_types_map["str_default"]
    ):
        raise TypeError("Unsupported operand type(s) for -: 'str' and 'str'")
    return gen_method(name, self.dtype)(self, other)


def is_bodo_string_series(self):
    """Returns True if self is a BodoSeries with dtype String."""
    return type(self) is BodoSeries and self.dtype in allowed_types_map["str_default"]


def validate_method_args(name, bound_sig):
    """Validates args and kwargs for Series.<name> methods.
    TODO: validate other methods as needed.
    """
    if name == "str.replace":
        repl = bound_sig.arguments.get("repl", None)
        # Same as Pandas validation
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")


def validate_dtype(name, obj):
    """Validates dtype of input series for Series.<name> methods."""
    if "." not in name:
        return

    dtype = obj._dtype
    accessor = name.split(".")[0]
    if accessor == "str":
        if dtype not in allowed_types_map.get(
            name, (pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.large_string()))
        ):
            raise AttributeError("Can only use .str accessor with string values!")
    if accessor == "dt":
        if dtype not in allowed_types_map.get(
            name, allowed_types_map["dt_default"]
        ) and not _is_pd_pa_timestamp(dtype):
            raise AttributeError("Can only use .dt accessor with datetimelike values!")


def gen_method(
    name, return_type, is_method=True, accessor_type="", allowed_types=[str]
):
    """Generates Series methods, supports optional/positional args."""

    def method(self, *args, **kwargs):
        """Generalized template for Series methods and argument validation using signature"""

        validate_dtype(accessor_type + name, self)

        if is_method:
            # Argument validation
            bound_sig = sig_bind(name, accessor_type, *args, **kwargs)
            validate_method_args(accessor_type + name, bound_sig)

        series = self._series if accessor_type else self
        dtype = series.dtype if not return_type else return_type

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        return _get_series_func_plan(
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
    # idx = 0: Series(Int64)
    (
        # NOTE: The methods below return int32 in Pandas by default.
        # In Bodo, the output dtype is int64 because we use PyArrow Compute.
        [
            "hour",
            "month",
            "dayofweek",
            "day_of_week",
            "quarter",
            "year",
            "day",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
            "weekday",
            "dayofyear",
        ],
        pd.ArrowDtype(pa.int64()),
    ),
    # idx = 0: Series(Int32)
    (
        [
            "daysinmonth",
            "days_in_month",
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
        None,  # preserves timezone, scale
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
    "str.decode": (
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.binary()),
        pd.ArrowDtype(pa.large_binary()),
    ),
    "str.join": (
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.list_(pa.string())),
        pd.ArrowDtype(pa.list_(pa.large_string())),
        pd.ArrowDtype(pa.large_list(pa.string())),
        pd.ArrowDtype(pa.large_list(pa.large_string())),
    ),
    "str_default": (
        pd.ArrowDtype(pa.large_string()),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_list(pa.large_string())),
        pd.ArrowDtype(pa.list_(pa.large_string())),
        pd.ArrowDtype(pa.list_(pa.string())),
        pd.ArrowDtype(pa.large_binary()),
        pd.ArrowDtype(pa.binary()),
    ),
    "dt.round": (pd.ArrowDtype(pa.timestamp("ns")),),
    "dt_default": (
        pd.ArrowDtype(pa.timestamp("ns")),
        pd.ArrowDtype(pa.date64()),
        pd.ArrowDtype(pa.date32()),
        pd.ArrowDtype(pa.time64("ns")),
        pd.ArrowDtype(pa.duration("ns")),
    ),
    "binop_scalar": (
        int,
        float,
        str,
        bool,
        pd.Timedelta,
        pd.DateOffset,
        datetime.timedelta,
        datetime.datetime,
        numpy.datetime64,
        numpy.timedelta64,
        numpy.int64,
        numpy.float64,
        numpy.bool_,
    ),
    "binop_dtlike": (
        pd.Timedelta,
        pd.DateOffset,
        datetime.timedelta,
        datetime.datetime,
        numpy.datetime64,
        numpy.timedelta64,
    ),
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

wrap_module_functions_and_methods(sys.modules[__name__])
