from __future__ import annotations

import inspect
import numbers
import typing as pt
import warnings
from collections.abc import Callable, Hashable

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
from bodo.pandas.utils import (
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    LazyPlan,
    LazyPlanDistributedArg,
    arrow_to_empty_df,
    check_args_fallback,
    get_lazy_single_manager_class,
    get_n_index_arrays,
    get_proj_expr_single,
    get_scalar_udf_result_type,
    is_single_colref_projection,
    make_col_ref_exprs,
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
                    if getattr(self._mgr, "_md_result_id", None) is not None:
                        # If the plan has been executed but the results are still
                        # distributed then re-use those results as is.
                        nrows = self._mgr._md_nrows
                        res_id = self._mgr._md_result_id
                        mgr = self._mgr
                    else:
                        # The data has been collected and is no longer distributed
                        # so we need to re-distribute the results.
                        nrows = len(self)
                        res_id = bodo.spawn.utils.scatter_data(self)
                        mgr = None
                    self._source_plan = LazyPlan(
                        "LogicalGetPandasReadParallel",
                        empty_data,
                        nrows,
                        LazyPlanDistributedArg(mgr, res_id),
                    )
                else:
                    self._source_plan = LazyPlan(
                        "LogicalGetPandasReadSeq",
                        empty_data,
                        self,
                    )

                return self._source_plan

        raise NotImplementedError(
            "Plan not available for this manager, recreate this series with from_pandas"
        )

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
        expr = LazyPlan(
            "ComparisonOpExpression",
            empty_data,
            lhs,
            rhs,
            op,
        )

        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
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
        expr = LazyPlan(
            "ConjunctionOpExpression",
            empty_data,
            lhs,
            rhs,
            op,
        )

        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
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
        expr = LazyPlan(
            "UnaryOpExpression",
            empty_data,
            source_expr,
            "__invert__",
        )
        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
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

        expr = LazyPlan("ArithOpExpression", empty_data, lhs, rhs, op)

        plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
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
        plan: plan_optimizer.LogicalOperator | None = None,
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
        from bodo.pandas.utils import count_plan

        match self._exec_state:
            case ExecState.PLAN:
                return (count_plan(self),)
            case ExecState.DISTRIBUTED:
                return (self._mgr._md_nrows,)
            case ExecState.COLLECTED:
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

                planLimit = LazyPlan(
                    "LogicalLimit",
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
        from bodo.pandas.utils import count_plan

        match self._exec_state:
            case ExecState.PLAN:
                return count_plan(self)
            case ExecState.DISTRIBUTED:
                return self._mgr._md_nrows
            case ExecState.COLLECTED:
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
        Apply function to elements in a Series
        """

        # Get output data type by running the UDF on a sample of the data.
        empty_series = get_scalar_udf_result_type(self, "map", arg, na_action=na_action)

        return _get_series_python_func_plan(
            self._plan, empty_series, "map", (arg, na_action), {}
        )


def _str_partition_helper(s, col):
    """Extracts column col from list series and returns as Pandas series."""
    series = pd.Series(
        [
            None if not isinstance(s.iloc[i], list) else s.iloc[i][col]
            for i in range(len(s))
        ]
    )
    return series

    @check_args_fallback(supported=["ascending", "na_position"])
    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: bool = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
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

        ascending = [ascending]
        na_position = [True if na_position == "first" else False]
        cols = [0]

        """ Create 0 length versions of the dataframe as sorted dataframe
            has the same structure. """
        zero_size_self = _empty_like(self)

        return wrap_plan(
            plan=LazyPlan(
                "LogicalOrder",
                zero_size_self,
                self._plan,
                ascending,
                na_position,
                cols,
                self._plan.pa_schema,
            ),
        )


class BodoStringMethods:
    """Support Series.str string processing methods same as Pandas."""

    def __init__(self, series):
        self._series = series

        # Validates series type
        if not (
            isinstance(series, BodoSeries)
            and isinstance(series.dtype, pd.ArrowDtype)
            and series.dtype.type is str
        ):
            raise AttributeError("Can only use .str accessor with string values!")

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
            warnings.warn(BodoLibFallbackWarning(msg))
            return object.__getattribute__(pd.Series(self._series).str, name)


class BodoDatetimeProperties:
    """Support Series.dt datetime accessors same as Pandas."""

    # TODO [BSE-4854]: support datetime methods

    def __init__(self, series):
        self._series = series
        # Validates series type
        if not (
            isinstance(series, BodoSeries)
            and series.dtype
            in (pd.ArrowDtype(pa.timestamp("ns")), pd.ArrowDtype(pa.date64())),
            pd.ArrowDtype(pa.time64("ns")),
        ):
            raise AttributeError("Can only use .dt accessor with datetimelike values")

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
            warnings.warn(BodoLibFallbackWarning(msg))
            return object.__getattribute__(pd.Series(self._series).dt, name)


def _get_series_python_func_plan(series_proj, empty_data, func_name, args, kwargs):
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
    expr = LazyPlan(
        "PythonScalarFuncExpression",
        empty_data,
        source_data,
        (
            func_name,
            True,  # is_series
            True,  # is_method
            args,  # args
            kwargs,  # kwargs
        ),
        (col_index,) + tuple(index_cols),
    )
    # Select Index columns explicitly for output
    index_col_refs = tuple(make_col_ref_exprs(index_cols, source_data))
    return wrap_plan(
        plan=LazyPlan(
            "LogicalProjection",
            empty_data,
            source_data,
            (expr,) + index_col_refs,
        ),
    )


def gen_partition(name):
    """Generates partition and rpartition using generalized template."""

    def partition(self, sep=" ", expand=True):
        """
        Splits string into 3 elements-before the separator, the separator itself,
        and the part after the separator.
        """
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

        # Create schema for output DataFrame with 3 columns
        arrow_schema = pa.schema(
            [pa.field(f"{idx}", pa.large_string()) for idx in range(3)]
        )
        empty_data = arrow_to_empty_df(arrow_schema)
        empty_series = pd.Series([], dtype=pd.ArrowDtype(pa.large_string()))

        # Create scalar function expression for each column: extract value at index idx from each row
        def create_expr(idx):
            return LazyPlan(
                "PythonScalarFuncExpression",
                empty_series,
                series_out._plan,
                (
                    "bodo.pandas.series._str_partition_helper",
                    True,  # is_series
                    False,  # is_method
                    (idx,),  # args
                    {},  # kwargs
                ),
                (0,),
            )

        expr = tuple(create_expr(idx) for idx in range(3))

        # Creates DataFrame with 3 columns
        df_plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            series_out._plan,
            expr,
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
        if name in sig_map:
            params = [
                inspect.Parameter(param[0], param[1])
                if not param[2]
                else inspect.Parameter(param[0], param[1], default=param[2][0])
                for param in sig_map[name]
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
    "replace": [
        ("to_replace", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("value", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("regex", inspect.Parameter.KEYWORD_ONLY, (False,)),
        ("inplace", inspect.Parameter.KEYWORD_ONLY, (False,)),
    ],
    "wrap": [
        ("width", inspect.Parameter.POSITIONAL_OR_KEYWORD, ()),
        ("expand_tabs", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("replace_whitespace", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("drop_whitespace", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("break_long_words", inspect.Parameter.KEYWORD_ONLY, (True,)),
        ("break_on_hyphens", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
    "normalize": [],
    "strftime": [
        ("date_format", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
    ],
    "month_name": [
        ("locale", inspect.Parameter.KEYWORD_ONLY, (None,)),
    ],
    "day_name": [
        ("locale", inspect.Parameter.KEYWORD_ONLY, (None,)),
    ],
    "floor": [
        ("freq", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("normalize", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
    "ceil": [
        ("freq", inspect.Parameter.POSITIONAL_OR_KEYWORD, (None,)),
        ("normalize", inspect.Parameter.KEYWORD_ONLY, (True,)),
    ],
}


def gen_method(name, return_type, is_method=True, accessor_type=""):
    """Generates Series methods, supports optional/positional args."""

    def method(self, *args, **kwargs):
        """Generalized template for Series methods and argument validation using signature"""
        if is_method:
            sig_bind(name, accessor_type, *args, **kwargs)  # Argument validation

        series = self._series if accessor_type else self
        dtype = self.dtype if not return_type else return_type

        index = series.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            name=series.name,
            index=index,
        )

        return _get_series_python_func_plan(
            series._plan, new_metadata, accessor_type + name, args, kwargs
        )

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
            "days_in_month",
            "quarter",
            "daysinmonth",
            "days_in_month",
            "quarter",
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

# Maps direct Series methods to return types
dir_methods = [
    # idx = 0: Series(Boolean)
    (
        [
            "isin",
            "notnull",
            "isnull",
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

# Maps Series.dt methods to return types
dt_methods = [
    # idx = 0: Series(Timestamp)
    (
        [
            "normalize",
            "floor",
            "ceil",
            # TODO: implement end_time
        ],
        pd.ArrowDtype(pa.timestamp("ns")),
    ),
    # idx = 1: Series(Float)
    (
        [
            # TODO: implement total_seconds (+support timedelta)
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
