import inspect
import typing as pt
import warnings
from collections.abc import Callable, Hashable

import pandas as pd
import pyarrow as pa

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
    check_args_fallback,
    get_lazy_single_manager_class,
    get_n_index_arrays,
    get_proj_expr_single,
    get_scalar_udf_result_type,
    is_single_colref_projection,
    make_col_ref_exprs,
    wrap_plan,
)


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
            "BinaryOpExpression",
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
    ) -> "BodoSeries":
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

    def isin(self, values):
        """
        Whether elements in Series are contained in values.

        """

        index = self.head(0).index
        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(pa.bool_()),
            name=self.name,
            index=index,
        )
        return _get_series_python_func_plan(
            self._plan, new_metadata, "isin", (values,), {}
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


def sig_bind(name, *args, **kwargs):
    """
    Binds args and kwargs to method's signature for argument validation.
    Single exception case is Series.str.wrap() which takes in **kwargs in place of individual
    keyword arguments. Thus, wrap_signature is manually created, to which the provided arguments are bound.
    """
    msg = ""
    try:
        if name == "wrap":
            wrap_params = [
                inspect.Parameter("width", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "expand_tabs", inspect.Parameter.KEYWORD_ONLY, default=True
                ),
                inspect.Parameter(
                    "replace_whitespace", inspect.Parameter.KEYWORD_ONLY, default=True
                ),
                inspect.Parameter(
                    "drop_whitespace", inspect.Parameter.KEYWORD_ONLY, default=True
                ),
                inspect.Parameter(
                    "break_long_words", inspect.Parameter.KEYWORD_ONLY, default=True
                ),
                inspect.Parameter(
                    "break_on_hyphens", inspect.Parameter.KEYWORD_ONLY, default=True
                ),
            ]
            wrap_signature = inspect.Signature(wrap_params)
            wrap_signature.bind(*args, **kwargs)
        else:
            sample_series = pd.Series(["a"])
            str_accessor = sample_series.str
            func = getattr(str_accessor, name)
            signature = inspect.signature(func)
            signature.bind(*args, **kwargs)
        return
    # Separated raising error from except statement to avoid nested errors
    except TypeError as e:
        msg = e
    raise TypeError(f"StringMethods.{name}() {msg}")


def gen_str_method(name, rettype):
    """Generalized generator for Series.str methods with optional/positional args."""

    def str_method(self, *args, **kwargs):
        """Generalized template for Series.str methods and argument validation using signature"""
        sig_bind(name, *args, **kwargs)  # Argument validation

        index = self._series.head(0).index
        new_metadata = pd.Series(
            dtype=rettype,
            name=self._series.name,
            index=index,
        )
        return _get_series_python_func_plan(
            self._series._plan, new_metadata, f"str.{name}", args, kwargs
        )

    return str_method


def gen_dt_accessor(name, rettype):
    """Generalized generator for Series.dt accessors"""

    def dt_accessor(self):
        """Generalized template for Series.dt accessors"""

        index = self._series.head(0).index
        new_metadata = pd.Series(
            dtype=rettype,
            name=self._series.name,
            index=index,
        )
        return _get_series_python_func_plan(
            self._series._plan, new_metadata, f"dt.{name}", (), {}
        )

    return dt_accessor


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

# Generates Series.str methods
for str_pair in series_str_methods:
    for func_name in str_pair[0]:
        func = gen_str_method(func_name, str_pair[1])
        setattr(BodoStringMethods, func_name, func)

# Generates Series.dt accessors
for dt_accessor_pair in dt_accessors:
    for accessor_name in dt_accessor_pair[0]:
        accessor = gen_dt_accessor(accessor_name, dt_accessor_pair[1])
        setattr(BodoDatetimeProperties, accessor_name, property(accessor))
