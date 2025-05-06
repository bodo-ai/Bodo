import typing as pt
from collections.abc import Callable, Hashable

import pandas as pd
import pyarrow as pa

from bodo.ext import plan_optimizer
from bodo.pandas.array_manager import LazySingleArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper, ExecState
from bodo.pandas.managers import LazyMetadataMixin, LazySingleBlockManager
from bodo.pandas.utils import (
    LazyPlan,
    check_args_fallback,
    get_lazy_single_manager_class,
    get_n_index_arrays,
    get_proj_expr_single,
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
            if self._mgr._plan is not None:
                return self._mgr._plan
            else:
                return plan_optimizer.LogicalGetSeriesRead(self._mgr._md_result_id)

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
        new_metadata = zero_size_self._cmp_method(zero_size_other, op)
        assert isinstance(new_metadata, pd.Series)

        # Extract argument expressions
        lhs = get_proj_expr_single(self._plan)
        rhs = get_proj_expr_single(other) if isinstance(other, LazyPlan) else other
        expr = LazyPlan("BinaryOpExpression", lhs, rhs, op)
        expr.out_schema = new_metadata.to_frame()

        plan = LazyPlan(
            "LogicalProjection",
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
        )
        return wrap_plan(new_metadata, plan=plan)

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
            raise TypeError(
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
        new_metadata = getattr(zero_size_self, op)(zero_size_other)
        assert isinstance(new_metadata, pd.Series), (
            "_conjunction_binop: new_metadata is not a Series"
        )

        # Extract argument expressions
        lhs = get_proj_expr_single(self._plan)
        rhs = get_proj_expr_single(other) if isinstance(other, LazyPlan) else other
        expr = LazyPlan("ConjunctionOpExpression", lhs, rhs, op)
        expr.out_schema = new_metadata.to_frame()

        plan = LazyPlan(
            "LogicalProjection",
            # Use the original table without the Series projection node.
            self._plan.args[0],
            (expr,),
        )
        return wrap_plan(new_metadata, plan=plan)

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
        new_metadata = _empty_like(self)

        assert isinstance(new_metadata, pd.Series)
        return wrap_plan(
            new_metadata,
            plan=LazyPlan("LogicalUnaryOp", self._plan, "__invert__"),
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
        self.execute_plan()

        if isinstance(self._mgr, LazyMetadataMixin) and (
            self._mgr._md_nrows is not None
        ):
            return (self._mgr._md_nrows,)
        return super().shape

    def head(self, n: int = 5):
        """
        Get the first n rows of the series. If head_s is present and n < len(head_s) we call head on head_s.
        Otherwise we use the data fetched from the workers.
        """
        if n == 0 and self._head_s is not None:
            return pd.Series(
                index=self._head_s.index,
                name=self._head_s.name,
                dtype=self._head_s.dtype,
            )

        if (self._head_s is None) or (n > self._head_s.shape[0]):
            if self._exec_state == ExecState.PLAN:
                from bodo.pandas.base import _empty_like

                new_metadata = _empty_like(self)
                planLimit = LazyPlan(
                    "LogicalLimit",
                    self._plan,
                    n,
                )

                return wrap_plan(new_metadata, planLimit)
            else:
                return super().head(n)
        else:
            # If head_s is available and larger than n, then use it directly.
            return self._head_s.head(n)

    def __len__(self):
        self.execute_plan()
        if self._lazy:
            return self._mgr._md_nrows
        return super().__len__()

    def _get_result_id(self) -> str | None:
        if isinstance(self._mgr, LazyMetadataMixin):
            return self._mgr._md_result_id
        return None

    @property
    def str(self):
        return StringMethods(self)

    @check_args_fallback(supported=["arg"])
    def map(self, arg, na_action=None):
        """
        Apply function to elements in a Series
        """
        from bodo.pandas.utils import arrow_to_empty_df

        # Get output data type by running the UDF on a sample of the data.
        # Saving the plan to avoid hitting LogicalGetDataframeRead gaps with head().
        # TODO: remove when LIMIT plan is properly supported for head().
        series_sample = self.head(1).execute_plan()
        pd_sample = pd.Series(series_sample)
        out_sample = pd_sample.map(arg)

        assert isinstance(out_sample, pd.Series), (
            f"BodoSeries.map(), expected output to be Series, got: {type(out_sample)}."
        )
        out_sample_df = out_sample.to_frame()
        empty_df = arrow_to_empty_df(pa.Schema.from_pandas(out_sample_df))

        # convert back to Series
        empty_series = empty_df.squeeze()
        empty_series.name = out_sample.name

        return _get_series_python_func_plan(self._plan, empty_series, "map", (arg,), {})


class StringMethods:
    """Support Series.str string processing methods same as Pandas."""

    def __init__(self, series):
        self._series = series

    def lower(self):
        index = self._series.head(0).index
        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(pa.large_string()),
            name=self._series.name,
            index=index,
        )
        return _get_series_python_func_plan(
            self._series._plan, new_metadata, "str.lower", (), {}
        )

    @check_args_fallback(supported=[])
    def strip(self, to_strip=None):
        index = self._series.head(0).index
        new_metadata = pd.Series(
            dtype=pd.ArrowDtype(pa.large_string()),
            name=self._series.name,
            index=index,
        )
        return _get_series_python_func_plan(
            self._series._plan, new_metadata, "str.strip", (), {}
        )


def _get_series_python_func_plan(series_proj, new_metadata, func_name, args, kwargs):
    """Create a plan for calling a Series method in Python. Creates a proper
    PythonScalarFuncExpression with the correct arguments and a LogicalProjection.
    """
    assert series_proj.plan_class == "LogicalProjection", "projection expected"
    input_expr = series_proj.args[1][0]
    assert input_expr.plan_class == "ColRefExpression", "Expected ColRefExpression"
    col_index = input_expr.args[1]
    source_data = series_proj.args[0]
    n_cols = len(source_data.out_schema.columns)
    index_cols = range(
        n_cols, n_cols + get_n_index_arrays(source_data.out_schema.index)
    )
    expr = LazyPlan(
        "PythonScalarFuncExpression",
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
    expr.out_schema = new_metadata.to_frame()
    # Select Index columns explicitly for output
    index_col_refs = tuple(make_col_ref_exprs(index_cols, source_data))
    return wrap_plan(
        new_metadata,
        plan=LazyPlan(
            "LogicalProjection",
            source_data,
            (expr,) + index_col_refs,
        ),
    )
