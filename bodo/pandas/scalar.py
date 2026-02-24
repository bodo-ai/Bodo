from __future__ import annotations

import typing as pt
from collections.abc import Callable

import pandas as pd

from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import scalarOutputNACheck


class BodoScalar(BodoLazyWrapper):
    wrapped_series: BodoSeries

    def __init__(self, wrapped_series: BodoSeries):
        self.wrapped_series = wrapped_series

    def _get_result_id(self) -> str | None:
        return self.wrapped_series._get_result_id()

    @classmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
    ) -> BodoLazyWrapper:
        return cls(BodoSeries.from_lazy_metadata(lazy_metadata, collect_func, del_func))

    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        self.wrapped_series.update_from_lazy_metadata(lazy_metadata)

    def execute_plan(self):
        return self.wrapped_series.execute_plan()

    @property
    def _lazy(self) -> bool:
        return self._get_result_id() is not None

    @property
    def _plan(self):
        return self.wrapped_series._plan

    def is_lazy_plan(self):
        return self.wrapped_series.is_lazy_plan()

    def get_value(self):
        import warnings

        from bodo.pandas.utils import BodoLibFallbackWarning

        prev_lazy_plan = self.is_lazy_plan()
        self.wrapped_series.execute_plan()
        if prev_lazy_plan:
            # If we were lazy before we need to confirm
            # that we have exactly one unique value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
                assert self.wrapped_series.nunique() in {0, 1}
                # Avoid getitem warning
                out = self.wrapped_series[0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
                out = self.wrapped_series[0]

        return scalarOutputNACheck(out, self.wrapped_series.dtype)

    @property
    def __pandas_priority__(self):
        """
        Override this so we don't call get_value during planning.'
        This is used by pandas during comparisons and arithmetic operations.'
        """
        return None

    def __getattribute__(self, name):
        # Delegate attribute access to the underlying scalar value
        #
        if name in {
            "wrapped_series",
            "_plan",
            "_lazy",
            "_exec_state",
            "get_value",
            "_get_result_id",
            "is_lazy_plan",
            "execute_plan",
            "update_from_lazy_metadata",
            "from_lazy_metadata",
            "__array__",
            "__pandas_priority__",
        }:
            return object.__getattribute__(self, name)
        scalar = self.get_value()
        return getattr(scalar, name)

    def __array__(self, dtype=None):
        import numpy as np

        scalar = self.get_value()
        return np.array(scalar, dtype=dtype)

    def _make_delegator(name):
        def delegator(self, *args, **kwargs):
            scalar = self.get_value()
            method = getattr(scalar, name)
            return method(*args, **kwargs)

        return delegator

    def _make_series_delegator(name):
        """Support binary operations using BodoSeries implementations."""

        def delegator(self, other):
            nonlocal name
            # Use direct BodoSeries implementation if other is BodoSeries since it
            # Handles BodoScalar using cross join properly. BodoSeries/BodoSeries
            # doesn't have cross join support yet.
            if type(other) is BodoSeries:
                name = _get_reversed_dunder(name)
                return getattr(other, name)(self)

            if pd.api.types.is_scalar(other):
                series = self.wrapped_series
                method = getattr(series, name)
                out = method(other)
                return BodoScalar(out)

            # Fallback if not supported (e.g. df.A.sum() + np.ones(3))
            scalar = self.get_value()
            method = getattr(scalar, name)
            return method(other)

        return delegator

    _dunder_methods = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
        "__pow__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__str__",
        "__repr__",
        "__int__",
        "__float__",
        "__bool__",
        "__hash__",
        "__bytes__",
        "__format__",
        "__dir__",
        "__sizeof__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
        "__index__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__and__",
        "__or__",
        "__xor__",
        "__rand__",
        "__ror__",
        "__rxor__",
        "__lshift__",
        "__rshift__",
        "__rlshift__",
        "__complex__",
        "__hash__",
        "__bool__",
        "__len__",
        "__contains__",
        "_is_na",
        "__class__",
    ]
    _series_dunder_methods = [
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
    ]
    # TODO: Support lazy operations if other is also a BodoScalar
    for _method_name in _dunder_methods:
        if _method_name not in locals():
            if _method_name in _series_dunder_methods:
                locals()[_method_name] = _make_series_delegator(_method_name)
            else:
                locals()[_method_name] = _make_delegator(_method_name)

    del _make_delegator, _dunder_methods, _make_series_delegator, _series_dunder_methods


def _get_reversed_dunder(name: str) -> str:
    if name.startswith("__r"):
        return "__" + name[3:]
    if name.startswith("__"):
        return "__r" + name[2:]
    raise ValueError(f"Not a dunder method: {name}")
