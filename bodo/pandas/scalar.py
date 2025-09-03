import typing as pt
from collections.abc import Callable

from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.series import BodoSeries


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

    def is_lazy_plan(self):
        return self.wrapped_series.is_lazy_plan()

    def get_value(self):
        import bodo.spawn.spawner

        self.wrapped_series.execute_plan()
        assert len(self.wrapped_series) == bodo.spawn.spawner.get_num_workers()
        assert self.wrapped_series.nunique() == 1
        return self.wrapped_series[0]

    def __getattribute__(self, name):
        # Delegate attribute access to the underlying scalar value
        #
        if name in {
            "wrapped_series",
            "_lazy",
            "_exec_state",
            "get_value",
            "_get_result_id",
        }:
            return object.__getattribute__(self, name)
        scalar = self.get_value()
        return getattr(scalar, name)

    def _make_delegator(name):
        def delegator(self, *args, **kwargs):
            scalar = self.get_value()
            method = getattr(scalar, name)
            return method(*args, **kwargs)

        return delegator

    _dunder_methods = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
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
    ]
    # TODO: Support lazy operations if other is also a BodoScalar
    for _method_name in _dunder_methods:
        if _method_name not in locals():
            locals()[_method_name] = _make_delegator(_method_name)

    del _make_delegator, _dunder_methods
