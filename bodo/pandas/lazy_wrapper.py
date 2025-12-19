from __future__ import annotations

import abc
import typing as pt
from collections.abc import Callable
from enum import Enum

from bodo.pandas.lazy_metadata import LazyMetadata


class ExecState(Enum):
    PLAN = 0
    DISTRIBUTED = 1
    COLLECTED = 2


class BodoLazyWrapper(abc.ABC):
    @abc.abstractmethod
    def _get_result_id(self) -> str | None:
        pass

    @classmethod
    @abc.abstractmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
    ) -> BodoLazyWrapper:
        return cls()

    @abc.abstractmethod
    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        pass

    @abc.abstractmethod
    def is_lazy_plan(self):
        pass

    @abc.abstractmethod
    def execute_plan(self):
        pass

    @property
    def _lazy(self) -> bool:
        return self._get_result_id() is not None

    @abc.abstractmethod
    def is_lazy_plan(self):
        pass

    @property
    def _exec_state(self) -> ExecState:
        if self.is_lazy_plan():
            return ExecState.PLAN
        else:
            if self._lazy:
                return ExecState.DISTRIBUTED
            else:
                return ExecState.COLLECTED
