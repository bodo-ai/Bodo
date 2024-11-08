import abc
import typing as pt
from collections.abc import Callable

from bodo.pandas.lazy_metadata import LazyMetadata


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
    ) -> "BodoLazyWrapper":
        return cls()

    @abc.abstractmethod
    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        pass

    @property
    def _lazy(self) -> bool:
        return self._get_result_id() is not None
