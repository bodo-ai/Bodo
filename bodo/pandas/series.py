from collections.abc import Hashable

import pandas as pd

from bodo.pandas.array_manager import LazySingleArrayManager
from bodo.pandas.managers import LazyMetadataMixin, LazySingleBlockManager


class BodoSeries(pd.Series):
    # We need to store the head_s to avoid data pull when head is called.
    # Since BlockManagers are in Cython it's tricky to override all methods
    # so some methods like head will still trigger data pull if we don't store head_s and
    # use it directly when available.
    _head_s: pd.Series | None = None
    _name: Hashable = None

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
        series._name = None
        series._head_s = head_s
        return series

    @property
    def shape(self):
        """
        Get the shape of the series. Data is fetched from metadata if present, otherwise the data fetched from workers is used.
        """
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
        if (self._head_s is None) or (n > self._head_s.shape[0]):
            return super().head(n)
        else:
            # If head_s is available and larger than n, then use it directly.
            return self._head_s.head(n)
