import typing as pt
from collections.abc import Callable

import pandas as pd

from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.managers import LazyBlockManager, LazyMetadataMixin
from bodo.pandas.utils import get_lazy_manager_class


class BodoDataFrame(pd.DataFrame, BodoLazyWrapper):
    # We need to store the head_df to avoid data pull when head is called.
    # Since BlockManagers are in Cython it's tricky to override all methods
    # so some methods like head will still trigger data pull if we don't store head_df and
    # use it directly when available.
    _head_df: pd.DataFrame | None = None

    @staticmethod
    def from_lazy_mgr(
        lazy_mgr: LazyArrayManager | LazyBlockManager,
        head_df: pd.DataFrame | None,
    ):
        """
        Create a BodoDataFrame from a lazy manager and possibly a head_df.
        If you want to create a BodoDataFrame from a pandas manager use _from_mgr
        """
        df = BodoDataFrame._from_mgr(lazy_mgr, [])
        df._head_df = head_df
        return df

    @classmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
    ) -> "BodoDataFrame":
        """
        Create a BodoDataFrame from a lazy metadata object.
        """
        assert isinstance(lazy_metadata.head, pd.DataFrame)
        lazy_mgr = get_lazy_manager_class()(
            None,
            None,
            result_id=lazy_metadata.result_id,
            nrows=lazy_metadata.nrows,
            head=lazy_metadata.head._mgr,
            collect_func=collect_func,
            del_func=del_func,
            index_data=lazy_metadata.index_data,
        )
        return cls.from_lazy_mgr(lazy_mgr, lazy_metadata.head)

    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        """
        Update the dataframe with new metadata.
        """
        assert self._lazy
        assert isinstance(lazy_metadata.head, pd.DataFrame)
        # Call delfunc to delete the old data.
        self._mgr._del_func(self._mgr._md_result_id)
        self._head_df = lazy_metadata.head
        self._mgr._md_nrows = lazy_metadata.nrows
        self._mgr._md_result_id = lazy_metadata.result_id
        self._mgr._md_head = lazy_metadata.head._mgr

    def head(self, n: int = 5):
        """
        Return the first n rows. If head_df is available and larger than n, then use it directly.
        Otherwise, use the default head method which will trigger a data pull.
        """
        if (self._head_df is None) or (n > self._head_df.shape[0]):
            return super().head(n)
        else:
            # If head_df is available and larger than n, then use it directly.
            return self._head_df.head(n)

    def to_parquet(self, *args, **kwargs):
        # TODO: Implement this BSE-4100
        print("Asking workers to write to parquet...")
        print("args: ", *args)
        print("kwargs: ", **kwargs)

        ## Dynamic codegen implementation would look something like this:
        # @submit_jit
        # def to_parquet_wrapper(df: pd.DataFrame, path):
        #     df.to_parquet(path)

        # to_parquet_wrapper(self, args[0])

    def _get_result_id(self) -> str | None:
        if isinstance(self._mgr, LazyMetadataMixin):
            return self._mgr._md_result_id
        return None
