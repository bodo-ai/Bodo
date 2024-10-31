import pandas as pd

from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.managers import LazyBlockManager


class BodoDataFrame(pd.DataFrame):
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
