from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.arrow.array import LazyArrowExtensionArray
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.base import *
from bodo.pandas.utils import fallback_wrapper
import os

# Non-performance critical scalars
from pandas import Timestamp

DataFrame = BodoDataFrame
Series = BodoSeries

# If not present or 0 then allow Python to give a symbol not found error
# if they try to use a Pandas feature that we haven't explicitly implemented.
# If present and non-zero then for functions that we haven't explicitly
# implemented it will try to call the Pandas version which will
# trigger any bodo dataframe/series arguments to be automatically converted
# to Pandas.
BODO_PANDAS_FALLBACK = int(os.environ.get("BODO_PANDAS_FALLBACK", 1))

def add_fallback():
    if BODO_PANDAS_FALLBACK != 0:
        import pandas
        import inspect
        import sys

        current_module = sys.modules[__name__]

        # Get all the functions and everything else accessible at the top-level
        # from the Pandas module.
        pandas_attrs = dir(pandas)
        # Do the same for things implemented in Bodo via the bodo.pandas.base import.
        bodo_df_lib_attrs = dir(current_module)

        for func in set(pandas_attrs).difference(bodo_df_lib_attrs):
            # Export the pandas functions that aren't implemented by bodo
            # into bodo.pandas.
            msg = (
                f"{func} is not implemented in Bodo DataFrames yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            setattr(current_module, func, fallback_wrapper(pandas, getattr(pandas, func), func, msg))

# Must do this at the end so that all functions we want to provide already exist.
add_fallback()
