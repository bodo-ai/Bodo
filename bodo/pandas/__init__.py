from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.arrow.array import LazyArrowExtensionArray
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.base import *
import os

DataFrame = BodoDataFrame
Series = BodoSeries

# If not present or 0 then allow Python to give a symbol not found error
# if they try to use a Pandas feature that we haven't explicitly implemented.
# If present and non-zero then try to convert bodo dataframe/series to
# Pandas equivalent and run standard Pandas for operations not explicitly
# support by bodo.pandas yet.
BODO_PANDAS_FALLBACK = int(os.environ.get("BODO_PANDAS_FALLBACK", 0))

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

        # Get the top-level Pandas funcs.
        pandas_funcs = [attr for attr in pandas_attrs if inspect.isfunction(getattr(pandas, attr))]
        # Get everything else accessible at the top-level of Pandas.
        pandas_nonfuncs = [attr for attr in pandas_attrs if not inspect.isfunction(getattr(pandas, attr))]

        # Get the top-level functions support by bodo.pandas.
        bodo_df_lib_funcs = [attr for attr in bodo_df_lib_attrs if inspect.isfunction(getattr(current_module, attr))]
        # Get the top-level non-functions support by bodo.pandas.
        bodo_df_lib_nonfuncs = [attr for attr in bodo_df_lib_attrs if not inspect.isfunction(getattr(current_module, attr))]

        # Get the pandas functions that don't have an equivalent yet in bodo.
        non_overloaded_funcs = set(pandas_funcs).difference(bodo_df_lib_funcs)
        # Get the pandas non-functions that don't have an equivalent yet in bodo.
        non_overloaded_nonfuncs = set(pandas_nonfuncs).difference(bodo_df_lib_nonfuncs)

        for func in non_overloaded_funcs:
            # Export the pandas functions into bodo.pandas.
            setattr(current_module, func, getattr(pandas, func))

        for nonfunc in non_overloaded_nonfuncs:
            # Export the pandas non-functions into bodo.pandas.
            setattr(current_module, nonfunc, getattr(pandas, nonfunc))

# Must do this at the end so that all functions we want to provide already exist.
add_fallback()
