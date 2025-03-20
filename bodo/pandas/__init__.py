from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.arrow.array import LazyArrowExtensionArray
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.utils import get_lazy_manager_class, get_lazy_single_manager_class
from bodo.pandas import plan_operators
import os

DataFrame = BodoDataFrame
Series = BodoSeries

BODO_PANDAS_FALLBACK = int(os.environ.get("BODO_PANDAS_FALLBACK", 0))

no_default=...

def read_parquet(path,
                 engine='auto',
                 columns=None,
                 storage_options=None,
                 use_nullable_dtypes=no_default,
                 dtype_backend=no_default,
                 filesystem=None,
                 filters=None,
                 **kwargs):
    if (engine != 'auto' or
        columns != None or
        storage_options != None or
        use_nullable_dtypes != no_default or
        dtype_backend != no_default or
        filesystem != None or
        filters != None or
        len(kwargs) > 0):
        assert False and "Unsupported option to read_parquet"
    pr = plan_operators.ParquetRead(path)
    return DataFrame(pr.schema, plan=pr)

def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)

def add_fallback():
    if BODO_PANDAS_FALLBACK != 0:
        import pandas
        import inspect
        import sys

        current_module = sys.modules[__name__]

        pandas_attrs = dir(pandas)
        bodo_df_lib_attrs = dir(current_module)

        pandas_funcs = [attr for attr in pandas_attrs if inspect.isfunction(getattr(pandas, attr))]
        pandas_nonfuncs = [attr for attr in pandas_attrs if not inspect.isfunction(getattr(pandas, attr))]
        pandas_nonfuncs_types = [(attr, type(getattr(pandas, attr))) for attr in pandas_attrs if not inspect.isfunction(getattr(pandas, attr))]

        """
        print("funcs:", pandas_funcs)
        print("------------------------")
        print("non-funcs:", pandas_nonfuncs_types)
        print("------------------------")
        """

        bodo_df_lib_funcs = [attr for attr in bodo_df_lib_attrs if inspect.isfunction(getattr(current_module, attr))]
        bodo_df_lib_nonfuncs = [attr for attr in bodo_df_lib_attrs if not inspect.isfunction(getattr(current_module, attr))]

        non_overloaded_funcs = set(pandas_funcs).difference(bodo_df_lib_funcs)
        non_overloaded_nonfuncs = set(pandas_nonfuncs).difference(bodo_df_lib_nonfuncs)

        """
        print("non_overloaded_funcs:", non_overloaded_funcs)
        print("------------------------")
        print("non_overloaded_non-funcs:", non_overloaded_nonfuncs)
        print("------------------------")
        print("overloaded_funcs:", overloaded_funcs)
        print("------------------------")
        print("overloaded_non-funcs:", overloaded_nonfuncs)
        print("------------------------")
        """

        for func in non_overloaded_funcs:
            setattr(current_module, func, getattr(pandas, func))
        for nonfunc in non_overloaded_nonfuncs:
            setattr(current_module, nonfunc, getattr(pandas, nonfunc))

# Must do this at the end so that all functions we want to provide already exist.
add_fallback()
