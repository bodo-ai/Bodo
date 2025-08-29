import typing as pt

# Import JIT compiler lazily
import bodo.decorators  # isort:skip # noqa
import numba
from numba.core.ccallback import CFunc

from bodo.hiframes.pd_dataframe_ext import init_dataframe
from bodo.hiframes.pd_index_ext import init_range_index
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import init_series
from bodo.libs.array import (
    arr_info_list_to_table,
    array_from_cpp_table,
    array_to_info,
    cpp_table_to_py_table,
    delete_table,
    table_type,
)
from bodo.utils.conversion import coerce_to_array, index_from_array
from bodo.utils.typing import unwrap_typeref


@numba.njit
def series_to_cpp_table_jit(series_type):  # pragma: no cover
    """Convert a Series to a cpp table (table_info)"""
    out_arr = coerce_to_array(series_type, use_nullable_array=True)
    out_info = array_to_info(out_arr)
    out_cpp_table = arr_info_list_to_table([out_info])
    return out_cpp_table


@numba.njit
def cpp_table_to_series_jit(in_cpp_table, series_arr_type):  # pragma: no cover
    """Convert a cpp table (table info) to a series using JIT"""
    series_data = array_from_cpp_table(in_cpp_table, 0, series_arr_type)
    # TODO: Add option to also convert index
    index = init_range_index(0, len(series_data), 1, None)
    out_series = init_series(series_data, index)
    delete_table(in_cpp_table)
    return out_series


def extract_cpp_index(cpp_table, n_cols, index_type, length):
    pass


@numba.extending.overload(extract_cpp_index)
def overload_extract_cpp_index(cpp_table, n_cols, index_type, length):
    """Helper for cpp_table_to_df_jit to extract *index_type* index from cpp_table
    (table info) assuming that the index arrays begin after n_cols in the cpp_table
    """

    index_type = unwrap_typeref(index_type)

    if isinstance(index_type, bodo.types.RangeIndexType):

        def impl(cpp_table, n_cols, index_type, length):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_range_index(0, length, 1, None)
    elif isinstance(index_type, MultiIndexType):
        n_levels = len(index_type.array_types)
        index_arrays = ",".join(
            f"array_from_cpp_table(cpp_table, n_cols + {i}, arr_types[{i}])"
            for i in range(n_levels)
        )
        names = ",".join(f"names[{i}]" for i in range(n_levels))

        func_text = "def impl(cpp_table, n_cols, index_type, length):\n"
        func_text += f"  return init_multi_index(({index_arrays},), ({names},), None)"

        locals = {}
        globals = {
            "arr_types": index_type.array_types,
            "names": tuple(name.literal_value for name in index_type.names_typ),
            "init_multi_index": bodo.hiframes.pd_multi_index_ext.init_multi_index,
            "array_from_cpp_table": array_from_cpp_table,
        }
        exec(func_text, globals, locals)
        return locals["impl"]
    else:
        arr_type = index_type.data

        def impl(cpp_table, n_cols, index_type, length):  # pragma: no cover
            index_arr = array_from_cpp_table(cpp_table, n_cols, arr_type)
            return index_from_array(index_arr)

    return impl


@numba.njit
def cpp_table_to_df_jit(
    cpp_table, out_cols_arr, column_names, py_table_type, index_type
):  # pragma: no cover
    """Convert a cpp table to a DataFrame using JIT.

    Args:
        cpp_table (table_info): Input table to convert to a DataFrame
        out_cols_arr (Array): Array of indices from cpp_table to select as columns.
        column_names (ColNamesMetaType): The names of the columns.
        py_table_type (TableType): Type of the columns of the dataframe.
        index_type (IndexType): The type of the index.

    Returns:
        DataFrameType
    """
    py_table = cpp_table_to_py_table(cpp_table, out_cols_arr, py_table_type, 0)
    index = extract_cpp_index(cpp_table, len(out_cols_arr), index_type, len(py_table))
    delete_table(cpp_table)
    return init_dataframe((py_table,), index, column_names)


def get_udf_cfunc_decorator() -> pt.Callable[[pt.Callable], CFunc]:
    """Decorator for creating C callbacks for map/apply that take in a table info and
    return a table info."""
    from bodo.decorators import _cfunc

    return _cfunc(table_type(table_type), cache=True)


def compile_cfunc(func, decorator):
    """Util for to compiling a cfunc and getting a pointer
    to the C callback (called once on each worker per cfunc).
    """
    import ctypes

    cfunc = decorator(func)
    return ctypes.c_void_p(cfunc.address).value
