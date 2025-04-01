import numba
from llvmlite import ir as lir
from numba.extending import intrinsic

from bodo.libs.array import cpp_table_to_py_table, delete_table, table_type
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager


def get_data_manager_pandas() -> str:
    """Get the value of mode.data_manager from pandas config.

    Returns:
        str: The value of the mode.data_manager option or 'block'
    """
    try:
        from pandas._config.config import _get_option

        return _get_option("mode.data_manager", silent=True)
    except ImportError:
        # _get_option and mode.data_manager are not supported in Pandas > 2.2.
        return "block"


def get_lazy_manager_class() -> type[LazyArrayManager | LazyBlockManager]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for DataFrame."""
    data_manager = get_data_manager_pandas()
    if data_manager == "block":
        return LazyBlockManager
    elif data_manager == "array":
        return LazyArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )


def get_lazy_single_manager_class() -> type[
    LazySingleArrayManager | LazySingleBlockManager
]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for Series."""
    data_manager = get_data_manager_pandas()
    if data_manager == "block":
        return LazySingleBlockManager
    elif data_manager == "array":
        return LazySingleArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )


@intrinsic
def cast_int64_to_table_ptr(typingctx, val):
    """Cast int64 value to C++ table pointer"""

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], lir.IntType(8).as_pointer())

    return table_type(numba.core.types.int64), codegen


@numba.njit
def cpp_table_to_py_table(in_table, out_cols_arr, out_table_type):
    """Convert a C++ table pointer to a Python table.
    Args:
        in_table (int64): C++ table pointer
        out_cols_arr (array(int64)): Array of column indices to be extracted
        out_table_type (types.Type): Type of the output table
    """
    cpp_table = cast_int64_to_table_ptr(in_table)
    out_table = cpp_table_to_py_table(cpp_table, out_cols_arr, out_table_type, 0)
    delete_table(cpp_table)
    return out_table
