import llvmlite.binding as ll
import numba
from numba.core import types

from bodo.libs import array_ext

ll.add_symbol("get_stats_alloc_arr", array_ext.get_stats_alloc)
ll.add_symbol("get_stats_free_arr", array_ext.get_stats_free)
ll.add_symbol("get_stats_mi_alloc_arr", array_ext.get_stats_mi_alloc)
ll.add_symbol("get_stats_mi_free_arr", array_ext.get_stats_mi_free)


get_stats_alloc_arr = types.ExternalFunction(
    "get_stats_alloc_arr",
    types.uint64(),
)

get_stats_free_arr = types.ExternalFunction(
    "get_stats_free_arr",
    types.uint64(),
)

get_stats_mi_alloc_arr = types.ExternalFunction(
    "get_stats_mi_alloc_arr",
    types.uint64(),
)

get_stats_mi_free_arr = types.ExternalFunction(
    "get_stats_mi_free_arr",
    types.uint64(),
)


@numba.njit
def get_allocation_stats():  # pragma: no cover
    """
    Get allocation stats for arrays allocated in Bodo's C++ runtime.
    All C extensions share the same MemSys object, so we only need to check one of the extensions
    """
    return get_allocation_stats_arr()


@numba.njit
def get_allocation_stats_arr():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ array runtime"""
    return (
        get_stats_alloc_arr(),
        get_stats_free_arr(),
        get_stats_mi_alloc_arr(),
        get_stats_mi_free_arr(),
    )
