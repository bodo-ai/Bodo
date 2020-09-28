from collections import namedtuple

import llvmlite.binding as ll
import numba
from numba.core import types

from bodo.io import parquet_cpp
from bodo.libs import array_ext, decimal_ext, quantile_alg

ll.add_symbol("get_stats_alloc_arr", array_ext.get_stats_alloc)
ll.add_symbol("get_stats_free_arr", array_ext.get_stats_free)
ll.add_symbol("get_stats_mi_alloc_arr", array_ext.get_stats_mi_alloc)
ll.add_symbol("get_stats_mi_free_arr", array_ext.get_stats_mi_free)
ll.add_symbol("get_stats_alloc_dec", decimal_ext.get_stats_alloc)
ll.add_symbol("get_stats_free_dec", decimal_ext.get_stats_free)
ll.add_symbol("get_stats_mi_alloc_dec", decimal_ext.get_stats_mi_alloc)
ll.add_symbol("get_stats_mi_free_dec", decimal_ext.get_stats_mi_free)
ll.add_symbol("get_stats_alloc_qa", quantile_alg.get_stats_alloc)
ll.add_symbol("get_stats_free_qa", quantile_alg.get_stats_free)
ll.add_symbol("get_stats_alloc_pq", parquet_cpp.get_stats_alloc)
ll.add_symbol("get_stats_free_pq", parquet_cpp.get_stats_free)
ll.add_symbol("get_stats_mi_alloc_pq", parquet_cpp.get_stats_mi_alloc)
ll.add_symbol("get_stats_mi_free_pq", parquet_cpp.get_stats_mi_free)
ll.add_symbol("get_stats_mi_alloc_qa", quantile_alg.get_stats_mi_alloc)
ll.add_symbol("get_stats_mi_free_qa", quantile_alg.get_stats_mi_free)

Mstats = namedtuple("Mstats", ["alloc", "free", "mi_alloc", "mi_free"])


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

get_stats_alloc_dec = types.ExternalFunction(
    "get_stats_alloc_dec",
    types.uint64(),
)

get_stats_free_dec = types.ExternalFunction(
    "get_stats_free_dec",
    types.uint64(),
)

get_stats_mi_alloc_dec = types.ExternalFunction(
    "get_stats_mi_alloc_dec",
    types.uint64(),
)

get_stats_mi_free_dec = types.ExternalFunction(
    "get_stats_mi_free_dec",
    types.uint64(),
)

get_stats_alloc_pq = types.ExternalFunction(
    "get_stats_alloc_pq",
    types.uint64(),
)

get_stats_free_pq = types.ExternalFunction(
    "get_stats_free_pq",
    types.uint64(),
)

get_stats_mi_alloc_pq = types.ExternalFunction(
    "get_stats_mi_alloc_pq",
    types.uint64(),
)

get_stats_mi_free_pq = types.ExternalFunction(
    "get_stats_mi_free_pq",
    types.uint64(),
)


get_stats_alloc_qa = types.ExternalFunction(
    "get_stats_alloc_qa",
    types.uint64(),
)

get_stats_free_qa = types.ExternalFunction(
    "get_stats_free_qa",
    types.uint64(),
)

get_stats_mi_alloc_qa = types.ExternalFunction(
    "get_stats_mi_alloc_qa",
    types.uint64(),
)

get_stats_mi_free_qa = types.ExternalFunction(
    "get_stats_mi_free_qa",
    types.uint64(),
)


@numba.njit
def get_allocation_stats():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ runtimes"""
    stats = [
        get_allocation_stats_arr(),
        get_allocation_stats_dec(),
        get_allocation_stats_pq(),
        get_allocation_stats_qa(),
    ]
    allocs, frees, mi_allocs, mi_frees = 0, 0, 0, 0
    for stat in stats:
        allocs += stat.alloc
        frees += stat.free
        mi_allocs += stat.mi_alloc
        mi_frees += stat.mi_free
    return Mstats(allocs, frees, mi_allocs, mi_frees)


@numba.njit
def get_allocation_stats_arr():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ array runtime"""
    return Mstats(
        get_stats_alloc_arr(),
        get_stats_free_arr(),
        get_stats_mi_alloc_arr(),
        get_stats_mi_free_arr(),
    )


@numba.njit
def get_allocation_stats_dec():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ decimal runtime"""
    return Mstats(
        get_stats_alloc_dec(),
        get_stats_free_dec(),
        get_stats_mi_alloc_dec(),
        get_stats_mi_free_dec(),
    )


@numba.njit
def get_allocation_stats_pq():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ parquet runtime"""
    return Mstats(
        get_stats_alloc_pq(),
        get_stats_free_pq(),
        get_stats_mi_alloc_pq(),
        get_stats_mi_free_pq(),
    )


@numba.njit
def get_allocation_stats_qa():  # pragma: no cover
    """get allocation stats for arrays allocated in Bodo's C++ qa runtime"""
    return Mstats(
        get_stats_alloc_qa(),
        get_stats_free_qa(),
        get_stats_mi_alloc_qa(),
        get_stats_mi_free_qa(),
    )
