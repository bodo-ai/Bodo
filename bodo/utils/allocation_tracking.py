from bodo.libs import array_ext


def get_allocation_stats():
    """
    Get allocation stats for arrays allocated in Bodo's C++ runtime.
    All C extensions share the same MemSys object, so we only need to check one of the extensions
    """
    return (
        array_ext.get_stats_alloc_py_wrapper(),
        array_ext.get_stats_free_py_wrapper(),
        array_ext.get_stats_mi_alloc_py_wrapper(),
        array_ext.get_stats_mi_free_py_wrapper(),
    )
