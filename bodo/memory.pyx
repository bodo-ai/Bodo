
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True

import atexit

from pyarrow.lib cimport CMemoryPool, MemoryPool
import pyarrow as pa

import bodo.memory_cpp


def set_default_buffer_pool_as_arrow_memory_pool():
    """
    Helper function to set our default BufferPool instance
    as the default MemoryPool for all PyArrow allocations.
    """
    py_ptr: int = bodo.memory_cpp.default_buffer_pool_ptr()
    # To set Arrow's default memory pool, we:
    # - Start with Python long with value of the raw pointer
    #   from the C++ BufferPool instance
    # - Cast it to a C int64_t (otherwise Cython might do some hidden casting)
    # - Cast it to a C void* (to make it a pointer)
    # - Cast it to a CMemoryPool* (which our BufferPool inherits from)
    # - Follow the same steps Arrow does to initialize a MemoryPool
    cdef:
        long long cptr = <long long>py_ptr
        void* ptr = <void*>cptr
        CMemoryPool* pool_ptr = <CMemoryPool*>ptr
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    pool.init(pool_ptr)
    pa.set_memory_pool(pool)


set_default_buffer_pool_as_arrow_memory_pool()

default_buffer_pool_smallest_size_class = bodo.memory_cpp.default_buffer_pool_smallest_size_class
default_buffer_pool_bytes_allocated = bodo.memory_cpp.default_buffer_pool_bytes_allocated
default_buffer_pool_bytes_pinned = bodo.memory_cpp.default_buffer_pool_bytes_pinned

atexit.register(bodo.memory_cpp.default_buffer_pool_cleanup)
