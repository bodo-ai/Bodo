# distutils: language = c++

# Imports standard Cython modules.
# Similar to PyArrow
# (https://github.com/apache/arrow/blob/apache-arrow-11.0.0/python/pyarrow/includes/common.pxd)

cimport cpython
from libc.stdint cimport *
from libcpp cimport bool as c_bool
from libcpp cimport nullptr
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.string cimport string as c_string
from libcpp.vector cimport vector
from pyarrow.includes.common cimport CStatus


cdef extern from * namespace "std" nogil:
    cdef shared_ptr[T] static_pointer_cast[T, U](shared_ptr[U])
