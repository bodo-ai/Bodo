# libcudf_hash_join.pyx
from cython.operator import dereference
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport pair, move
from libcpp.optional cimport optional
from libc.stdint cimport int32_t
from rmm.librmm.memory_resource cimport device_memory_resource
from enum import IntEnum

from pylibcudf.table cimport Table
from pylibcudf.column cimport Column
from pylibcudf.types import NullEquality
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.types cimport null_equality, size_type
from pylibcudf.libcudf.join cimport gather_map_type, gather_map_pair_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream
from rmm.librmm.device_uvector cimport device_uvector
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.librmm.device_buffer cimport device_buffer

cdef Column _column_from_gather_map(
    gather_map_type& gather_map, Stream stream, DeviceMemoryResource mr
):
    # helper to convert a gather map to a Column
    return Column.from_libcudf(
        move(
            make_unique[column](
                #move(gather_map),
                move(dereference(gather_map.get())),
                device_buffer(),
                0
            )
        ),
        stream,
        mr
    )

cdef extern from "cudf/join/hash_join.hpp" namespace "cudf":
    cdef enum class nullable_join "cudf::nullable_join":
        YES
        NO

    cdef cppclass hash_join:
        hash_join(table_view build,
                  null_equality compare_nulls,
                  cuda_stream_view stream) except +
        hash_join(table_view build,
                  nullable_join has_nulls,
                  null_equality compare_nulls,
                  double load_factor,
                  cuda_stream_view stream) except +

        # Join methods
        pair[unique_ptr[device_uvector[size_type]],
             unique_ptr[device_uvector[size_type]]] \
        inner_join(table_view probe,
                   optional[size_t] output_size,        # std::optional<std::size_t>
                   cuda_stream_view stream,
                   device_memory_resource* mr) const

        pair[unique_ptr[device_uvector[size_type]],
             unique_ptr[device_uvector[size_type]]] \
        left_join(table_view probe,
                  optional[size_t] output_size,
                  cuda_stream_view stream,
                  device_memory_resource* mr) const

        pair[unique_ptr[device_uvector[size_type]],
             unique_ptr[device_uvector[size_type]]] \
        full_join(table_view probe,
                  optional[size_t] output_size,
                  cuda_stream_view stream,
                  device_memory_resource* mr) const

class NullableJoin(IntEnum):
    YES = <int>nullable_join.YES
    NO = <int>nullable_join.NO

cdef class HashJoin:
    cdef hash_join* _ptr

    def __cinit__(self,
                  build_table,        # pylibcudf.Table Python object
                  compare_nulls,      # NullEquality or int
                  has_nulls, # =None,     # NullableJoin or None
                  double load_factor, #=0.5,
                  stream):       # pylibcudf stream wrapper

        # --- declare all C variables up front ---
        cdef table_view c_build
        cdef cuda_stream_view c_stream
        cdef null_equality c_eq
        cdef nullable_join c_nj
        cdef Table tb
        cdef Stream cudfstream 

        tb = <Table>build_table
        cudfstream = <Stream>stream

        # --- convert build_table -> table_view (Python call may raise) ---
        try:
            # build_table.view() is a Python call; the .pxd must declare it returns table_view
            c_build = tb.view()
        except Exception as e:
            raise TypeError("build_table must be a pylibcudf.Table with .view() returning table_view") from e

        # --- convert compare_nulls -> null_equality ---
        if isinstance(compare_nulls, NullEquality):
            c_eq = <null_equality><int>compare_nulls
        elif isinstance(compare_nulls, int):
            c_eq = <null_equality>compare_nulls
        else:
            raise TypeError("compare_nulls must be NullEquality or int")

        # --- convert stream -> cuda_stream_view ---
        try:
            c_stream = cudfstream.view()
        except Exception as e:
            raise TypeError("stream must be a pylibcudf stream wrapper with .view() returning cuda_stream_view") from e

        # --- choose constructor and call C++ ---
        if has_nulls is None:
            self._ptr = new hash_join(c_build, c_eq, c_stream)
        else:
            if isinstance(has_nulls, NullableJoin):
                c_nj = <nullable_join><int>has_nulls
            elif isinstance(has_nulls, int):
                c_nj = <nullable_join>has_nulls
            else:
                raise TypeError("has_nulls must be NullableJoin, int, or None")
            c_nj = NullableJoin.YES
            self._ptr = new hash_join(c_build, c_nj, c_eq, load_factor, c_stream)

    def __dealloc__(self):
        if self._ptr is not NULL:
            del self._ptr
            self._ptr = NULL

    def inner_join(self,
                   probe_view,
                   output_size,        # std::optional<std::size_t>
                   stream,
                   mr):
        cdef table_view c_probe
        cdef cuda_stream_view c_stream
        cdef pair[unique_ptr[device_uvector[size_type]],
                  unique_ptr[device_uvector[size_type]]] res
        cdef Table tb
        cdef Stream cudfstream 
        cdef optional[size_t] c_output_size
        cdef DeviceMemoryResource dmr
        cdef device_memory_resource* c_dmr

        tb = <Table>probe_view
        cudfstream = <Stream>stream
        dmr = <DeviceMemoryResource>mr

        try:
            c_probe = tb.view()
        except Exception as e:
            raise TypeError("build_table must be a pylibcudf.Table with .view() returning table_view") from e

        try:
            c_stream = cudfstream.view()
        except Exception as e:
            raise TypeError("stream must be a pylibcudf stream wrapper with .view() returning cuda_stream_view") from e

        if output_size is None:
            c_output_size = optional[size_t]()
        else:
            c_output_size = optional[size_t](<size_t>output_size)
        
        c_dmr = dmr.get_mr()

        res = self._ptr.inner_join(c_probe, c_output_size, c_stream, c_dmr)

        return _column_from_gather_map(res.first, cudfstream, dmr), _column_from_gather_map(res.second, cudfstream, dmr)

    """
    def left_join(self,
                  table_view probe_view,
                  object output_size,        # std::optional<std::size_t>
                  cuda_stream_view stream,
                  device_memory_resource* mr):
        cdef pair[unique_ptr[device_uvector[size_type]],
                  unique_ptr[device_uvector[size_type]]] res
        res = self._ptr.left_join(probe_view, output_size, stream, mr)
        return _wrap_device_uvector(res.first), _wrap_device_uvector(res.second)

    def full_join(self,
                  table_view probe_view,
                  object output_size,        # std::optional<std::size_t>
                  cuda_stream_view stream,
                  device_memory_resource* mr):
        cdef pair[unique_ptr[device_uvector[size_type]],
                  unique_ptr[device_uvector[size_type]]] res
        res = self._ptr.full_join(probe_view, output_size, stream, mr)
        return _wrap_device_uvector(res.first), _wrap_device_uvector(res.second)
    """
