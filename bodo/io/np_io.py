"""
File to support the numpy file IO API (np.fromfile(), np.tofile()).
The actual definition of fromfile is inside untyped pass with the
other IO operations.
"""

import llvmlite.binding as ll
import numpy as np
from numba.core import types
from numba.extending import intrinsic, overload, overload_method

import bodo
from bodo.libs import hio
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.utils.utils import check_java_installation

ll.add_symbol("get_file_size", hio.get_file_size)
ll.add_symbol("file_read", hio.file_read)
ll.add_symbol("file_read_parallel", hio.file_read_parallel)
ll.add_symbol("file_write", hio.file_write_py_entrypt)
ll.add_symbol("file_write_parallel", hio.file_write_parallel_py_entrypt)


_get_file_size = types.ExternalFunction("get_file_size", types.int64(types.voidptr))
_file_read = types.ExternalFunction(
    "file_read", types.void(types.voidptr, types.voidptr, types.intp, types.intp)
)
_file_read_parallel = types.ExternalFunction(
    "file_read_parallel",
    types.void(types.voidptr, types.voidptr, types.intp, types.intp),
)

file_write = types.ExternalFunction(
    "file_write", types.void(types.voidptr, types.voidptr, types.intp)
)

_file_write_parallel = types.ExternalFunction(
    "file_write_parallel",
    types.void(types.voidptr, types.voidptr, types.intp, types.intp, types.intp),
)


@intrinsic
def get_dtype_size(typingctx, dtype=None):
    assert isinstance(dtype, types.DTypeSpec)

    def codegen(context, builder, sig, args):
        num_bytes = context.get_abi_sizeof(context.get_data_type(dtype.dtype))
        return context.get_constant(types.intp, num_bytes)

    return types.intp(dtype), codegen


@overload_method(types.Array, "tofile")
def tofile_overload(arr, fname):
    # TODO: fix Numba to convert literal
    if fname == string_type or isinstance(fname, types.StringLiteral):

        def tofile_impl(arr, fname):  # pragma: no cover
            # check_java_installation is a check for hdfs that java is installed
            check_java_installation(fname)

            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            # TODO: unicode name
            file_write(unicode_to_utf8(fname), A.ctypes, dtype_size * A.size)
            bodo.utils.utils.check_and_propagate_cpp_exception()

        return tofile_impl


# from llvmlite import ir as lir
# @intrinsic
# def print_array_ptr(typingctx, arr_ty):
#     assert isinstance(arr_ty, types.Array)
#     def codegen(context, builder, sig, args):
#         out = make_array(sig.args[0])(context, builder, args[0])
#         cgutils.printf(builder, "%p ", out.data)
#         cgutils.printf(builder, "%lf ", builder.bitcast(out.data, lir.IntType(64).as_pointer()))
#         return context.get_dummy_value()
#     return types.void(arr_ty), codegen


def file_write_parallel(fname, arr, start, count):  # pragma: no cover
    pass


# TODO: fix A.ctype inlined case
@overload(file_write_parallel)
def file_write_parallel_overload(fname, arr, start, count):
    if fname == string_type:  # avoid str literal

        def _impl(fname, arr, start, count):  # pragma: no cover
            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            elem_size = dtype_size * bodo.libs.distributed_api.get_tuple_prod(
                A.shape[1:]
            )
            # bodo.cprint(start, count, elem_size)
            # TODO: unicode name
            _file_write_parallel(
                unicode_to_utf8(fname), A.ctypes, start, count, elem_size
            )
            bodo.utils.utils.check_and_propagate_cpp_exception()

        return _impl


def file_read_parallel(fname, arr, start, count):  # pragma: no cover
    return


@overload(file_read_parallel)
def file_read_parallel_overload(fname, arr, start, count, offset):
    if fname == string_type:

        def _impl(fname, arr, start, count, offset):  # pragma: no cover
            dtype_size = get_dtype_size(arr.dtype)
            _file_read_parallel(
                unicode_to_utf8(fname),
                arr.ctypes,
                (start * dtype_size) + offset,  # Offset is given in bytes
                count * dtype_size,
            )
            bodo.utils.utils.check_and_propagate_cpp_exception()

        return _impl


def file_read(fname, arr, size, offset):  # pragma: no cover
    return


@overload(file_read)
def file_read_overload(fname, arr, size, offset):
    if fname == string_type:
        # TODO: unicode name
        def impl(fname, arr, size, offset):  # pragma: no cover
            _file_read(unicode_to_utf8(fname), arr.ctypes, size, offset)
            bodo.utils.utils.check_and_propagate_cpp_exception()

        return impl


def get_file_size(fname, count, offset, dtype_size):  # pragma: no cover
    return 0


@overload(get_file_size)
def get_file_size_overload(fname, count, offset, dtype_size):
    if fname == string_type:
        # TODO: unicode name
        def impl(fname, count, offset, dtype_size):  # pragma: no cover
            # TODO(Nick): What is the best way to handle error cases
            if offset < 0:
                return -1
            s = _get_file_size(unicode_to_utf8(fname)) - offset
            bodo.utils.utils.check_and_propagate_cpp_exception()
            if count != -1:
                s = min(s, count * dtype_size)
            # TODO(Nick): What is the best way to handle error cases
            if s < 0:
                return -1
            return s

        return impl
