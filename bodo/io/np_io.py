# Copyright (C) 2019 Bodo Inc. All rights reserved.
import numpy as np
import bodo
import numba
from numba.core import types
from numba.extending import overload, intrinsic, overload_method
from bodo.libs.str_ext import string_type

from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

from bodo.libs import hio
import llvmlite.binding as ll

ll.add_symbol("get_file_size", hio.get_file_size)
ll.add_symbol("file_read", hio.file_read)
ll.add_symbol("file_read_parallel", hio.file_read_parallel)
ll.add_symbol("file_write", hio.file_write)
ll.add_symbol("file_write_parallel", hio.file_write_parallel)


_get_file_size = types.ExternalFunction("get_file_size", types.int64(types.voidptr))
_file_read = types.ExternalFunction(
    "file_read", types.void(types.voidptr, types.voidptr, types.intp)
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


dummy_use = numba.njit(lambda a: None)


# @overload(np.fromfile, no_unliteral=True)
# def fromfile_overload(fname, dtype):
#     if fname != string_type:
#         raise("np.fromfile() invalid filename type")
#     if dtype is not None and not isinstance(dtype, types.DTypeSpec):
#         raise("np.fromfile() invalid dtype")
#
#     # FIXME: import here since hio has hdf5 which might not be available
#     from .. import hio
#     import llvmlite.binding as ll
#     ll.add_symbol('get_file_size', hio.get_file_size)
#     ll.add_symbol('file_read', hio.file_read)
#
#     def fromfile_impl(fname, dtype):
#         size = get_file_size(fname)
#         dtype_size = get_dtype_size(dtype)
#         A = np.empty(size//dtype_size, dtype=dtype)
#         file_read(fname, A.ctypes, size)
#         return A
#
#     return fromfile_impl


def _handle_np_fromfile(assign, lhs, rhs):
    """translate np.fromfile() to native
    """
    # TODO: dtype in kws
    if len(rhs.args) != 2:  # pragma: no cover
        raise ValueError("np.fromfile(): file name and dtype expected")

    _fname = rhs.args[0]
    _dtype = rhs.args[1]

    def fromfile_impl(fname, dtype):  # pragma: no cover
        size = get_file_size(fname)
        dtype_size = get_dtype_size(dtype)
        A = np.empty(size // dtype_size, dtype=dtype)
        file_read(fname, A, size)
        read_arr = A

    f_block = compile_to_numba_ir(
        fromfile_impl,
        {
            "np": np,
            "get_file_size": get_file_size,
            "file_read": file_read,
            "get_dtype_size": get_dtype_size,
        },
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [_fname, _dtype])
    nodes = f_block.body[:-3]  # remove none return
    nodes[-1].target = lhs
    return nodes


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
            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            # TODO: unicode name
            file_write(fname._data, A.ctypes, dtype_size * A.size)
            dummy_use(fname)

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
            _file_write_parallel(fname._data, A.ctypes, start, count, elem_size)
            dummy_use(fname)

        return _impl


def file_read_parallel(fname, arr, start, count):  # pragma: no cover
    return


@overload(file_read_parallel)
def file_read_parallel_overload(fname, arr, start, count):
    if fname == string_type:

        def _impl(fname, arr, start, count):  # pragma: no cover
            dtype_size = get_dtype_size(arr.dtype)
            _file_read_parallel(
                fname._data, arr.ctypes, start * dtype_size, count * dtype_size
            )
            dummy_use(fname)

        return _impl


def file_read(fname, arr, size):  # pragma: no cover
    return


@overload(file_read)
def file_read_overload(fname, arr, size):
    if fname == string_type:
        # TODO: unicode name
        def impl(fname, arr, size):  # pragma: no cover
            _file_read(fname._data, arr.ctypes, size)
            dummy_use(fname)

        return impl


def get_file_size(fname):  # pragma: no cover
    return 0


@overload(get_file_size)
def get_file_size_overload(fname):
    if fname == string_type:
        # TODO: unicode name
        def impl(fname):  # pragma: no cover
            s = _get_file_size(fname._data)
            dummy_use(fname)
            return s

        return impl
