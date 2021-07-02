# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for binary (bytes) objects, which are usually immutable.
It is equivalent to string array, except that it stores a 'bytes' object for each
element instead of 'str'.
"""
import operator

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
)

import bodo
from bodo.libs import hstr_ext
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.utils.typing import BodoError, is_list_like_index_type

_bytes_fromhex = types.ExternalFunction(
    "bytes_fromhex", types.int64(types.voidptr, types.voidptr, types.uint64)
)
ll.add_symbol("bytes_to_hex", hstr_ext.bytes_to_hex)
ll.add_symbol("bytes_fromhex", hstr_ext.bytes_fromhex)
bytes_type = types.Bytes(types.uint8, 1, "C", readonly=True)


# type for ndarray with bytes object values
class BinaryArrayType(types.ArrayCompatible):
    def __init__(self):
        super(BinaryArrayType, self).__init__(name="BinaryArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return bytes_type

    def copy(self):
        return BinaryArrayType()


binary_array_type = BinaryArrayType()


@overload(len, no_unliteral=True)
def bin_arr_len_overload(bin_arr):
    if bin_arr == binary_array_type:
        return lambda bin_arr: len(bin_arr._data)  # pragma: no cover


@overload_attribute(BinaryArrayType, "size")
def bin_arr_size_overload(bin_arr):
    return lambda bin_arr: len(bin_arr._data)  # pragma: no cover


@overload_attribute(BinaryArrayType, "shape")
def bin_arr_shape_overload(bin_arr):
    return lambda bin_arr: (len(bin_arr._data),)  # pragma: no cover


@overload_attribute(BinaryArrayType, "nbytes")
def bin_arr_nbytes_overload(bin_arr):
    return lambda bin_arr: bin_arr._data.nbytes  # pragma: no cover


@overload_attribute(BinaryArrayType, "ndim")
def overload_bin_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@intrinsic
def init_binary_arr(typingctx, data_typ=None):
    """create a new binary array from input data array(char) array data"""
    assert isinstance(data_typ, ArrayItemArrayType) and data_typ.dtype == types.Array(
        types.uint8, 1, "C"
    )

    def codegen(context, builder, sig, args):
        (data_arr,) = args
        bin_array = context.make_helper(builder, binary_array_type)
        bin_array.data = data_arr
        context.nrt.incref(builder, data_typ, data_arr)
        return bin_array._getvalue()

    return binary_array_type(data_typ), codegen


@intrinsic
def init_bytes_type(typingctx, data_typ, length_type):
    """create a new bytes array from input data array(uint8) data and length,
    where it is assumed that length <= len(data)"""
    assert data_typ == types.Array(types.uint8, 1, "C")
    assert length_type == types.int64

    def codegen(context, builder, sig, args):
        # Convert input/output to structs to reference fields
        int_arr = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        length = args[1]
        bytes_array = cgutils.create_struct_proxy(bytes_type)(context, builder)

        # Initialize the fields of the byte array (mostly copied from Numba)
        bytes_array.meminfo = context.nrt.meminfo_alloc(builder, length)
        bytes_array.nitems = length
        bytes_array.itemsize = lir.Constant(bytes_array.itemsize.type, 1)
        bytes_array.data = context.nrt.meminfo_data(builder, bytes_array.meminfo)
        bytes_array.parent = cgutils.get_null_value(bytes_array.parent.type)
        bytes_array.shape = cgutils.pack_array(
            builder, [length], context.get_value_type(types.intp)
        )
        bytes_array.strides = int_arr.strides

        # Memcpy the data from int array to bytes array, truncating if necessary.
        cgutils.memcpy(builder, bytes_array.data, int_arr.data, length)
        return bytes_array._getvalue()

    return bytes_type(data_typ, length_type), codegen


@overload_method(BinaryArrayType, "copy", no_unliteral=True)
def binary_arr_copy_overload(arr):
    """implement copy by copying internal array(item) array"""

    def copy_impl(arr):  # pragma: no cover
        return init_binary_arr(arr._data.copy())

    return copy_impl


@overload_method(types.Bytes, "hex")
def binary_arr_hex(arr):
    """
    Implementation of Bytes.hex. This is handled in regular Python here:
    https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Objects/clinic/bytesobject.c.h#L807
    https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Objects/bytesobject.c#L2464
    https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Python/pystrhex.c#L164
    https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Python/pystrhex.c#L7

    Note: We ignore sep and bytes_per_sep_group because sep is always NULL
    """
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(arr):
        # Allocate the unicode output. Since 256 = 16^2, we
        # allocate 2 elements for every byte (+1 null terminator)
        length = len(arr) * 2
        output = numba.cpython.unicode._empty_string(kind, length, 1)
        bytes_to_hex(output, arr)
        return output

    return impl


# Support casting uint8ptr to void* for hash impl
@lower_cast(types.CPointer(types.uint8), types.voidptr)
def cast_uint8_array_to_voidptr(context, builder, fromty, toty, val):
    return val


# Support accessing data from jit functions
make_attribute_wrapper(types.Bytes, "data", "_data")


@overload_method(types.Bytes, "__hash__")
def bytes_hash(arr):
    def impl(arr):  # pragma: no cover
        # Implement hash with _Py_HashBytes
        # TODO: cache
        return numba.cpython.hashing._Py_HashBytes(arr._data, len(arr))

    return impl


@intrinsic
def bytes_to_hex(typingctx, output, arr):
    """Call C implementation of bytes_to_hex"""

    def codegen(context, builder, sig, args):
        output_arr = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        bytes_arr = cgutils.create_struct_proxy(sig.args[1])(
            context, builder, value=args[1]
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        hex_func = builder.module.get_or_insert_function(fnty, name="bytes_to_hex")
        builder.call(hex_func, (output_arr.data, bytes_arr.data, bytes_arr.nitems))

    return types.void(output, arr), codegen


@overload(operator.getitem, no_unliteral=True)
def binary_arr_getitem(arr, ind):
    if arr != binary_array_type:  # pragma: no cover
        return

    # Indexing is supported for any indexing support for ArrayItemArray
    if isinstance(ind, types.Integer):

        def impl(arr, ind):  # pragma: no cover
            data = arr._data[ind]
            return init_bytes_type(data, len(data))

        return impl

    # bool arr, int arr, and slice
    if (
        is_list_like_index_type(ind)
        and (ind.dtype == types.bool_ or isinstance(ind.dtype, types.Integer))
    ) or isinstance(ind, types.SliceType):
        return lambda arr, ind: init_binary_arr(arr._data[ind])  # pragma: no cover

    raise BodoError(
        f"getitem for Binary Array with indexing type {ind} not supported."
    )  # pragma: no cover


def bytes_fromhex(hex_str):
    """Internal call to support bytes.fromhex().
    Untyped pass replaces bytes.fromhex() with this call since class
    methods are not supported in Numba's typing
    """


@overload(bytes_fromhex)
def overload_bytes_fromhex(hex_str):
    """
    Bytes.fromhex is implemented using the Python implementation:
    https://github.com/python/cpython/blob/1d08d85cbe49c0748a8ee03aec31f89ab8e81496/Objects/bytesobject.c#L2359
    """
    # Use types.unliteral to avoid issues with string literals
    hex_str = types.unliteral(hex_str)
    if hex_str == bodo.string_type:
        kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

        def impl(hex_str):  # pragma: no cover
            if not hex_str._is_ascii or hex_str._kind != kind:
                raise TypeError("bytes.fromhex is only supported on ascii strings")
            # Allocate 1 byte per 2 characters. This overestimates if we skip spaces
            data_arr = np.empty(len(hex_str) // 2, np.uint8)
            # Populate the array
            length = _bytes_fromhex(data_arr.ctypes, hex_str._data, len(hex_str))
            # Wrap the result in a Bytes obj, truncating if necessary
            return init_bytes_type(data_arr, length)

        return impl

    raise BodoError(f"bytes.fromhex not supported with argument type {hex_str}")
