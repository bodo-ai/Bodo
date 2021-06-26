# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for binary (bytes) objects, which are usually immutable.
It is equivalent to string array, except that it stores a 'bytes' object for each
element instead of 'str'.
"""
import operator

import numba
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    overload,
    overload_attribute,
    overload_method,
)

from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.utils.typing import BodoError, is_list_like_index_type

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
def init_bytes_type(typingctx, data_typ=None):
    """create a new bytes array from input data array(uint8) data"""
    assert data_typ == types.Array(types.uint8, 1, "C")

    def codegen(context, builder, sig, args):
        # Convert input/output to structs to reference fields
        int_arr = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        bytes_array = cgutils.create_struct_proxy(bytes_type)(context, builder)

        # Initialize the fields of the byte array (mostly copied from Numba)
        bytes_array.meminfo = context.nrt.meminfo_alloc(builder, int_arr.nitems)
        bytes_array.nitems = int_arr.nitems
        bytes_array.itemsize = lir.Constant(bytes_array.itemsize.type, 1)
        bytes_array.data = context.nrt.meminfo_data(builder, bytes_array.meminfo)
        bytes_array.parent = cgutils.get_null_value(bytes_array.parent.type)
        bytes_array.shape = int_arr.shape
        bytes_array.strides = int_arr.strides

        # Memcpy the data from int array to bytes array
        cgutils.memcpy(builder, bytes_array.data, int_arr.data, bytes_array.nitems)
        return bytes_array._getvalue()

    return bytes_type(data_typ), codegen


@overload_method(BinaryArrayType, "copy", no_unliteral=True)
def binary_arr_copy_overload(arr):
    """implement copy by copying internal array(item) array"""

    def copy_impl(arr):  # pragma: no cover
        return init_binary_arr(arr._data.copy())

    return copy_impl


@overload_method(types.Bytes, "hex")
def binary_arr_hex(arr):
    def impl(arr):
        parts = []
        for i in range(len(arr)):
            byte_num = arr[i]
            parts.extend(int_to_hex_list(byte_num))
        return "".join(parts)

    return impl


@overload(hex)
def overload_hex(int_val):
    if isinstance(int_val, types.Integer):

        def impl(int_val):
            return "".join(["0x"] + int_to_hex_list(int_val))

        return impl


@numba.njit
def int_to_hex_list(int_val):
    """
    Convert an integer into a list of hex strings.
    """
    char_map = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
    }
    # TODO [BE-926]: Replace with a more efficient implementation
    # that matches CPython
    # Add default value for typing
    result = [""]
    while int_val > 0:
        digit = int_val % 16
        int_val = int_val // 16
        result.append(char_map[digit])
    return result[::-1]


@overload(operator.getitem, no_unliteral=True)
def binary_arr_getitem(arr, ind):
    if arr != binary_array_type:
        return

    # Indexing is supported for any indexing support for ArrayItemArray
    if isinstance(ind, types.Integer):
        return lambda arr, ind: init_bytes_type(arr._data[ind])

    # bool arr, int arr, and slice
    if (
        is_list_like_index_type(ind)
        and (ind.dtype == types.bool_ or isinstance(ind.dtype, types.Integer))
    ) or isinstance(ind, types.SliceType):
        return lambda arr, ind: init_binary_arr(arr._data[ind])

    raise BodoError(
        f"getitem for Binary Array with indexing type {ind} not supported."
    )  # pragma: no cover
