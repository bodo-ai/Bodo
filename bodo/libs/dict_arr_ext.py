# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Dictionary encoded array data type, similar to DictionaryArray of Arrow.
The purpose is to improve memory consumption and performance over string_array_type for
string arrays that have a lot of repetitive values (typical in practice).
Can be extended to be used with types other than strings as well.
See:
https://bodo.atlassian.net/browse/BE-2295
https://bodo.atlassian.net/wiki/spaces/B/pages/993722369/Dictionary-encoded+String+Array+Support+in+Parquet+read+compute+...
https://arrow.apache.org/docs/cpp/api/array.html#dictionary-encoded
"""

import operator
import re

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_new_ref, lower_builtin, lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.libs.bool_arr_ext import init_bool_array
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import (
    StringArrayType,
    get_str_arr_item_length,
    overload_str_arr_astype,
    pre_alloc_string_array,
)
from bodo.utils.typing import BodoArrayIterator, raise_bodo_error

# we use nullable int32 for dictionary indices to match Arrow for faster and easier IO.
# more than 2 billion unique values doesn't make sense for a dictionary-encoded array.
dict_indices_arr_type = IntegerArrayType(types.int32)


class DictionaryArrayType(types.IterableType, types.ArrayCompatible):
    """Data type for dictionary-encoded arrays"""

    def __init__(self, arr_data_type):
        self.data = arr_data_type
        super(DictionaryArrayType, self).__init__(
            name=f"DictionaryArrayType({arr_data_type})"
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)

    @property
    def dtype(self):
        return self.data.dtype

    def copy(self):
        return DictionaryArrayType(self.data)

    @property
    def indices_type(self):
        return dict_indices_arr_type

    @property
    def indices_dtype(self):
        return dict_indices_arr_type.dtype

    def unify(self, typingctx, other):
        if other == bodo.string_array_type:
            return bodo.string_array_type


dict_str_arr_type = DictionaryArrayType(bodo.string_array_type)


# TODO(ehsan): make DictionaryArrayType inner data mutable using a payload structure?
@register_model(DictionaryArrayType)
class DictionaryArrayModel(models.StructModel):
    """dictionary array data model, storing int32 indices and array data"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", dict_indices_arr_type),
            # flag to indicate whether the dictionary is the same across all ranks
            # to avoid extra communication. This may be false after parquet read but
            # set to true after other operations like shuffle
            ("has_global_dictionary", types.bool_),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DictionaryArrayType, "data", "_data")
make_attribute_wrapper(DictionaryArrayType, "indices", "_indices")
make_attribute_wrapper(
    DictionaryArrayType, "has_global_dictionary", "_has_global_dictionary"
)


lower_builtin("getiter", dict_str_arr_type)(numba.np.arrayobj.getiter_array)


@intrinsic
def init_dict_arr(typingctx, data_t, indices_t, glob_dict_t=None):
    """Create a dictionary-encoded array with provided index and data values."""

    assert indices_t == dict_indices_arr_type, "invalid indices type for dict array"

    def codegen(context, builder, signature, args):
        data, indices, glob_dict = args
        # create dict arr struct and store values
        dict_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dict_arr.data = data
        dict_arr.indices = indices
        dict_arr.has_global_dictionary = glob_dict

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data)
        context.nrt.incref(builder, signature.args[1], indices)

        return dict_arr._getvalue()

    ret_typ = DictionaryArrayType(data_t)
    sig = ret_typ(data_t, indices_t, types.bool_)
    return sig, codegen


@typeof_impl.register(pa.DictionaryArray)
def typeof_dict_value(val, c):
    # only support dict-encoded string arrays for now, TODO(ehsan): support other types
    if val.type.value_type == pa.string():
        return dict_str_arr_type


def to_pa_dict_arr(A):
    """convert array 'A' to a PyArrow dictionary-encoded array if it is not already.
    'A' can be a Pandas or Numpy array
    """
    if isinstance(A, pa.DictionaryArray):
        return A

    # convert np.nan, pd.NA to None to avoid PyArrow error
    for i in range(len(A)):
        if pd.isna(A[i]):
            A[i] = None

    return pa.array(A).dictionary_encode()


@unbox(DictionaryArrayType)
def unbox_dict_arr(typ, val, c):
    """
    Unbox a PyArrow dictionary array of string values.
    Simple unboxing to enable testing with PyArrow arrays for now.
    TODO(ehsan): improve performance by copying buffers directly in C++
    """

    # if regular arrays of strings can be unboxed as dictionary encoded arrays
    if bodo.hiframes.boxing._use_dict_str_type:
        to_pa_dict_arr_obj = c.pyapi.unserialize(
            c.pyapi.serialize_object(to_pa_dict_arr)
        )
        val = c.pyapi.call_function_objargs(to_pa_dict_arr_obj, [val])
        c.pyapi.decref(to_pa_dict_arr_obj)

    dict_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # get a numpy array of string objects to unbox
    # dict_arr.data = val.dictionary.to_numpy(False)
    data_obj = c.pyapi.object_getattr_string(val, "dictionary")
    false_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))
    np_str_arr_obj = c.pyapi.call_method(data_obj, "to_numpy", (false_obj,))
    dict_arr.data = c.unbox(typ.data, np_str_arr_obj).value

    # get a Pandas Int32 array to unbox
    # dict_arr.indices = pd.array(val.indices, "Int32")
    indices_obj = c.pyapi.object_getattr_string(val, "indices")
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    int32_str_obj = c.pyapi.string_from_constant_string("Int32")
    pd_int_arr_obj = c.pyapi.call_method(
        pd_class_obj, "array", (indices_obj, int32_str_obj)
    )
    dict_arr.indices = c.unbox(dict_indices_arr_type, pd_int_arr_obj).value
    # assume dictionarys are not the same across all ranks to be conservative
    dict_arr.has_global_dictionary = c.context.get_constant(types.bool_, False)

    c.pyapi.decref(data_obj)
    c.pyapi.decref(false_obj)
    c.pyapi.decref(np_str_arr_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(int32_str_obj)
    c.pyapi.decref(pd_int_arr_obj)

    # if we created a new PyArrow array, decref it since not coming from user context
    if bodo.hiframes.boxing._use_dict_str_type:
        c.pyapi.decref(val)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(dict_arr._getvalue(), is_error=is_error)


@box(DictionaryArrayType)
def box_dict_arr(typ, val, c):
    """box dict array into numpy array of string objects"""

    dict_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # create a PyArrow dictionary array fron indices and data
    # pa.DictionaryArray.from_arrays(dict_arr.data, dict_arr.indices)
    mod_name = c.context.insert_const_string(c.builder.module, "pyarrow")
    pa_class_obj = c.pyapi.import_module_noblock(mod_name)
    pa_dict_arr_class = c.pyapi.object_getattr_string(pa_class_obj, "DictionaryArray")
    c.context.nrt.incref(c.builder, typ.data, dict_arr.data)
    data_arr_obj = c.box(typ.data, dict_arr.data)
    c.context.nrt.incref(c.builder, dict_indices_arr_type, dict_arr.indices)
    indices_obj = c.box(dict_indices_arr_type, dict_arr.indices)
    pa_dict_arr_obj = c.pyapi.call_method(
        pa_dict_arr_class, "from_arrays", (indices_obj, data_arr_obj)
    )

    # convert to numpy array of string objects
    # pa_dict_arr.to_numpy(False)
    false_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))
    np_str_arr_obj = c.pyapi.call_method(pa_dict_arr_obj, "to_numpy", (false_obj,))

    c.pyapi.decref(pa_class_obj)
    c.pyapi.decref(data_arr_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(pa_dict_arr_class)
    c.pyapi.decref(pa_dict_arr_obj)
    c.pyapi.decref(false_obj)

    c.context.nrt.decref(c.builder, typ, val)
    return np_str_arr_obj


@overload(len, no_unliteral=True)
def overload_dict_arr_len(A):
    if isinstance(A, DictionaryArrayType):
        return lambda A: len(A._indices)  # pragma: no cover


@overload_attribute(DictionaryArrayType, "shape")
def overload_dict_arr_shape(A):
    return lambda A: (len(A._indices),)  # pragma: no cover


@overload_attribute(DictionaryArrayType, "ndim")
def overload_dict_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(DictionaryArrayType, "size")
def overload_dict_arr_size(A):
    return lambda A: len(A._indices)  # pragma: no cover


@overload_method(DictionaryArrayType, "tolist", no_unliteral=True)
def overload_dict_arr_tolist(A):
    return lambda A: list(A)  # pragma: no cover


# TODO(ehsan): more optimized version for dictionary-encoded case
overload_method(DictionaryArrayType, "astype", no_unliteral=True)(
    overload_str_arr_astype
)


@overload_method(DictionaryArrayType, "copy", no_unliteral=True)
def overload_dict_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        return init_dict_arr(
            A._data.copy(), A._indices.copy(), A._has_global_dictionary
        )

    return copy_impl


@overload_attribute(DictionaryArrayType, "dtype")
def overload_dict_arr_dtype(A):
    return lambda A: A._data.dtype  # pragma: no cover


@overload_attribute(DictionaryArrayType, "nbytes")
def dict_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._indices.nbytes  # pragma: no cover


@lower_constant(DictionaryArrayType)
def lower_constant_dict_arr(context, builder, typ, pyval):
    """embed constant dict array value by getting constant values for underlying
    indices and data arrays.
    """

    if bodo.hiframes.boxing._use_dict_str_type and isinstance(pyval, np.ndarray):
        pyval = pa.array(pyval).dictionary_encode()

    data_arr = pyval.dictionary.to_numpy(False)
    indices_arr = pd.array(pyval.indices, "Int32")

    data_arr = context.get_constant_generic(builder, typ.data, data_arr)
    indices_arr = context.get_constant_generic(
        builder, dict_indices_arr_type, indices_arr
    )

    has_global_dictionary = context.get_constant(types.bool_, False)
    dic_array = lir.Constant.literal_struct(
        [data_arr, indices_arr, has_global_dictionary]
    )
    return dic_array


@overload(operator.getitem, no_unliteral=True)
def dict_arr_getitem(A, ind):
    if not isinstance(A, DictionaryArrayType):
        return

    if isinstance(ind, types.Integer):

        def dict_arr_getitem_impl(A, ind):  # pragma: no cover
            # return empty string for NA to match string_array_type behavior
            if bodo.libs.array_kernels.isna(A._indices, ind):
                return ""
            dict_ind = A._indices[ind]
            return A._data[dict_ind]

        return dict_arr_getitem_impl

    # we just need to update indices for all non-scalar output cases
    # we could also trim down the dictionary in some cases to save memory but doesn't
    # seem to be worth it
    return lambda A, ind: init_dict_arr(
        A._data, A._indices[ind], A._has_global_dictionary
    )  # pragma: no cover


@overload_method(DictionaryArrayType, "_decode", no_unliteral=True)
def overload_dict_arr_decode(A):
    """decode dictionary encoded array to a regular string array.
    Used as a fallback when dict array is not supported yet.
    """

    def impl(A):  # pragma: no cover
        data = A._data
        indices = A._indices
        n = len(indices)
        str_lengths = [get_str_arr_item_length(data, i) for i in range(len(data))]

        n_chars = 0
        for i in range(n):
            if not bodo.libs.array_kernels.isna(indices, i):
                n_chars += str_lengths[indices[i]]

        out_arr = pre_alloc_string_array(n, n_chars)
        for i in range(n):
            if bodo.libs.array_kernels.isna(indices, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            ind = indices[i]
            if bodo.libs.array_kernels.isna(data, ind):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = data[ind]

        return out_arr

    return impl


@overload(operator.setitem)
def dict_arr_setitem(A, idx, val):
    if not isinstance(A, DictionaryArrayType):
        return

    raise_bodo_error("DictionaryArrayType is read-only and doesn't support setitem yet")


@numba.njit(no_cpython_wrapper=True)
def find_dict_ind(arr, val):
    """find index of 'val' in dictionary of 'arr'. Return -1 if not found.
    NOTE: Assumes that values in the dictionary are unique.
    """
    dict_ind = -1
    data = arr._data
    for i in range(len(data)):
        if bodo.libs.array_kernels.isna(data, i):
            continue
        if data[i] == val:
            dict_ind = i
            break

    return dict_ind


@numba.njit(no_cpython_wrapper=True)
def dict_arr_eq(arr, val):
    """implements equality comparison between a dictionary array and a scalar value"""
    n = len(arr)
    # NOTE: Assumes that values in the dictionary are unique.
    dict_ind = find_dict_ind(arr, val)

    if dict_ind == -1:
        return init_bool_array(
            np.full(n, False, np.bool_), arr._indices._null_bitmap.copy()
        )

    return arr._indices == dict_ind


@numba.njit(no_cpython_wrapper=True)
def dict_arr_ne(arr, val):
    """implements inequality comparison between a dictionary array and a scalar value"""
    n = len(arr)
    # NOTE: Assumes that values in the dictionary are unique.
    dict_ind = find_dict_ind(arr, val)

    if dict_ind == -1:
        return init_bool_array(
            np.full(n, True, np.bool_), arr._indices._null_bitmap.copy()
        )

    return arr._indices != dict_ind


def get_binary_op_overload(op, lhs, rhs):
    """return an optimized implementation for binary operation with dictionary array
    if possible.
    Currently supports only equality and inequality comparisons.
    """
    # NOTE: equality/inequality between two arrays is implemented in str_arr_ext.py
    if op == operator.eq:
        if lhs == dict_str_arr_type and types.unliteral(rhs) == bodo.string_type:
            return lambda lhs, rhs: dict_arr_eq(lhs, rhs)  # pragma: no cover
        if rhs == dict_str_arr_type and types.unliteral(lhs) == bodo.string_type:
            return lambda lhs, rhs: dict_arr_eq(rhs, lhs)  # pragma: no cover

    if op == operator.ne:
        if lhs == dict_str_arr_type and types.unliteral(rhs) == bodo.string_type:
            return lambda lhs, rhs: dict_arr_ne(lhs, rhs)  # pragma: no cover
        if rhs == dict_str_arr_type and types.unliteral(lhs) == bodo.string_type:
            return lambda lhs, rhs: dict_arr_ne(rhs, lhs)  # pragma: no cover


def convert_dict_arr_to_int(arr, dtype):  # pragma: no cover
    return arr


@overload(convert_dict_arr_to_int)
def convert_dict_arr_to_int_overload(arr, dtype):
    """convert dictionary array to integer array without materializing all strings"""

    def impl(arr, dtype):  # pragma: no cover
        # convert dictionary array to integer array
        data_dict = arr._data
        int_vals = bodo.libs.int_arr_ext.alloc_int_array(len(data_dict), dtype)
        for j in range(len(data_dict)):
            if bodo.libs.array_kernels.isna(data_dict, j):
                bodo.libs.array_kernels.setna(int_vals, j)
                continue
            # convert to int64 to support string arrays, see comment in fix_arr_dtype
            int_vals[j] = np.int64(data_dict[j])

        # create output array using dictionary indices
        n = len(arr)
        indices = arr._indices
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(n, dtype)
        for i in range(n):
            if bodo.libs.array_kernels.isna(indices, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = int_vals[indices[i]]

        return out_arr

    return impl


@lower_cast(DictionaryArrayType, StringArrayType)
def cast_dict_str_arr_to_str_arr(context, builder, fromty, toty, val):
    """
    Cast a DictionaryArrayType with string data to StringArrayType
    by calling decode_if_dict_array.
    """
    if fromty != dict_str_arr_type:
        # Only support this cast with dictionary arrays of strings.
        return
    func = bodo.utils.typing.decode_if_dict_array_overload(fromty)
    sig = toty(fromty)
    res = context.compile_internal(builder, func, sig, (val,))
    return impl_ret_new_ref(context, builder, toty, res)


@numba.jit(cache=True, no_cpython_wrapper=True)
def str_replace(arr, pat, repl, flags, regex):  # pragma: no cover
    """implement optimized string replace for dictionary array.
    Only transforms the dictionary array and just copies the indices.
    """
    # Pandas implementation:
    # https://github.com/pandas-dev/pandas/blob/60c2940fcf28ee84b64ebda813adfd78a68eea9f/pandas/core/strings/object_array.py#L141
    data_arr = arr._data
    n_data = len(data_arr)
    out_str_arr = pre_alloc_string_array(n_data, -1)

    if regex:
        e = re.compile(pat, flags)
        for i in range(n_data):
            if bodo.libs.array_kernels.isna(data_arr, i):
                bodo.libs.array_kernels.setna(out_str_arr, i)
                continue
            out_str_arr[i] = e.sub(repl=repl, string=data_arr[i])
    else:
        for i in range(n_data):
            if bodo.libs.array_kernels.isna(data_arr, i):
                bodo.libs.array_kernels.setna(out_str_arr, i)
                continue
            out_str_arr[i] = data_arr[i].replace(pat, repl)

    return init_dict_arr(out_str_arr, arr._indices.copy(), arr._has_global_dictionary)
