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

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba import generated_jit
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_new_ref, lower_builtin, lower_constant
from numba.core.typing import signature
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
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.ext import stream_join_cpp
from bodo.hiframes.pd_series_ext import if_series_to_array_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import (
    StringArrayType,
    get_str_arr_item_length,
    overload_str_arr_astype,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.utils.typing import (
    BodoArrayIterator,
    is_list_like_index_type,
    is_overload_none,
    raise_bodo_error,
)
from bodo.utils.utils import synchronize_error_njit

ll.add_symbol("generate_array_id", stream_join_cpp.generate_array_id)


# we use nullable int32 for dictionary indices to match Arrow for faster and easier IO.
# more than 2 billion unique values doesn't make sense for a dictionary-encoded array.
dict_indices_arr_type = IntegerArrayType(types.int32)


class DictionaryArrayType(types.IterableType, types.ArrayCompatible):
    """Data type for dictionary-encoded arrays"""

    def __init__(self, arr_data_type):
        self.data = arr_data_type
        super().__init__(name=f"DictionaryArrayType({arr_data_type})")

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
        if other == string_array_type:
            return string_array_type


dict_str_arr_type = DictionaryArrayType(string_array_type)


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
            # flag to indicate whether the dictionary has unique values on this rank.
            # This is used to support optimized implementations where decisions can be
            # made just based on indices.
            ("has_unique_local_dictionary", types.bool_),
            # Dictionary id that helps identify identical equivalent dictionaries quickly.
            # This is unique inside a rank but not globally.
            ("dict_id", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DictionaryArrayType, "data", "_data")
make_attribute_wrapper(DictionaryArrayType, "indices", "_indices")
make_attribute_wrapper(
    DictionaryArrayType, "has_global_dictionary", "_has_global_dictionary"
)
make_attribute_wrapper(
    DictionaryArrayType, "has_unique_local_dictionary", "_has_unique_local_dictionary"
)
make_attribute_wrapper(DictionaryArrayType, "dict_id", "_dict_id")

lower_builtin("getiter", dict_str_arr_type)(numba.np.arrayobj.getiter_array)


@intrinsic
def init_dict_arr(
    typingctx, data_t, indices_t, glob_dict_t, unique_dict_t, dict_id_if_present_t
):
    """Create a dictionary-encoded array with provided index and data values."""

    assert indices_t == dict_indices_arr_type, "invalid indices type for dict array"
    assert (
        is_overload_none(dict_id_if_present_t) or dict_id_if_present_t == types.int64
    ), "ID must be none if we are generating an ID or an id of a matching dictionary"
    generate_id = is_overload_none(dict_id_if_present_t)

    def codegen(context, builder, sig, args):
        data, indices, glob_dict, unique_dict, dict_id = args
        # create dict arr struct and store values
        dict_arr = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        dict_arr.data = data
        dict_arr.indices = indices
        dict_arr.has_global_dictionary = glob_dict
        dict_arr.has_unique_local_dictionary = unique_dict
        if generate_id:
            # Fetch the length of the data.
            # TODO: Is this too much compilation?
            len_sig = signature(types.int64, sig.args[0])
            nitems = context.compile_internal(
                builder, lambda a: len(a), len_sig, [data]
            )
            new_dict_id = generate_dict_id_codegen(
                context, builder, signature(types.int64, types.int64), [nitems]
            )
            dict_arr.dict_id = new_dict_id
        else:
            dict_arr.dict_id = dict_id

        # increase refcount of stored values
        context.nrt.incref(builder, sig.args[0], data)
        context.nrt.incref(builder, sig.args[1], indices)

        return dict_arr._getvalue()

    ret_typ = DictionaryArrayType(data_t)
    sig = ret_typ(data_t, indices_t, types.bool_, types.bool_, dict_id_if_present_t)
    return sig, codegen


def generate_dict_id_codegen(context, builder, sig, args):
    """Codegen function for generate_dict_id. This is exposed directly
    to enable inlining into other intrinsics.
    """
    (nitems,) = args
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64)])
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="generate_array_id"
    )
    id_args = [nitems]
    new_dict_id = builder.call(fn_tp, id_args)
    return new_dict_id


@intrinsic
def generate_dict_id(typingctx, length_t):
    """Generate a new id for a dictionary with the
    given length. This is exposed directly for APIs that can use
    caching.

    Args:
        length_t (types.int64): The length of the array.

    Returns:
        types.int64: The new dict id.
    """
    assert length_t == types.int64, "Length must be types.int64"
    return types.int64(length_t), generate_dict_id_codegen


@typeof_impl.register(pa.DictionaryArray)
def typeof_dict_value(val, c):
    # only support dict-encoded string arrays for now, TODO(ehsan): support other types
    if val.type.value_type == pa.string() or val.type.value_type == pa.large_string():
        return dict_str_arr_type


def to_pa_dict_arr(A):
    """convert array 'A' to a PyArrow dictionary-encoded array if it is not already.
    'A' can be a Pandas or Numpy array
    """
    if isinstance(A, pa.DictionaryArray):
        return A

    # avoid calling pd.array() for dict-encoded data since the all-null case fails in
    # Arrow, see test_basic.py::test_dict_scalar_to_array
    if (
        isinstance(A, pd.arrays.ArrowStringArray)
        and pa.types.is_dictionary(A._pa_array.type)
        and (
            pa.types.is_string(A._pa_array.type.value_type)
            or pa.types.is_large_string(A._pa_array.type.value_type)
        )
        and pa.types.is_int32(A._pa_array.type.index_type)
    ):
        return A._pa_array.combine_chunks()

    return pd.array(A, "string[pyarrow]")._pa_array.combine_chunks().dictionary_encode()


@unbox(DictionaryArrayType)
def unbox_dict_arr(typ, val, c):
    """
    Unbox a PyArrow dictionary array of string values.
    Simple unboxing to enable testing with PyArrow arrays for now.
    TODO(ehsan): improve performance by copying buffers directly in C++
    """

    # make sure input is a PyArrow dictionary array
    # bodo.hiframes.boxing._use_dict_str_type=True types regular string arrays as
    # dict-encoded arrays.
    # Also, Bodo boxes dict-encoded arrays as Pandas ArrowStringArray which can be
    # passed back into JIT.
    to_pa_dict_arr_obj = c.pyapi.unserialize(c.pyapi.serialize_object(to_pa_dict_arr))
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
    pd_class_obj = c.pyapi.import_module(mod_name)
    int32_str_obj = c.pyapi.string_from_constant_string("Int32")
    pd_int_arr_obj = c.pyapi.call_method(
        pd_class_obj, "array", (indices_obj, int32_str_obj)
    )
    dict_arr.indices = c.unbox(dict_indices_arr_type, pd_int_arr_obj).value
    # assume dictionaries are not the same across all ranks to be conservative
    dict_arr.has_global_dictionary = c.context.get_constant(types.bool_, False)
    # assume dictionaries are not unique to be conservative
    dict_arr.has_unique_local_dictionary = c.context.get_constant(types.bool_, False)
    # Fetch the length of the dictionary.
    nitems_obj = c.pyapi.call_method(np_str_arr_obj, "__len__", [])
    nitems = c.unbox(types.int64, nitems_obj).value
    # Generate a new id for this dictionary
    new_dict_id = generate_dict_id_codegen(
        c.context, c.builder, signature(types.int64, types.int64), [nitems]
    )
    dict_arr.dict_id = new_dict_id

    c.pyapi.decref(data_obj)
    c.pyapi.decref(nitems_obj)
    c.pyapi.decref(false_obj)
    c.pyapi.decref(np_str_arr_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(int32_str_obj)
    c.pyapi.decref(pd_int_arr_obj)

    # decref since val is output of to_pa_dict_arr() and not coming from user context
    c.pyapi.decref(val)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(dict_arr._getvalue(), is_error=is_error)


@box(DictionaryArrayType)
def box_dict_arr(typ, val, c):
    """box dict array into numpy array of string objects"""
    dict_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    if typ == dict_str_arr_type:
        # box to Pandas ArrowStringArray to minimize boxing overhead
        from bodo.libs.array import array_info_type, array_to_info_codegen

        arr_info = array_to_info_codegen(
            c.context, c.builder, array_info_type(typ), (val,)
        )
        fnty = lir.FunctionType(
            c.pyapi.pyobj,
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
            ],
        )
        box_fname = "pd_pyarrow_array_from_string_array"
        fn_get = cgutils.get_or_insert_function(c.builder.module, fnty, name=box_fname)
        arr = c.builder.call(
            fn_get,
            [arr_info, c.context.get_constant(types.int32, 0)],
        )
        c.context.nrt.decref(c.builder, typ, val)
        return arr

    else:
        # create a PyArrow dictionary array from indices and data
        # pa.DictionaryArray.from_arrays(dict_arr.data, dict_arr.indices)
        mod_name = c.context.insert_const_string(c.builder.module, "pyarrow")
        pa_class_obj = c.pyapi.import_module(mod_name)
        pa_dict_arr_class = c.pyapi.object_getattr_string(
            pa_class_obj, "DictionaryArray"
        )
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


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_dict_arr_len(A):
    if isinstance(A, DictionaryArrayType):
        return lambda A: len(A._indices)  # pragma: no cover


@overload_attribute(DictionaryArrayType, "shape", jit_options={"cache": True})
def overload_dict_arr_shape(A):
    return lambda A: (len(A._indices),)  # pragma: no cover


@overload_attribute(DictionaryArrayType, "ndim", jit_options={"cache": True})
def overload_dict_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(DictionaryArrayType, "size", jit_options={"cache": True})
def overload_dict_arr_size(A):
    return lambda A: len(A._indices)  # pragma: no cover


@overload_method(
    DictionaryArrayType, "tolist", no_unliteral=True, jit_options={"cache": True}
)
def overload_dict_arr_tolist(A):
    return lambda A: list(A)  # pragma: no cover


# TODO(ehsan): more optimized version for dictionary-encoded case
overload_method(DictionaryArrayType, "astype", no_unliteral=True)(
    overload_str_arr_astype
)


@overload_method(
    DictionaryArrayType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_dict_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        return init_dict_arr(
            A._data.copy(),
            A._indices.copy(),
            A._has_global_dictionary,
            A._has_unique_local_dictionary,
            A._dict_id,
        )

    return copy_impl


@overload_attribute(DictionaryArrayType, "dtype", jit_options={"cache": True})
def overload_dict_arr_dtype(A):
    return lambda A: A._data.dtype  # pragma: no cover


@overload_attribute(DictionaryArrayType, "nbytes", jit_options={"cache": True})
def dict_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._indices.nbytes  # pragma: no cover


@lower_constant(DictionaryArrayType)
def lower_constant_dict_arr(context, builder, typ, pyval):
    """embed constant dict array value by getting constant values for underlying
    indices and data arrays.
    """
    if isinstance(pyval, pd.arrays.ArrowStringArray):
        pyval = pyval._pa_array.combine_chunks()
        if isinstance(pyval, pa.ChunkedArray):
            pyval = pyval.combine_chunks()
    else:
        if bodo.hiframes.boxing._use_dict_str_type and isinstance(pyval, np.ndarray):
            pyval = pa.array(pyval).dictionary_encode()

    data_arr = pyval.dictionary.to_numpy(False)
    indices_arr = pd.array(pyval.indices, "Int32")

    data_arr = context.get_constant_generic(builder, typ.data, data_arr)
    indices_arr = context.get_constant_generic(
        builder, dict_indices_arr_type, indices_arr
    )

    has_global_dictionary = context.get_constant(types.bool_, False)
    has_unique_local_dictionary = context.get_constant(types.bool_, False)
    # TODO(njriasan): FIXME. We cannot call the C++ function because it doesn't
    # get registered as a global function. There is probably a way to do this
    # with cgutils.global_constant, but that could mess up the id's properties.
    # For now, future work with dict_ids so check to ensure they are valid.
    dict_id = context.get_constant(types.int64, -1)

    dict_array = lir.Constant.literal_struct(
        [
            data_arr,
            indices_arr,
            has_global_dictionary,
            has_unique_local_dictionary,
            dict_id,
        ]
    )
    return dict_array


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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
        A._data,
        A._indices[ind],
        A._has_global_dictionary,
        A._has_unique_local_dictionary,
        A._dict_id,
    )  # pragma: no cover


@overload_method(
    DictionaryArrayType, "_decode", no_unliteral=True, jit_options={"cache": True}
)
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


@overload(operator.setitem, jit_options={"cache": True})
def dict_arr_setitem(A, idx, val):
    if not isinstance(A, DictionaryArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    # Setitem is supported if input array has the same dictionary as target array.
    # NOTE: checking dictionary values to be the same can be slow so the setitem caller
    # needs to make sure this is the case.
    if val == dict_str_arr_type and (
        (
            is_list_like_index_type(idx)
            and (isinstance(idx.dtype, types.Integer) or idx.dtype == types.bool_)
        )
        or isinstance(idx, types.SliceType)
    ):

        def impl_dict_arr_setitem(A, idx, val):  # pragma: no cover
            A._indices[idx] = val._indices

        return impl_dict_arr_setitem

    raise_bodo_error(
        f"DictionaryArrayType setitem not supported for idx type {idx} and value type {val}"
    )


@numba.njit(no_cpython_wrapper=True)
def find_dict_ind_unique(arr, val):  # pragma: no cover
    """find index of 'val' in dictionary of 'arr'. Return -1 if not found.
    Assumes that values in the dictionary are unique.
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
def find_dict_ind_non_unique(arr, val):  # pragma: no cover
    """
    Find indexes of value 'val' in dictionary of 'arr'. Return empty set if not found.
    Does not assume that values in the dictionary are unique.

    Args:
        arr (dictionary encoded array): The array to search
        val (string): The scalar string to search the array for

    Returns:
        Set(int): A set of the indicies. Empty if no matching indicies are found.
    """
    output_set = set()
    data = arr._data
    for i in range(len(data)):
        if bodo.libs.array_kernels.isna(data, i):
            continue
        if data[i] == val:
            output_set.add(i)

    return output_set


@numba.njit(no_cpython_wrapper=True)
def dict_arr_eq(arr, val):  # pragma: no cover
    """implements equality comparison between a dictionary array and a scalar value"""
    n = len(arr)
    if arr._has_unique_local_dictionary:
        dict_ind = find_dict_ind_unique(arr, val)
        if dict_ind == -1:
            # TODO: Add an API for just copying the null bitmap?
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)
            for i in range(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
            return out_arr
        return arr._indices == dict_ind
    else:
        # In this case, we may have multiple indices with a value
        dict_ind_set = find_dict_ind_non_unique(arr, val)

        if len(dict_ind_set) == 0:
            # TODO: Add an API for just copying the null bitmap?
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)
            for i in range(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
            return out_arr

        # TODO: Add an API for just copying the null bitmap?
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in range(n):
            if bodo.libs.array_kernels.isna(arr, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = arr._indices[i] in dict_ind_set
        return out_arr


@numba.njit(no_cpython_wrapper=True)
def dict_arr_ne(arr, val):  # pragma: no cover
    """implements inequality comparison between a dictionary array and a scalar value"""
    n = len(arr)
    if arr._has_unique_local_dictionary:
        # In bodo, if we have a global dictionary, then we know that
        # the values in the dictionary are unique.
        dict_ind = find_dict_ind_unique(arr, val)
        if dict_ind == -1:
            # TODO: Add an API for just copying the null bitmap?
            out_arr = bodo.libs.bool_arr_ext.alloc_true_bool_array(n)
            for i in range(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
            return out_arr
        return arr._indices != dict_ind
    else:
        # In this case, we may have multiple indices with a value
        dict_ind_set = find_dict_ind_non_unique(arr, val)

        if len(dict_ind_set) == 0:
            # TODO: Add an API for just copying the null bitmap?
            out_arr = bodo.libs.bool_arr_ext.alloc_true_bool_array(n)
            for i in range(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
            return out_arr

        # TODO: Add an API for just copying the null bitmap?
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in range(n):
            if bodo.libs.array_kernels.isna(arr, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = arr._indices[i] not in dict_ind_set
        return out_arr


def get_binary_op_overload(op, lhs, rhs):
    """return an optimized implementation for binary operation with dictionary array
    if possible.
    Currently supports only equality and inequality comparisons.
    """
    # NOTE: equality/inequality between two arrays is implemented in str_arr_ext.py
    if op == operator.eq:
        if lhs == dict_str_arr_type and types.unliteral(rhs) == bodo.types.string_type:
            return lambda lhs, rhs: bodo.libs.dict_arr_ext.dict_arr_eq(
                lhs, rhs
            )  # pragma: no cover
        if rhs == dict_str_arr_type and types.unliteral(lhs) == bodo.types.string_type:
            return lambda lhs, rhs: bodo.libs.dict_arr_ext.dict_arr_eq(
                rhs, lhs
            )  # pragma: no cover

    if op == operator.ne:
        if lhs == dict_str_arr_type and types.unliteral(rhs) == bodo.types.string_type:
            return lambda lhs, rhs: bodo.libs.dict_arr_ext.dict_arr_ne(
                lhs, rhs
            )  # pragma: no cover
        if rhs == dict_str_arr_type and types.unliteral(lhs) == bodo.types.string_type:
            return lambda lhs, rhs: bodo.libs.dict_arr_ext.dict_arr_ne(
                rhs, lhs
            )  # pragma: no cover


def convert_dict_arr_to_int(arr, dtype):  # pragma: no cover
    return arr


@overload(convert_dict_arr_to_int, jit_options={"cache": True})
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


def cat_dict_str(arrs, sep):  # pragma: no cover
    pass


@overload(cat_dict_str)
def cat_dict_str_overload(arrs, sep):
    """optimized function for concatenating dictionary array elements one by one.
    Keeps a map of index combinations in input dictionaries to avoid recomputing
    repeated values.
    """
    n_arrs = len(arrs)
    func_text = "def impl(arrs, sep):\n"
    func_text += "  ind_map = {}\n"
    func_text += "  out_strs = []\n"
    func_text += "  n = len(arrs[0])\n"
    for i in range(n_arrs):
        func_text += f"  indices{i} = arrs[{i}]._indices\n"
    for i in range(n_arrs):
        func_text += f"  data{i} = arrs[{i}]._data\n"
    func_text += "  out_indices = bodo.libs.int_arr_ext.alloc_int_array(n, np.int32)\n"
    func_text += "  for i in range(n):\n"
    na_check = " or ".join(
        [f"bodo.libs.array_kernels.isna(arrs[{i}], i)" for i in range(n_arrs)]
    )
    func_text += f"    if {na_check}:\n"
    func_text += "      bodo.libs.array_kernels.setna(out_indices, i)\n"
    func_text += "      continue\n"
    for i in range(n_arrs):
        func_text += f"    ind{i} = indices{i}[i]\n"
    ind_tup = "(" + ", ".join(f"ind{i}" for i in range(n_arrs)) + ")"
    func_text += f"    if {ind_tup} not in ind_map:\n"
    func_text += "      out_ind = len(out_strs)\n"
    func_text += f"      ind_map[{ind_tup}] = out_ind\n"
    sep_str = "''" if is_overload_none(sep) else "sep"
    str_list = ", ".join([f"data{i}[ind{i}]" for i in range(n_arrs)])
    func_text += f"      v = {sep_str}.join([{str_list}])\n"
    func_text += "      out_strs.append(v)\n"
    func_text += "    else:\n"
    func_text += f"      out_ind = ind_map[{ind_tup}]\n"
    func_text += "    out_indices[i] = out_ind\n"
    func_text += (
        "  out_str_arr = bodo.libs.str_arr_ext.str_arr_from_sequence(out_strs)\n"
    )
    func_text += "  return bodo.libs.dict_arr_ext.init_dict_arr(out_str_arr, out_indices, False, False, None)\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "np": np,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


def unset_dict_global(arr):  # pragma: no cover
    pass


@overload(unset_dict_global, jit_options={"cache": True})
def overload_unset_global_dict(arr):
    """Unset global dictionary flag if input is a dictionary-encoded array"""
    if arr != dict_str_arr_type:
        return lambda arr: arr  # pragma: no cover

    def impl_unset_global_dict(arr):  # pragma: no cover
        return init_dict_arr(
            arr._data,
            arr._indices,
            False,
            arr._has_unique_local_dictionary,
            arr._dict_id,
        )

    return impl_unset_global_dict


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


@register_jitable
def dict_arr_to_numeric(arr, errors, downcast):  # pragma: no cover
    """
    Optimized pd.to_numeric() for dictionary-encoded string arrays
    """
    dict_arr = arr._data
    dict_arr_out = pd.to_numeric(dict_arr, errors, downcast)

    # Assign output values from dict_arr_out
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    out_arr = bodo.utils.utils.alloc_type(n_indices, dict_arr_out, (-1,))
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
            continue

        dict_ind = indices_arr[i]
        if bodo.libs.array_kernels.isna(dict_arr_out, dict_ind):
            bodo.libs.array_kernels.setna(out_arr, i)
            continue

        out_arr[i] = dict_arr_out[dict_ind]

    return out_arr


@register_jitable
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

    # NOTE: this operation may introduce non-unique values in the dictionary. Therefore,
    # We have to set _has_unique_local_dictionary to false
    return init_dict_arr(
        out_str_arr, arr._indices.copy(), arr._has_global_dictionary, False, None
    )


@register_jitable
def str_startswith(arr, pat, na):  # pragma: no cover
    """
    Implement optimized string startswith for dictionary array.
    Compute startswith once on each dictionary element.
    """
    # Pandas implementation:
    # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/strings/object_array.py#L131

    # Get the dictionary (info1)
    dict_arr = arr._data
    n_dict = len(dict_arr)
    # Create bool array to store outputs
    dict_arr_out = bodo.libs.bool_arr_ext.alloc_bool_array(n_dict)
    # Compute startswith on the dictionary values
    for i in range(n_dict):
        # We assume there are no NaNs in the dictionary (info1)
        dict_arr_out[i] = dict_arr[i].startswith(pat)

    # Iterate over the array and assign values in the output
    # boolean array from dict_arr_out.
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = dict_arr_out[indices_arr[i]]
    return out_arr


@register_jitable
def str_endswith(arr, pat, na):  # pragma: no cover
    """
    Implement optimized string endswith for dictionary array.
    Compute endswith once on each dictionary element.
    """
    # Pandas implementation:
    # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/strings/object_array.py#L135

    # Get the dictionary (info1)
    dict_arr = arr._data
    n_dict = len(dict_arr)
    # Create bool array to store outputs
    dict_arr_out = bodo.libs.bool_arr_ext.alloc_bool_array(n_dict)
    # Compute endswith on the dictionary values
    for i in range(n_dict):
        # We assume there are no NaNs in the dictionary (info1)
        dict_arr_out[i] = dict_arr[i].endswith(pat)

    # Iterate over the array and assign values in the output
    # boolean array from dict_arr_out.
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = dict_arr_out[indices_arr[i]]
    return out_arr


@numba.njit
def str_series_contains_regex(arr, pat, case, flags, na, regex):  # pragma: no cover
    """
    Equivalent of bodo.hiframes.series_str_impl.series_contains_regex
    but for dictionary encoded string arrays
    """
    ## Compute operation on the dictionary.
    ## This is optimal since ideally this is much shorter than the actual array
    ## and we can save the computation cost.

    # Get the dictionary array (info1)
    dict_arr = arr._data
    # Wrap it in a pandas Series so we can extract a Pandas String array
    # and call pandas' _str_contains on it. Normal boxing will create a
    # numpy object array.
    dict_arr_S = pd.Series(dict_arr)
    # Compute the operation on the dictionary and save the output
    with numba.objmode(dict_arr_out=bodo.types.boolean_array_type):
        dict_arr_out = pd.array(dict_arr_S.array, "string")._str_contains(
            pat, case, flags, na, regex
        )

    ## Create output by indexing into dict_arr_out

    # Get indices (info2)
    indices_arr = arr._indices
    # length of indices == length of str series == length of output
    n_indices = len(indices_arr)
    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)
    # Loop over the indices and get the value from dict_arr_out
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = dict_arr_out[indices_arr[i]]
    return out_arr


@register_jitable
def str_contains_non_regex(arr, pat, case):  # pragma: no cover
    """
    Implement optimized string contains for dictionary array.
    Compute contains once on each dictionary element.
    This is for the non-regex case.
    """
    # Pandas implementation:
    # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/strings/object_array.py#L115

    # Get the dictionary (info1)
    dict_arr = arr._data
    n_dict = len(dict_arr)
    # Create bool array to store outputs
    dict_arr_out = bodo.libs.bool_arr_ext.alloc_bool_array(n_dict)

    if not case:
        upper_pat = pat.upper()

    # Compute contains on the dictionary values
    for i in range(n_dict):
        # We assume there re no NaNs in the dictionary (info1)
        if case:
            dict_arr_out[i] = pat in dict_arr[i]
        else:
            dict_arr_out[i] = upper_pat in dict_arr[i].upper()

    # Iterate over the array and assign values in the output
    # boolean array from dict_arr_out.
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = dict_arr_out[indices_arr[i]]
    return out_arr


@numba.njit
def str_match(arr, pat, case, flags, na, do_full_match=False):  # pragma: no cover
    """Implement optimized string match for dictionary encoded arrays

    Args:
        arr (_type_): Dictionary encoded array
        pat (_type_): Regex pattern
        case (_type_): If True, case sensitive
        flags (_type_): Regex flags
        na (_type_): Fill value for missing values
    """
    dict_arr = arr._data
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)
    dict_arr_S = pd.Series(dict_arr)
    # Compute the operation on the dictionary and save the output
    dict_arr_out = None
    if do_full_match:
        with numba.objmode(dict_arr_out=bodo.types.boolean_array_type):
            dict_arr_out = dict_arr_S.array._str_fullmatch(pat, case, flags, na)
    else:
        with numba.objmode(dict_arr_out=bodo.types.boolean_array_type):
            dict_arr_out = dict_arr_S.array._str_match(pat, case, flags, na)

    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = dict_arr_out[indices_arr[i]]
    return out_arr


def create_simple_str2str_methods(func_name, func_args, can_create_non_unique):
    """
    Returns the dictionary-encoding optimized implementation for
    capitalize, lower, swapcase, title, upper, lstrip, rstrip, strip
    center, ljust, rjust, zfill
    "func_name" is the name of the function to be created, and
    "func_args" is a tuple whose elements are the arguments that the function
    takes in.
    "can_create_non_unique" is a boolean value which indicates if the function
    in question can create non-unique values in output the dictionary array
    For example: func_name = "center", func_args = ("arr", "width", "fillchar")
    """
    func_text = (
        f"def str_{func_name}({', '.join(func_args)}):\n"
        "    data_arr = arr._data\n"
        "    n_data = len(data_arr)\n"
        "    out_str_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n_data, -1)\n"
        "    for i in range(n_data):\n"
        "        if bodo.libs.array_kernels.isna(data_arr, i):\n"
        "            bodo.libs.array_kernels.setna(out_str_arr, i)\n"
        "            continue\n"
        f"        out_str_arr[i] = data_arr[i].{func_name}({', '.join(func_args[1:])})\n"
    )

    if can_create_non_unique:
        func_text += "    return init_dict_arr(out_str_arr, arr._indices.copy(), arr._has_global_dictionary, False, None)\n"
    else:
        func_text += "    return init_dict_arr(out_str_arr, arr._indices.copy(), arr._has_global_dictionary, arr._has_unique_local_dictionary, None)\n"

    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "numba": numba, "init_dict_arr": init_dict_arr},
        loc_vars,
    )
    return loc_vars[f"str_{func_name}"]


def _register_simple_str2str_methods():
    # install simple string to string transformation functions

    # a dictionary that maps function names to function arguments
    args_dict = {
        **dict.fromkeys(
            ["capitalize", "lower", "swapcase", "title", "upper"], ("arr",)
        ),
        **dict.fromkeys(["lstrip", "rstrip", "strip"], ("arr", "to_strip")),
        **dict.fromkeys(["center", "ljust", "rjust"], ("arr", "width", "fillchar")),
        **dict.fromkeys(["zfill"], ("arr", "width")),
    }

    # a dictionary that maps function names to a boolean flag that
    # indicates if the function can create non-unique values in output the dicitonary
    # array
    can_create_unique_dict = {
        **dict.fromkeys(
            [
                "capitalize",
                "lower",
                "title",
                "upper",
                "lstrip",
                "rstrip",
                "strip",
                "center",
                "zfill",
                "ljust",
                "rjust",
            ],
            True,
        ),
        **dict.fromkeys(["swapcase"], False),
    }
    for func_name in args_dict.keys():
        func_impl = create_simple_str2str_methods(
            func_name, args_dict[func_name], can_create_unique_dict[func_name]
        )
        func_impl = register_jitable(func_impl)
        globals()[f"str_{func_name}"] = func_impl


_register_simple_str2str_methods()


@register_jitable
def str_index(arr, sub, start, end):  # pragma: no cover
    """Implement optimized string index for dictionary encoded arrays
    The function will return the index of the first occurrence of
    sub in arr[start, end) if sub is present and raise ValueError instead of -1
     if the value is not found

    Args:
        arr : dictionary encoded array
        sub (str): the substring to search for
        start (int): where to start the search
        end (int): where to end the search
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_data = len(data_arr)
    n_indices = len(indices_arr)
    out_dict_arr = bodo.libs.int_arr_ext.alloc_int_array(n_data, np.int64)
    out_arr = bodo.libs.int_arr_ext.alloc_int_array(n_indices, np.int64)
    error_flag = False
    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            bodo.libs.array_kernels.setna(out_dict_arr, i)
        else:
            out_dict_arr[i] = data_arr[i].find(sub, start, end)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i) or bodo.libs.array_kernels.isna(
            out_dict_arr, indices_arr[i]
        ):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = out_dict_arr[indices_arr[i]]
            if out_arr[i] == -1:
                error_flag = True
    error_message = "substring not found" if error_flag else ""
    synchronize_error_njit("ValueError", error_message)
    return out_arr


@register_jitable
def str_rindex(arr, sub, start, end):  # pragma: no cover
    """Implement optimized string rindex for dictionary encoded arrays
    The function will return the index of the last occurrence of
    sub in arr[start, end) if sub is present and raise ValueError instead of -1
     if the value is not found

    Args:
        arr : dictionary encoded array
        sub (str): the substring to search for
        start (int): where to start the search
        end (int): where to end the search
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_data = len(data_arr)
    n_indices = len(indices_arr)
    out_dict_arr = bodo.libs.int_arr_ext.alloc_int_array(n_data, np.int64)
    out_arr = bodo.libs.int_arr_ext.alloc_int_array(n_indices, np.int64)
    error_flag = False
    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            bodo.libs.array_kernels.setna(out_dict_arr, i)
        else:
            out_dict_arr[i] = data_arr[i].rindex(sub, start, end)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(arr, i) or bodo.libs.array_kernels.isna(
            out_dict_arr, indices_arr[i]
        ):
            bodo.libs.array_kernels.setna(out_arr, i)
        else:
            out_arr[i] = out_dict_arr[indices_arr[i]]
            if out_arr[i] == -1:
                error_flag = True
    error_message = "substring not found" if error_flag else ""
    synchronize_error_njit("ValueError", error_message)
    return out_arr


def create_find_methods(func_name):
    """
    Returns the dictionary-encoding optimized implementation for find, rfind
    Returns lowest/highest indexes on each dictionary element
    """
    func_text = (
        f"def str_{func_name}(arr, sub, start, end):\n"
        "  data_arr = arr._data\n"
        "  indices_arr = arr._indices\n"
        "  n_data = len(data_arr)\n"
        "  n_indices = len(indices_arr)\n"
        "  tmp_dict_arr = bodo.libs.int_arr_ext.alloc_int_array(n_data, np.int64)\n"
        "  out_int_arr = bodo.libs.int_arr_ext.alloc_int_array(n_indices, np.int64)\n"
        # First iterate through the dictionary
        "  for i in range(n_data):\n"
        "    if bodo.libs.array_kernels.isna(data_arr, i):\n"
        "      bodo.libs.array_kernels.setna(tmp_dict_arr, i)\n"
        "      continue\n"
        f"    tmp_dict_arr[i] = data_arr[i].{func_name}(sub, start, end)\n"
        # Populate the output array
        "  for i in range(n_indices):\n"
        "    if bodo.libs.array_kernels.isna(indices_arr, i) or bodo.libs.array_kernels.isna(\n"
        "      tmp_dict_arr, indices_arr[i]\n"
        "    ):\n"
        "      bodo.libs.array_kernels.setna(out_int_arr, i)\n"
        "    else:\n"
        "      out_int_arr[i] = tmp_dict_arr[indices_arr[i]]\n"
        "  return out_int_arr"
    )
    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "numba": numba, "init_dict_arr": init_dict_arr, "np": np},
        loc_vars,
    )
    return loc_vars[f"str_{func_name}"]


def _register_find_methods():
    # install str_find/rfind
    func_names = ["find", "rfind"]
    for func_name in func_names:
        func_impl = create_find_methods(func_name)
        func_impl = register_jitable(func_impl)
        globals()[f"str_{func_name}"] = func_impl


_register_find_methods()


@register_jitable
def str_count(arr, pat, flags):  # pragma: no cover
    """
    Implement optimized string count for dictionary array
    Count the number of occurrences of pattern in each string
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_data = len(data_arr)
    n_indices = len(indices_arr)
    out_dict_arr = bodo.libs.int_arr_ext.alloc_int_array(n_data, np.int64)
    out_int_arr = bodo.libs.int_arr_ext.alloc_int_array(n_indices, np.int64)
    regex = re.compile(pat, flags)
    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            bodo.libs.array_kernels.setna(out_dict_arr, i)
            continue
        out_dict_arr[i] = bodo.libs.str_ext.str_findall_count(regex, data_arr[i])
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(indices_arr, i) or bodo.libs.array_kernels.isna(
            out_dict_arr, indices_arr[i]
        ):
            bodo.libs.array_kernels.setna(out_int_arr, i)
        else:
            out_int_arr[i] = out_dict_arr[indices_arr[i]]
    return out_int_arr


@register_jitable
def str_len(arr):  # pragma: no cover
    """
    Implement optimized string len for dictionary array
    Return the length of each string
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_indices = len(indices_arr)
    # na_empty_as_one is set to false as we want to
    # manually set na.
    out_dict_arr = bodo.libs.array_kernels.get_arr_lens(data_arr, False)
    out_int_arr = bodo.libs.int_arr_ext.alloc_int_array(n_indices, np.int64)
    for i in range(n_indices):
        if bodo.libs.array_kernels.isna(indices_arr, i) or bodo.libs.array_kernels.isna(
            out_dict_arr, indices_arr[i]
        ):
            bodo.libs.array_kernels.setna(out_int_arr, i)
        else:
            out_int_arr[i] = out_dict_arr[indices_arr[i]]
    return out_int_arr


@register_jitable
def str_slice(arr, start, stop, step):  # pragma: no cover
    """
    Implement optimized string slice for dictionary array.
    Slice substrings from each element in the dictionary.
    """
    data_arr = arr._data
    n_data = len(data_arr)
    out_str_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n_data, -1)
    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            bodo.libs.array_kernels.setna(out_str_arr, i)
            continue
        out_str_arr[i] = data_arr[i][start:stop:step]

    # Slice can result in duplicate values in the data array, so we have to set
    # _has_unique_local_dictionary to False in the output
    return init_dict_arr(
        out_str_arr, arr._indices.copy(), arr._has_global_dictionary, False, None
    )


@register_jitable
def str_get(arr, i):  # pragma: no cover
    """
    Implement optimized string get for dictionary array
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_data = len(data_arr)
    n_indices = len(indices_arr)
    out_str_arr = pre_alloc_string_array(n_data, -1)
    out_arr = pre_alloc_string_array(n_indices, -1)
    for j in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, j) or not (
            -len(data_arr[j]) <= i < len(data_arr[j])
        ):
            bodo.libs.array_kernels.setna(out_str_arr, j)
            continue
        out_str_arr[j] = data_arr[j][i]

    for j in range(n_indices):
        if bodo.libs.array_kernels.isna(indices_arr, j) or bodo.libs.array_kernels.isna(
            out_str_arr, indices_arr[j]
        ):
            bodo.libs.array_kernels.setna(out_arr, j)
            continue
        out_arr[j] = out_str_arr[indices_arr[j]]
    return out_arr


@register_jitable
def str_repeat_int(arr, repeats):  # pragma: no cover
    """
    Implement string repeat for dictionary array
    when repeats is integer
    """
    data_arr = arr._data
    n_data = len(data_arr)
    out_str_arr = pre_alloc_string_array(n_data, -1)

    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            bodo.libs.array_kernels.setna(out_str_arr, i)
            continue
        out_str_arr[i] = data_arr[i] * repeats

    # NOTE: _has_unique_local_dictionary must be false in the case that repeats is 0, as we would generate copies in the data array
    return init_dict_arr(
        out_str_arr,
        arr._indices.copy(),
        arr._has_global_dictionary,
        arr._has_unique_local_dictionary and repeats != 0,
        None,
    )


def create_str2bool_methods(func_name):
    """
    Returns the dictionary-encoding optimized implementation for
    isalnum, isalpha, isdigit, isspae, islower, isupper, istitle, isnumeric, isdecimal
    """
    func_text = (
        f"def str_{func_name}(arr):\n"
        "    data_arr = arr._data\n"
        "    indices_arr = arr._indices\n"
        "    n_data = len(data_arr)\n"
        "    n_indices = len(indices_arr)\n"
        "    out_dict_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_data)\n"
        "    out_bool_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_indices)\n"
        "    for i in range(n_data):\n"
        "        if bodo.libs.array_kernels.isna(data_arr, i):\n"
        "            bodo.libs.array_kernels.setna(out_dict_arr, i)\n"
        "            continue\n"
        f"        out_dict_arr[i] = np.bool_(data_arr[i].{func_name}())\n"
        "    for i in range(n_indices):\n"
        "        if bodo.libs.array_kernels.isna(indices_arr, i) or bodo.libs.array_kernels.isna(\n"
        "            data_arr, indices_arr[i]"
        "        ):\n"
        "            bodo.libs.array_kernels.setna(out_bool_arr, i)\n"
        "        else:\n"
        "            out_bool_arr[i] = out_dict_arr[indices_arr[i]]\n"
        "    return out_bool_arr"
    )

    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "numba": numba, "np": np, "init_dict_arr": init_dict_arr},
        loc_vars,
    )
    return loc_vars[f"str_{func_name}"]


def _register_str2bool_methods():
    # install str2bool functions
    for func_name in bodo.hiframes.pd_series_ext.str2bool_methods:
        func_impl = create_str2bool_methods(func_name)
        func_impl = register_jitable(func_impl)
        globals()[f"str_{func_name}"] = func_impl


_register_str2bool_methods()


@register_jitable
def str_extract(arr, pat, flags, n_cols):  # pragma: no cover
    """
    Implement optimized string extract for dictionary encoded array
    Return a list of dictionary encoded arrays where each array
    represents a capture group
    """
    data_arr = arr._data
    indices_arr = arr._indices
    n_data = len(data_arr)
    n_indices = len(indices_arr)
    regex = re.compile(pat, flags=flags)
    # a list consists of the dictionary arrays for each output column
    out_dict_arr_list = []
    for _ in range(n_cols):
        out_dict_arr_list.append(pre_alloc_string_array(n_data, -1))
    is_out_na_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n_data)
    out_indices_arr = indices_arr.copy()
    for i in range(n_data):
        if bodo.libs.array_kernels.isna(data_arr, i):
            is_out_na_arr[i] = True
            for j in range(n_cols):
                bodo.libs.array_kernels.setna(out_dict_arr_list[j], i)
            continue
        m = regex.search(data_arr[i])
        if m:
            is_out_na_arr[i] = False
            g = m.groups()
            for j in range(n_cols):
                out_dict_arr_list[j][i] = g[j]
        else:
            is_out_na_arr[i] = True
            for j in range(n_cols):
                bodo.libs.array_kernels.setna(out_dict_arr_list[j], i)

    for i in range(n_indices):
        if is_out_na_arr[out_indices_arr[i]]:
            bodo.libs.array_kernels.setna(out_indices_arr, i)

    out_arr_list = [
        # Note: extract can return duplicate values, so we have to
        # set _has_unique_local_dictionary=False
        init_dict_arr(
            out_dict_arr_list[i],
            out_indices_arr.copy(),
            arr._has_global_dictionary,
            False,
            None,
        )
        for i in range(n_cols)
    ]

    return out_arr_list


def create_extractall_methods(is_multi_group):
    """Returns the dictionary-encoding optimized implementations for
    extractall

    Args:
        is_multi_group (bool): True if the regex pattern consists
        of multiple groups, False otherwise

        For example,
        pd.Series(["чьь1т33", "ьнн2с222", "странаст2", np.nan, "ьнне33ст3"] * 2,
        ["е3", "не3", "н2с2", "AA", "C"] * 2).str.extractall(r"([чен]+)\\d+([ст]+)\\d+")
        will invoke the implementation with is_multi_group set to true since
        the regex expression has two capture groups.
        On the other hand,
        pd.Series(["a1b1", "b1", np.nan, "a2", "c2", "ddd", "dd4d1", "d22c2"],
        [4, 3, 5, 1, 0, 2, 6, 11]).str.extractall(r"(?P<BBB>[abd]+)\\d+")
        will invoke the implementation with is_multi_group set to False,
        as the regex consists of only one capture group.
    """
    # Two implementations will be generated:
    # str_extractall: invoked when regex has single capture group
    # str_extractall_multi: invoked when regex regex multiple capture groups
    multi_group = "_multi" if is_multi_group else ""
    func_text = (
        f"def str_extractall{multi_group}(arr, regex, n_cols, index_arr):\n"
        "    data_arr = arr._data\n"
        "    indices_arr = arr._indices\n"
        "    n_data = len(data_arr)\n"
        "    n_indices = len(indices_arr)\n"
        "    indices_count = [0 for _ in range(n_data)]\n"
        "    for i in range(n_indices):\n"
        "        if not bodo.libs.array_kernels.isna(indices_arr, i):\n"
        "            indices_count[indices_arr[i]] += 1\n"
        "    dict_group_count = []\n"
        # calculate the total number of rows
        # out_dict_len: the length of the output dictionary array
        # out_ind_len: the length of the output indices array
        "    out_dict_len = out_ind_len = 0\n"
        # the first for-loop is for calculating the number of matches
        # of each string
        "    for i in range(n_data):\n"
        "        if bodo.libs.array_kernels.isna(data_arr, i):\n"
        "            continue\n"
        "        m = regex.findall(data_arr[i])\n"
        # len(m): the number of matches for each string
        # dic_group_count[i] maps the old dict index to
        # its new position and length
        "        dict_group_count.append((out_dict_len, len(m)))\n"
        "        out_dict_len += len(m)\n"
        "        out_ind_len += indices_count[i] * len(m)\n"
        "    out_dict_arr_list = []\n"
        "    for _ in range(n_cols):\n"
        "        out_dict_arr_list.append(pre_alloc_string_array(out_dict_len, -1))\n"
        "    out_indices_arr = bodo.libs.int_arr_ext.alloc_int_array(out_ind_len, np.int32)\n"
        "    out_ind_arr = bodo.utils.utils.alloc_type(out_ind_len, index_arr, (-1,))\n"
        "    out_match_arr = np.empty(out_ind_len, np.int64)\n"
        # the second for-loop is for generating the dictionary arrays
        "    curr_ind = 0\n"
        "    for i in range(n_data):\n"
        "        if bodo.libs.array_kernels.isna(data_arr, i):\n"
        "            continue\n"
        "        m = regex.findall(data_arr[i])\n"
        "        for s in m:\n"
        "            for j in range(n_cols):\n"
        f"                out_dict_arr_list[j][curr_ind] = s{'[j]' if is_multi_group else ''}\n"
        "            curr_ind += 1\n"
        # the third for-loop is for populating the index and match arrays
        "    curr_ind = 0\n"
        "    for i in range(n_indices):\n"
        "        if bodo.libs.array_kernels.isna(indices_arr, i):\n"
        "            continue\n"
        "        n_rows = dict_group_count[indices_arr[i]][1]\n"
        "        for k in range(n_rows):\n"
        "            out_indices_arr[curr_ind] = dict_group_count[indices_arr[i]][0] + k\n"
        "            out_ind_arr[curr_ind] = index_arr[i]\n"
        "            out_match_arr[curr_ind] = k\n"
        "            curr_ind += 1\n"
        "    out_arr_list = [\n"
        # Note: This operation can produce duplicate values.
        "        init_dict_arr(\n"
        "            out_dict_arr_list[i], out_indices_arr.copy(), arr._has_global_dictionary, False, None\n"
        "        )\n"
        "        for i in range(n_cols)\n"
        "    ]\n"
        "    return (out_ind_arr, out_match_arr, out_arr_list) \n"
    )

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "np": np,
            "init_dict_arr": init_dict_arr,
            "pre_alloc_string_array": pre_alloc_string_array,
        },
        loc_vars,
    )
    return loc_vars[f"str_extractall{multi_group}"]


def _register_extractall_methods():
    # install various implementations for extractall
    for is_multi_group in [True, False]:
        multi_group = "_multi" if is_multi_group else ""
        func_impl = create_extractall_methods(is_multi_group)
        func_impl = register_jitable(func_impl)
        globals()[f"str_extractall{multi_group}"] = func_impl


_register_extractall_methods()


@generated_jit(nopython=True)
def is_dict_encoded(t):
    """This is a testing utility, that can be used to check if a given array/series
    is dict encoded. This should never be called internally in the engine,
    you should just check the type at compile time."""
    t = if_series_to_array_type(t)

    if t == bodo.types.dict_str_arr_type:

        def impl(t):
            return True

    else:
        assert t == bodo.types.string_array_type

        def impl(t):
            return False

    return impl
