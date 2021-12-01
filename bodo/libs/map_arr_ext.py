# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for map values.
Corresponds to Spark's MapType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Map arrays: https://github.com/apache/arrow/blob/master/format/Schema.fbs

The implementation uses an array(struct) array underneath similar to Spark and Arrow.
For example: [{1: 2.1, 3: 1.1}, {5: -1.0}]
[[{"key": 1, "value" 2.1}, {"key": 3, "value": 1.1}], [{"key": 5, "value": -1.0}]]
"""
import operator

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    offset_type,
)
from bodo.libs.struct_arr_ext import StructArrayType, _get_struct_arr_payload
from bodo.utils.cg_helpers import (
    dict_keys,
    dict_merge_from_seq2,
    dict_values,
    gen_allocate_array,
    get_array_elem_counts,
    get_bitmap_bit,
    is_na_value,
    pyarray_setitem,
    seq_getitem,
    set_bitmap_bit,
)
from bodo.utils.typing import BodoError

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import array_ext, hdist  # isort:skip

ll.add_symbol("count_total_elems_list_array", array_ext.count_total_elems_list_array)
ll.add_symbol("map_array_from_sequence", array_ext.map_array_from_sequence)
ll.add_symbol("np_array_from_map_array", array_ext.np_array_from_map_array)


class MapArrayType(types.ArrayCompatible):
    """Data type for arrays of maps"""

    def __init__(self, key_arr_type, value_arr_type):
        self.key_arr_type = key_arr_type
        self.value_arr_type = value_arr_type
        super(MapArrayType, self).__init__(
            name="MapArrayType({}, {})".format(key_arr_type, value_arr_type)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.DictType(self.key_arr_type.dtype, self.value_arr_type.dtype)

    def copy(self):
        return MapArrayType(self.key_arr_type, self.value_arr_type)


def _get_map_arr_data_type(map_type):
    """get array(struct) array data type for underlying data array of map type"""
    struct_arr_type = StructArrayType(
        (map_type.key_arr_type, map_type.value_arr_type), ("key", "value")
    )
    return ArrayItemArrayType(struct_arr_type)


@register_model(MapArrayType)
class MapArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # storing a array(struct) array as data without a separate payload since it has
        # a payload and supports inplace update so there is no need for another payload
        data_arr_type = _get_map_arr_data_type(fe_type)
        members = [
            ("data", data_arr_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(MapArrayType, "data", "_data")


@unbox(MapArrayType)
def unbox_map_array(typ, val, c):
    """
    Unbox a numpy array with dictionary values.
    """
    # get length
    n_maps = bodo.utils.utils.object_length(c, val)

    # can be handled in C if key/value arrays are Numpy and in handled dtypes
    handle_in_c = all(
        isinstance(t, types.Array)
        and t.dtype
        in (
            types.int64,
            types.float64,
            types.bool_,
            datetime_date_type,
        )
        for t in (typ.key_arr_type, typ.value_arr_type)
    )

    if handle_in_c:
        fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            c.builder.module, fnty, name="count_total_elems_list_array"
        )
        n_elems_all = cgutils.pack_array(
            c.builder, [n_maps, c.builder.call(fn_tp, [val])]
        )
    else:
        n_elems_all = get_array_elem_counts(c, c.builder, c.context, val, typ)

    # allocate data array
    data_arr_type = _get_map_arr_data_type(typ)
    data_arr = gen_allocate_array(c.context, c.builder, data_arr_type, n_elems_all, c)
    data_payload = _get_array_item_arr_payload(
        c.context, c.builder, data_arr_type, data_arr
    )

    # get null and offset array pointers to pass to unboxing
    null_bitmap_ptr = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, data_payload.null_bitmap
    ).data
    offsets_ptr = c.context.make_array(types.Array(offset_type, 1, "C"))(
        c.context, c.builder, data_payload.offsets
    ).data

    # get key/value array pointers to pass to C
    struct_arr_payload = _get_struct_arr_payload(
        c.context, c.builder, data_arr_type.dtype, data_payload.data
    )
    key_arr = c.builder.extract_value(struct_arr_payload.data, 0)
    value_arr = c.builder.extract_value(struct_arr_payload.data, 1)

    # set all values to not NA in struct since NA not possible
    sig = types.none(types.Array(types.uint8, 1, "C"))
    _is_error, _res = c.pyapi.call_jit_code(
        lambda A: A.fill(255), sig, [struct_arr_payload.null_bitmap]
    )

    if handle_in_c:
        key_arr_ptr = c.context.make_array(data_arr_type.dtype.data[0])(
            c.context, c.builder, key_arr
        ).data
        value_arr_ptr = c.context.make_array(data_arr_type.dtype.data[1])(
            c.context, c.builder, value_arr
        ).data

        # function signature of map_array_from_sequence
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # obj
                lir.IntType(8).as_pointer(),  # key data
                lir.IntType(8).as_pointer(),  # value data
                lir.IntType(offset_type.bitwidth).as_pointer(),  # offsets
                lir.IntType(8).as_pointer(),  # null_bitmap
                lir.IntType(32),  # key ctype
                lir.IntType(32),  # value ctype
            ],
        )
        fn = cgutils.get_or_insert_function(
            c.builder.module, fnty, name="map_array_from_sequence"
        )

        key_ctype = bodo.utils.utils.numba_to_c_type(typ.key_arr_type.dtype)
        value_ctype = bodo.utils.utils.numba_to_c_type(typ.value_arr_type.dtype)
        c.builder.call(
            fn,
            [
                val,
                c.builder.bitcast(key_arr_ptr, lir.IntType(8).as_pointer()),
                c.builder.bitcast(value_arr_ptr, lir.IntType(8).as_pointer()),
                offsets_ptr,
                null_bitmap_ptr,
                lir.Constant(lir.IntType(32), key_ctype),
                lir.Constant(lir.IntType(32), value_ctype),
            ],
        )
    else:
        _unbox_map_array_generic(
            typ, val, c, n_maps, key_arr, value_arr, offsets_ptr, null_bitmap_ptr
        )

    map_array = c.context.make_helper(c.builder, typ)
    map_array.data = data_arr

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(map_array._getvalue(), is_error=is_error)


def _unbox_map_array_generic(
    typ, val, c, n_maps, key_arr, value_arr, offsets_ptr, null_bitmap_ptr
):
    """unbox map array using generic Numba unboxing to handle all item types that can be
    unboxed.
    """
    from bodo.libs.array_item_arr_ext import _unbox_array_item_array_copy_data

    # TODO: refactor to avoid duplication with _unbox_array_item_array_generic
    context = c.context
    builder = c.builder
    # TODO: error checking for pyapi calls
    # get pd.NA object to check for new NA kind
    mod_name = context.insert_const_string(builder.module, "pandas")
    pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
    C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

    zero_offset = c.context.get_constant(offset_type, 0)
    builder.store(zero_offset, offsets_ptr)

    # pseudocode for code generation:
    # curr_item_ind = 0
    # for i in range(len(A)):
    #   offsets[i] = curr_item_ind
    #   dict_obj = A[i]
    #   if isna(dict_obj):
    #     set null_bitmap i'th bit to 0
    #   else:
    #     set null_bitmap i'th bit to 1
    #     key_list_obj = dict_keys(dict_obj)
    #     value_list_obj = dict_values(dict_obj)
    #     n_items = len(key_list_obj)
    #     unbox(key_list_obj) and copy data to key_arr
    #     unbox(value_list_obj) and copy data to value_arr
    #     curr_item_ind += n_items
    # offsets[n] = curr_item_ind;

    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(
        builder, context.get_constant(types.int64, 0)
    )
    # for each array
    with cgutils.for_range(builder, n_maps) as loop:
        dict_ind = loop.index
        item_ind = builder.load(curr_item_ind)

        # offsets[i] = curr_item_ind
        builder.store(
            builder.trunc(item_ind, lir.IntType(offset_type.bitwidth)),
            builder.gep(offsets_ptr, [dict_ind]),
        )
        # dict_obj = A[i]
        dict_obj = seq_getitem(builder, context, val, dict_ind)
        # set NA bit to 0
        set_bitmap_bit(builder, null_bitmap_ptr, dict_ind, 0)

        # check for NA
        is_na = is_na_value(builder, context, dict_obj, C_NA)
        is_na_cond = builder.icmp_unsigned("!=", is_na, lir.Constant(is_na.type, 1))
        with builder.if_then(is_na_cond):
            # set NA bit to 1
            set_bitmap_bit(builder, null_bitmap_ptr, dict_ind, 1)
            key_list_obj = dict_keys(builder, context, dict_obj)
            value_list_obj = dict_values(builder, context, dict_obj)
            n_items = bodo.utils.utils.object_length(c, key_list_obj)
            _unbox_array_item_array_copy_data(
                typ.key_arr_type, key_list_obj, c, key_arr, item_ind, n_items
            )
            _unbox_array_item_array_copy_data(
                typ.value_arr_type, value_list_obj, c, value_arr, item_ind, n_items
            )
            # curr_item_ind += n_items
            builder.store(builder.add(item_ind, n_items), curr_item_ind)
            c.pyapi.decref(key_list_obj)
            c.pyapi.decref(value_list_obj)
        c.pyapi.decref(dict_obj)

    # offsets[n] = curr_item_ind;
    builder.store(
        builder.trunc(builder.load(curr_item_ind), lir.IntType(offset_type.bitwidth)),
        builder.gep(offsets_ptr, [n_maps]),
    )

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)


@box(MapArrayType)
def box_map_arr(typ, val, c):
    """box packed native representation of map array into python objects"""
    map_array = c.context.make_helper(c.builder, typ, val)
    data_arr = map_array.data

    # allocate data array
    data_arr_type = _get_map_arr_data_type(typ)
    data_payload = _get_array_item_arr_payload(
        c.context, c.builder, data_arr_type, data_arr
    )

    # get null and offset array pointers to pass to boxing
    null_bitmap_ptr = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, data_payload.null_bitmap
    ).data
    offsets_ptr = c.context.make_array(types.Array(offset_type, 1, "C"))(
        c.context, c.builder, data_payload.offsets
    ).data

    # get key/value arrays
    struct_arr_payload = _get_struct_arr_payload(
        c.context, c.builder, data_arr_type.dtype, data_payload.data
    )
    key_arr = c.builder.extract_value(struct_arr_payload.data, 0)
    value_arr = c.builder.extract_value(struct_arr_payload.data, 1)

    # use C boxing when possible to avoid compilation and runtime overheads
    # otherwise, use generic llvm/Numba boxing
    if all(
        isinstance(t, types.Array)
        and t.dtype in (types.int64, types.float64, types.bool_, datetime_date_type)
        for t in (typ.key_arr_type, typ.value_arr_type)
    ):

        key_arr_ptr = c.context.make_array(data_arr_type.dtype.data[0])(
            c.context, c.builder, key_arr
        ).data
        value_arr_ptr = c.context.make_array(data_arr_type.dtype.data[1])(
            c.context, c.builder, value_arr
        ).data

        fnty = lir.FunctionType(
            c.context.get_argument_type(types.pyobject),
            [
                lir.IntType(64),  # num_maps
                lir.IntType(8).as_pointer(),  # key data
                lir.IntType(8).as_pointer(),  # value data
                lir.IntType(offset_type.bitwidth).as_pointer(),  # offsets
                lir.IntType(8).as_pointer(),  # null_bitmap
                lir.IntType(32),  # key ctype
                lir.IntType(32),  # value ctype
            ],
        )
        fn_get = cgutils.get_or_insert_function(
            c.builder.module, fnty, name="np_array_from_map_array"
        )

        key_ctype = bodo.utils.utils.numba_to_c_type(typ.key_arr_type.dtype)
        value_ctype = bodo.utils.utils.numba_to_c_type(typ.value_arr_type.dtype)
        arr = c.builder.call(
            fn_get,
            [
                data_payload.n_arrays,
                c.builder.bitcast(key_arr_ptr, lir.IntType(8).as_pointer()),
                c.builder.bitcast(value_arr_ptr, lir.IntType(8).as_pointer()),
                offsets_ptr,
                null_bitmap_ptr,
                lir.Constant(lir.IntType(32), key_ctype),
                lir.Constant(lir.IntType(32), value_ctype),
            ],
        )

    else:
        arr = _box_map_array_generic(
            typ,
            c,
            data_payload.n_arrays,
            key_arr,
            value_arr,
            offsets_ptr,
            null_bitmap_ptr,
        )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


def _box_map_array_generic(
    typ, c, n_maps, key_arr, value_arr, offsets_ptr, null_bitmap_ptr
):
    """box map array using generic Numba boxing to handle all item types that can be
    boxed.
    """
    # TODO: refactor with _box_array_item_array_generic
    context = c.context
    builder = c.builder
    # TODO: error checking for pyapi calls

    # pseudocode for code generation:
    # out_arr = np.ndarray(n, np.object_)
    # curr_item_ind = 0
    # for i in range(n):
    #   if isna(A[i]):
    #     out_arr[i] = np.nan
    #   else:
    #     n_items = offsets[i + 1] - offsets[i]
    #     dict_obj = dict_new(n_items)
    #     for j in range(n_items):
    #        key = A.data[0][curr_item_ind]
    #        value = A.data[1][curr_item_ind]
    #        dict_obj[j][key] = value
    #        curr_item_ind += 1
    #     A[i] = dict_obj

    # create array of objects with num_items shape
    mod_name = context.insert_const_string(builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype_obj = c.pyapi.object_getattr_string(np_class_obj, "object_")
    num_items_obj = c.pyapi.long_from_longlong(n_maps)
    out_arr = c.pyapi.call_method(np_class_obj, "ndarray", (num_items_obj, dtype_obj))
    # get np.nan to set NA
    nan_obj = c.pyapi.object_getattr_string(np_class_obj, "nan")

    zip_func_obj = c.pyapi.unserialize(c.pyapi.serialize_object(zip))

    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(64), 0))
    # for each map
    with cgutils.for_range(builder, n_maps) as loop:
        map_ind = loop.index
        # A[i] = np.nan
        pyarray_setitem(builder, context, out_arr, map_ind, nan_obj)
        # check for NA
        na_bit = get_bitmap_bit(builder, null_bitmap_ptr, map_ind)
        not_na_cond = builder.icmp_unsigned(
            "!=", na_bit, lir.Constant(lir.IntType(8), 0)
        )
        with builder.if_then(not_na_cond):
            # n_items = offsets[i + 1] - offsets[i]
            n_items = builder.sext(
                builder.sub(
                    builder.load(
                        builder.gep(
                            offsets_ptr,
                            [builder.add(map_ind, lir.Constant(map_ind.type, 1))],
                        )
                    ),
                    builder.load(builder.gep(offsets_ptr, [map_ind])),
                ),
                lir.IntType(64),
            )
            # create dict obj
            item_ind = builder.load(curr_item_ind)
            dict_obj = c.pyapi.dict_new()
            # box key and value arrays
            f = lambda data_arr, item_ind, n_items: data_arr[
                item_ind : item_ind + n_items
            ]
            _is_error, key_arr_slice = c.pyapi.call_jit_code(
                f,
                (typ.key_arr_type)(typ.key_arr_type, types.int64, types.int64),
                [key_arr, item_ind, n_items],
            )
            _is_error, value_arr_slice = c.pyapi.call_jit_code(
                f,
                (typ.value_arr_type)(typ.value_arr_type, types.int64, types.int64),
                [value_arr, item_ind, n_items],
            )
            key_arr_obj = c.pyapi.from_native_value(
                typ.key_arr_type, key_arr_slice, c.env_manager
            )
            value_arr_obj = c.pyapi.from_native_value(
                typ.value_arr_type, value_arr_slice, c.env_manager
            )

            # get zip(keys, values) iterator
            key_value_iter_obj = c.pyapi.call_function_objargs(
                zip_func_obj, (key_arr_obj, value_arr_obj)
            )
            # dict.update(zip_arr_iter)
            dict_merge_from_seq2(builder, context, dict_obj, key_value_iter_obj)

            builder.store(builder.add(item_ind, n_items), curr_item_ind)
            pyarray_setitem(builder, context, out_arr, map_ind, dict_obj)
            c.pyapi.decref(key_value_iter_obj)
            c.pyapi.decref(key_arr_obj)
            c.pyapi.decref(value_arr_obj)
            c.pyapi.decref(dict_obj)

    c.pyapi.decref(zip_func_obj)
    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(num_items_obj)
    c.pyapi.decref(nan_obj)
    return out_arr


def init_map_arr_codegen(context, builder, sig, args):
    """
    Codegen function for Map Arrays. This used by init_map_arr
    and instrinsics that cannot directly call init_map_arr
    """
    (data_arr,) = args
    map_array = context.make_helper(builder, sig.return_type)
    map_array.data = data_arr
    context.nrt.incref(builder, sig.args[0], data_arr)
    return map_array._getvalue()


@intrinsic
def init_map_arr(typingctx, data_typ=None):
    """create a new map array from input data list(struct) array data"""
    assert isinstance(data_typ, ArrayItemArrayType) and isinstance(
        data_typ.dtype, StructArrayType
    )
    map_arr_type = MapArrayType(data_typ.dtype.data[0], data_typ.dtype.data[1])
    return map_arr_type(data_typ), init_map_arr_codegen


def alias_ext_init_map_arr(lhs_name, args, alias_map, arg_aliases):
    """
    Aliasing for init_map_arr function.
    """
    assert len(args) == 1
    # Data is stored inside map_arr struct so it should alias
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_map_arr", "bodo.libs.map_arr_ext")
] = alias_ext_init_map_arr


@numba.njit
def pre_alloc_map_array(num_maps, nested_counts, struct_typ):
    data = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
        num_maps, nested_counts, struct_typ
    )
    return init_map_arr(data)


def pre_alloc_map_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_map_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 3 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_map_arr_ext_pre_alloc_map_array = (
    pre_alloc_map_array_equiv
)


@overload(len, no_unliteral=True)
def overload_map_arr_len(A):
    if isinstance(A, MapArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(MapArrayType, "shape")
def overload_map_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(MapArrayType, "dtype")
def overload_map_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@overload_attribute(MapArrayType, "ndim")
def overload_map_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(MapArrayType, "nbytes")
def overload_map_arr_nbytes(A):
    return lambda A: A._data.nbytes  # pragma: no cover


@overload_method(MapArrayType, "copy")
def overload_map_arr_copy(A):
    return lambda A: init_map_arr(A._data.copy())  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def map_arr_setitem(arr, ind, val):
    """
    Support for setitem on MapArrays. MapArrays are currently
    an immutable type, so this should only be used when initializing
    a MapArray, for example when used creating a map array as the result
    of DataFrame.apply().
    """

    if not isinstance(arr, MapArrayType):
        return

    # NOTE: assuming that the array is being built and all previous elements are set
    # TODO: make sure array is being build

    typ_tuple = (arr.key_arr_type, arr.value_arr_type)

    if isinstance(ind, types.Integer):

        def map_arr_setitem_impl(arr, ind, val):  # pragma: no cover
            keys = val.keys()

            # Setitem requires resizing the underlying arrays which has a lot of complexity.
            # To simplify this limited use case, we copy the data twice.
            # TODO: Replace the struct array allocation with modifying the underlying array_item_array directly
            struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(
                len(val), (-1,), typ_tuple, ("key", "value")
            )
            for i, key in enumerate(keys):
                # Struct arrays are organized as a tuple of arrays, 1 per field.
                # The field names tell Bodo which array to insert into.
                struct_arr[i] = bodo.libs.struct_arr_ext.init_struct(
                    (key, val[key]), ("key", "value")
                )
            # The _data array is the underlying array_item_array, which is an array
            # of struct arrays.
            arr._data[ind] = struct_arr

        return map_arr_setitem_impl

    raise BodoError(
        "operator.setitem with MapArrays is only supported with an integer index."
    )


@overload(operator.getitem, no_unliteral=True)
def map_arr_getitem(arr, ind):
    if not isinstance(arr, MapArrayType):
        return

    if isinstance(ind, types.Integer):
        # TODO: warning if value is NA?
        def map_arr_getitem_impl(arr, ind):  # pragma: no cover
            if ind < 0:
                ind += len(arr)
            out = dict()
            offsets = bodo.libs.array_item_arr_ext.get_offsets(arr._data)
            struct_arr = bodo.libs.array_item_arr_ext.get_data(arr._data)
            key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)
            start_offset = offsets[ind]
            end_offset = offsets[ind + 1]
            for i in range(start_offset, end_offset):
                out[key_data[i]] = value_data[i]
            return out

        return map_arr_getitem_impl

    raise BodoError(
        "operator.getitem with MapArrays is only supported with an integer index."
    )
