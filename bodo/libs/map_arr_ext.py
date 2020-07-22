# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for map values.
Corresponds to Spark's MapType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Map arrays: https://github.com/apache/arrow/blob/master/format/Schema.fbs

The implementation uses an array(struct) array underneath similar to Spark and Arrow.
For example: [{1: 2.1, 3: 1.1}, {5: -1.0}]
[[{"key": 1, "value" 2.1}, {"key": 3, "value": 1.1}], [{"key": 5, "value": -1.0}]]
"""
import operator
import numpy as np
from collections import namedtuple
import numba
import bodo

from numba.core import types, cgutils
from numba.extending import (
    typeof_impl,
    type_callable,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    lower_builtin,
    box,
    unbox,
    lower_getattr,
    intrinsic,
    overload_method,
    overload,
    overload_attribute,
)
from numba.parfors.array_analysis import ArrayAnalysis
from numba.core.imputils import impl_ret_borrowed
from numba.typed.typedobjectutils import _cast

from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    offset_typ,
)
from bodo.libs.struct_arr_ext import StructArrayType, _get_struct_arr_payload
from bodo.utils.typing import (
    is_list_like_index_type,
    BodoError,
    get_overload_const_str,
    get_overload_const_int,
    is_overload_constant_str,
    is_overload_constant_int,
)
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.utils.cg_helpers import (
    set_bitmap_bit,
    pyarray_getitem,
    pyarray_setitem,
    list_check,
    is_na_value,
    get_bitmap_bit,
    get_array_elem_counts,
    seq_getitem,
    gen_allocate_array,
    to_arr_obj_if_list_obj,
    dict_keys,
    dict_values,
)
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext


ll.add_symbol("count_total_elems_list_array", array_ext.count_total_elems_list_array)
ll.add_symbol("map_array_from_sequence", array_ext.map_array_from_sequence)
ll.add_symbol("np_array_from_map_array", array_ext.np_array_from_map_array)


class MapArrayType(types.ArrayCompatible):
    """Data type for arrays of maps
    """

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
    """get array(struct) array data type for underlying data array of map type
    """
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
        and t.dtype in (types.int64, types.float64, types.bool_, datetime_date_type,)
        for t in (typ.key_arr_type, typ.value_arr_type)
    )

    if handle_in_c:
        fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_tp = c.builder.module.get_or_insert_function(
            fnty, name="count_total_elems_list_array"
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
    offsets_ptr = c.context.make_array(types.Array(offset_typ, 1, "C"))(
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
                lir.IntType(32).as_pointer(),  # offsets
                lir.IntType(8).as_pointer(),  # null_bitmap
                lir.IntType(32),  # key ctype
                lir.IntType(32),  # value ctype
            ],
        )
        fn = c.builder.module.get_or_insert_function(
            fnty, name="map_array_from_sequence"
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

    zero32 = c.context.get_constant(types.int32, 0)
    builder.store(zero32, offsets_ptr)

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
            builder.trunc(item_ind, lir.IntType(32)),
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
        builder.trunc(builder.load(curr_item_ind), lir.IntType(32)),
        builder.gep(offsets_ptr, [n_maps]),
    )

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)


@box(MapArrayType)
def box_map_arr(typ, val, c):
    """box packed native representation of map array into python objects
    """
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
    offsets_ptr = c.context.make_array(types.Array(offset_typ, 1, "C"))(
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
                lir.IntType(32).as_pointer(),  # offsets
                lir.IntType(8).as_pointer(),  # null_bitmap
                lir.IntType(32),  # key ctype
                lir.IntType(32),  # value ctype
            ],
        )
        fn_get = c.builder.module.get_or_insert_function(
            fnty, name="np_array_from_map_array"
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
        pass

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def init_map_arr(typingctx, data_typ=None):
    """create a new map array from input data list(struct) array data
    """
    assert isinstance(data_typ, ArrayItemArrayType) and isinstance(
        data_typ.dtype, StructArrayType
    )
    map_arr_type = MapArrayType(data_typ.dtype.data[0], data_typ.dtype.data[1])

    def codegen(context, builder, sig, args):
        (data_arr,) = args
        map_array = context.make_helper(builder, map_arr_type)
        map_array.data = data_arr
        context.nrt.incref(builder, data_typ, data_arr)
        return map_array._getvalue()

    return map_arr_type(data_typ), codegen


@overload(len, no_unliteral=True)
def overload_map_arr_len(A):
    if isinstance(A, MapArrayType):
        return lambda A: len(A._data)  # pragma: no cover
