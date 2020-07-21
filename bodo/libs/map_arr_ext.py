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
)
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext


ll.add_symbol("count_total_elems_list_array", array_ext.count_total_elems_list_array)
ll.add_symbol("map_array_from_sequence", array_ext.map_array_from_sequence)


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

    if handle_in_c:
        # allocate data array
        data_arr_type = _get_map_arr_data_type(typ)
        data_arr = gen_allocate_array(
            c.context, c.builder, data_arr_type, n_elems_all, c
        )
        payload = _get_array_item_arr_payload(
            c.context, c.builder, data_arr_type, data_arr
        )

        # get null and offset array pointers to pass to C
        null_bitmap_ptr = c.context.make_array(types.Array(types.uint8, 1, "C"))(
            c.context, c.builder, payload.null_bitmap
        ).data
        offsets_ptr = c.context.make_array(types.Array(offset_typ, 1, "C"))(
            c.context, c.builder, payload.offsets
        ).data

        # get key/value array pointers to pass to C
        struct_arr_payload = _get_struct_arr_payload(
            c.context, c.builder, data_arr_type.dtype, payload.data
        )
        key_arr = c.builder.extract_value(struct_arr_payload.data, 0)
        value_arr = c.builder.extract_value(struct_arr_payload.data, 1)
        key_arr_ptr = c.context.make_array(data_arr_type.dtype.data[0])(
            c.context, c.builder, key_arr
        ).data
        value_arr_ptr = c.context.make_array(data_arr_type.dtype.data[1])(
            c.context, c.builder, value_arr
        ).data

        # set all values to not NA in struct since NA not possible
        sig = types.none(types.Array(types.uint8, 1, "C"))
        _is_error, _res = c.pyapi.call_jit_code(
            lambda A: A.fill(255), sig, [struct_arr_payload.null_bitmap]
        )

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
        pass

    map_array = c.context.make_helper(c.builder, typ)
    map_array.data = data_arr

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(map_array._getvalue(), is_error=is_error)
