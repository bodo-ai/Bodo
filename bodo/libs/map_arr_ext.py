# Copyright (C) 2022 Bodo Inc. All rights reserved.
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
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils.typing import BodoError, is_list_like_index_type

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import array_ext, hdist  # isort:skip


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

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


def _get_map_arr_data_type(map_type: MapArrayType) -> ArrayItemArrayType:
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
    Unbox an array with dictionary values.
    """
    return bodo.libs.array.unbox_nested_array(typ, val, c)


@box(MapArrayType)
def box_map_arr(typ, val, c):
    """box packed native representation of map array into python objects"""
    return bodo.libs.array.box_nested_array(typ, val, c)


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
        if isinstance(val, bodo.StructArrayType):
            if val.data != typ_tuple or val.names != (
                "key",
                "value",
            ):  # pragma: no cover
                return None

            def map_arr_setitem_impl(arr, ind, val):  # pragma: no cover
                arr._data[ind] = val

            return map_arr_setitem_impl

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
            start_offset = offsets[np.int64(ind)]
            end_offset = offsets[np.int64(ind) + 1]
            for i in range(start_offset, end_offset):
                out[key_data[i]] = value_data[i]
            return out

        return map_arr_getitem_impl

    if (is_list_like_index_type(ind) and ind.dtype == types.bool_) or isinstance(
        ind, types.SliceType
    ):

        def map_arr_getitem_impl(arr, ind):  # pragma: no cover
            # Reuse the array item array implementation
            return init_map_arr(arr._data[ind])

        return map_arr_getitem_impl

    raise BodoError(
        f"getitem for MapArray with indexing type {ind} not supported."
    )  # pragma: no cover
