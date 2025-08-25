"""Array implementation for map values.
Corresponds to Spark's MapType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Map arrays: https://github.com/apache/arrow/blob/master/format/Schema.fbs

The implementation uses an array(struct) array underneath similar to Spark and Arrow.
For example: [{1: 2.1, 3: 1.1}, {5: -1.0}]
[[{"key": 1, "value" 2.1}, {"key": 3, "value": 1.1}], [{"key": 5, "value": -1.0}]]
"""

import operator

import numba
import numpy as np
from numba.core import cgutils, types
from numba.extending import (
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
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils.typing import (
    BodoError,
    is_list_like_index_type,
    is_overload_none,
    unwrap_typeref,
)

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import array_ext, hdist  # noqa: F401  # isort:skip


class MapScalarType(types.Type):
    """Data type for map elements taken as scalars from map arrays. A regular
    dictionary doesn't work in the general case since values can have nulls and a
    key/value pair could be null, which is not supported by Numba's dictionaries.
    """

    def __init__(self, key_arr_type, value_arr_type):
        self.key_arr_type = key_arr_type
        self.value_arr_type = value_arr_type
        super().__init__(name=f"MapScalarType({key_arr_type}, {value_arr_type})")

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


class MapArrayType(types.ArrayCompatible):
    """Data type for arrays of maps"""

    def __init__(self, key_arr_type, value_arr_type):
        self.key_arr_type = key_arr_type
        self.value_arr_type = value_arr_type
        super().__init__(name=f"MapArrayType({key_arr_type}, {value_arr_type})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return MapScalarType(self.key_arr_type, self.value_arr_type)

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
    return bodo.libs.array.unbox_array_using_arrow(typ, val, c)


@box(MapArrayType)
def box_map_arr(typ, val, c):
    """box packed native representation of map array into python objects"""
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


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
def init_map_arr(typingctx, data_typ):
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


numba.core.ir_utils.alias_func_extensions[("init_map_arr", "bodo.libs.map_arr_ext")] = (
    alias_ext_init_map_arr
)


def pre_alloc_map_array(
    num_maps, nested_counts, struct_typ, dict_ref_arr=None
):  # pragma: no cover
    pass


@overload(pre_alloc_map_array)
def overload_pre_alloc_map_array(
    num_maps, nested_counts, struct_typ, dict_ref_arr=None
):
    if not is_overload_none(dict_ref_arr):

        def impl_pre_alloc_map_array(
            num_maps, nested_counts, struct_typ, dict_ref_arr=None
        ):  # pragma: no cover
            data = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
                num_maps,
                nested_counts,
                bodo.libs.array_item_arr_ext.get_data(dict_ref_arr._data),
            )
            return init_map_arr(data)

        return impl_pre_alloc_map_array

    def impl_pre_alloc_map_array(
        num_maps, nested_counts, struct_typ, dict_ref_arr=None
    ):  # pragma: no cover
        data = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
            num_maps, nested_counts, struct_typ
        )
        return init_map_arr(data)

    return impl_pre_alloc_map_array


def pre_alloc_map_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_map_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) > 0
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

        if isinstance(val, MapScalarType):

            def map_arr_setitem_impl(arr, ind, val):  # pragma: no cover
                struct_arr = bodo.libs.struct_arr_ext.init_struct_arr(
                    len(val._keys),
                    (val._keys, val._values),
                    val._null_bitmask,
                    ("key", "value"),
                )
                arr._data[ind] = struct_arr

            return map_arr_setitem_impl

        if not isinstance(val, types.DictType):  # pragma: no cover
            raise BodoError(
                f"Unsupported operator.setitem with MapArrays with index '{ind}' and value '{val}'."
            )

        def map_arr_setitem_impl(arr, ind, val):  # pragma: no cover
            keys = val.keys()

            # Setitem requires resizing the underlying arrays which has a lot of complexity.
            # To simplify this limited use case, we copy the data twice.
            # TODO: Replace the struct array allocation with modifying the underlying array_item_array directly
            struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(
                len(val), (-1,), typ_tuple, ("key", "value"), None
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

    # Handle setting a slice of the array during construction
    if isinstance(ind, types.SliceType) and val == arr:

        def map_arr_setitem_impl(arr, ind, val):  # pragma: no cover
            # NOTE: [:len(val)] is necessary since val could be over-allocated
            arr._data[ind] = val._data[: len(val)]

        return map_arr_setitem_impl

    raise BodoError(
        f"Unsupported operator.setitem with MapArrays with index '{ind}' and value '{val}'."
    )


@overload(operator.getitem, no_unliteral=True)
def map_arr_getitem(arr, ind):
    if not isinstance(arr, MapArrayType):
        return

    if isinstance(ind, types.Integer):

        def map_arr_getitem_impl(arr, ind):  # pragma: no cover
            if ind < 0:
                ind += len(arr)
            offsets = bodo.libs.array_item_arr_ext.get_offsets(arr._data)
            struct_arr = bodo.libs.array_item_arr_ext.get_data(arr._data)
            key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)
            nulls = bodo.libs.struct_arr_ext.get_null_bitmap(struct_arr)
            start_offset = offsets[np.int64(ind)]
            end_offset = offsets[np.int64(ind) + 1]
            new_nulls = np.empty((end_offset - start_offset + 7) >> 3, np.uint8)
            curr_bit = 0
            for i in range(start_offset, end_offset):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(nulls, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(new_nulls, curr_bit, bit)
                curr_bit += 1
            return init_map_value(
                key_data[start_offset:end_offset],
                value_data[start_offset:end_offset],
                new_nulls,
            )

        return map_arr_getitem_impl

    if (
        is_list_like_index_type(ind)
        and (ind.dtype == types.bool_ or isinstance(ind.dtype, types.Integer))
    ) or isinstance(ind, types.SliceType):

        def map_arr_getitem_impl(arr, ind):  # pragma: no cover
            # Reuse the array item array implementation
            return init_map_arr(arr._data[ind])

        return map_arr_getitem_impl

    raise BodoError(
        f"getitem for MapArray with indexing type {ind} not supported."
    )  # pragma: no cover


def contains_map_array(arr):
    """Returns True if the array contains any maps or is a map"""
    if isinstance(arr, bodo.MapArrayType):
        return True
    elif isinstance(arr, bodo.ArrayItemArrayType):
        return contains_map_array(arr.dtype)
    elif isinstance(arr, bodo.StructArrayType):
        return any(contains_map_array(t) for t in arr.data)
    else:
        return False


@register_model(MapScalarType)
class MapValueTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # Stores key and value arrays
        members = [
            ("keys", fe_type.key_arr_type),
            ("values", fe_type.value_arr_type),
            ("null_bitmask", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(MapScalarType, "keys", "_keys")
make_attribute_wrapper(MapScalarType, "values", "_values")
make_attribute_wrapper(MapScalarType, "null_bitmask", "_null_bitmask")


@intrinsic
def init_map_value(typingctx, key_arr_type, value_arr_type, null_bitmask_type):
    """Create a MapValue from key and value arrays"""
    assert null_bitmask_type == types.Array(types.uint8, 1, "C"), (
        "init_map_value invalid null_bitmask_type"
    )

    def codegen(context, builder, signature, args):
        map_val = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        map_val.keys = args[0]
        map_val.values = args[1]
        map_val.null_bitmask = args[2]

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], args[0])
        context.nrt.incref(builder, signature.args[1], args[1])
        context.nrt.incref(builder, signature.args[2], args[2])

        return map_val._getvalue()

    out_type = MapScalarType(key_arr_type, value_arr_type)
    return out_type(key_arr_type, value_arr_type, null_bitmask_type), codegen


def key_values_to_dict(key_arr, value_arr, null_bitmask):
    """Convert key and value arrays into dictionary"""
    return {
        k: v
        for i, (k, v) in enumerate(zip(key_arr, value_arr))
        if bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmask, i)
    }


@overload(len)
def overload_len(A):
    if isinstance(A, MapScalarType):

        def impl_len(A):
            return len(A._keys)

        return impl_len


@box(MapScalarType)
def box_map_value(typ, val, c):
    """box map value into python dictionary objects"""

    map_value = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(c.builder, typ.key_arr_type, map_value.keys)
    key_arr_obj = c.pyapi.from_native_value(
        typ.key_arr_type, map_value.keys, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.value_arr_type, map_value.values)
    value_arr_obj = c.pyapi.from_native_value(
        typ.value_arr_type, map_value.values, c.env_manager
    )
    c.context.nrt.incref(
        c.builder, types.Array(types.uint8, 1, "C"), map_value.null_bitmask
    )
    null_bitmask_obj = c.pyapi.from_native_value(
        types.Array(types.uint8, 1, "C"), map_value.null_bitmask, c.env_manager
    )

    key_values_to_dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(key_values_to_dict)
    )
    dict_obj = c.pyapi.call_function_objargs(
        key_values_to_dict_obj, [key_arr_obj, value_arr_obj, null_bitmask_obj]
    )
    c.pyapi.decref(key_arr_obj)
    c.pyapi.decref(value_arr_obj)
    c.pyapi.decref(null_bitmask_obj)
    c.pyapi.decref(key_values_to_dict_obj)

    c.context.nrt.decref(c.builder, typ, val)
    return dict_obj


@overload(operator.getitem, no_unliteral=True)
def map_val_getitem(val, ind):
    if not isinstance(val, MapScalarType):
        return

    # Convert to dict for proper getitem
    # NOTE: values can be NA but we assume NA checks are done before accessing the
    # values to avoid adding complexity of Optional output here
    return lambda val, ind: dict(val)[ind]  # pragma: no cover


@overload(dict)
def dict_dict_overload(val):
    """Calling dict() constructor on dict value should be a copy"""
    if isinstance(val, types.DictType):
        return lambda val: val.copy()  # pragma: no cover


@overload(dict)
def dict_map_val_overload(val):
    """Support dict constructor for MapScalarType input"""
    if not isinstance(val, MapScalarType):
        return

    key_type = val.key_arr_type.dtype
    value_type = val.value_arr_type.dtype

    def dict_map_val_impl(val):  # pragma: no cover
        keys = val._keys
        values = val._values
        null_bitmask = val._null_bitmask
        n = len(val._keys)
        out = numba.typed.typeddict.Dict.empty(key_type, value_type)
        for i in range(n):
            if bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmask, i):
                k = keys[i]
                v = values[i]
                out[k] = v
        return out

    return dict_map_val_impl


@overload(list)
def list_map_val_overload(val):
    """Support list constructor for MapScalarType input"""
    if not isinstance(val, MapScalarType):
        return

    def list_map_val_impl(val):  # pragma: no cover
        keys = val._keys
        null_bitmask = val._null_bitmask
        n = len(keys)
        out = []
        for i in range(n):
            if bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmask, i):
                out.append(keys[i])
        return out

    return list_map_val_impl


def scalar_to_map_array(scalar_val, length, _arr_typ):
    pass


@overload(scalar_to_map_array)
def overload_array_to_repeated_map_array(scalar_val, length, _arr_typ):
    """
    Create an MapArray of length `length` by repeating scalar_val `length` times

    Args:
        scalar_val (MapScalarType): The map value to be repeated
        length (int): Length of the output MapArray
        _arr_typ (types.Type): MapArrayType for output
    Returns:
        An MapArray of length `length`
    """

    arr_type = unwrap_typeref(_arr_typ)
    struct_arr_type = bodo.libs.struct_arr_ext.StructArrayType(
        (arr_type.key_arr_type, arr_type.value_arr_type), ("key", "value")
    )

    def impl(scalar_val, length, _arr_typ):  # pragma: no cover
        out_arr = pre_alloc_map_array(length, (-1,), struct_arr_type)
        for i in range(length):
            out_arr[i] = scalar_val
        return out_arr

    return impl
