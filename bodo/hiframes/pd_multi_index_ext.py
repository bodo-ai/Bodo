# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Support for MultiIndex type of Pandas
"""
import numba
import pandas as pd
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.utils.typing import get_val_type_maybe_str_literal


# NOTE: minimal MultiIndex support that just stores the index arrays without factorizing
# the data into `levels` and `codes`
# TODO: support factorizing similar to pd.core.algorithms._factorize_array
class MultiIndexType(types.Type):
    """type class for pd.MultiIndex object"""

    def __init__(self, array_types, names_typ=None, name_typ=None):
        # NOTE: store array types instead of just dtypes since we currently store whole
        # arrays
        # NOTE: array_types and names_typ should be tuples of types
        names_typ = (types.none,) * len(array_types) if names_typ is None else names_typ
        # name is stored in MultiIndex for compatibility witn Index (not always used)
        name_typ = types.none if name_typ is None else name_typ
        self.array_types = array_types
        self.names_typ = names_typ
        self.name_typ = name_typ
        super(MultiIndexType, self).__init__(
            name="MultiIndexType({}, {}, {})".format(array_types, names_typ, name_typ)
        )

    ndim = 1

    def copy(self):
        return MultiIndexType(self.array_types, self.names_typ, self.name_typ)

    @property
    def nlevels(self):
        return len(self.array_types)


# NOTE: just storing the arrays. TODO: store `levels` and `codes`
# even though `name` attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(MultiIndexType)
class MultiIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Tuple(fe_type.array_types)),
            ("names", types.Tuple(fe_type.names_typ)),
            ("name", fe_type.name_typ),
        ]
        super(MultiIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(MultiIndexType, "data", "_data")
make_attribute_wrapper(MultiIndexType, "names", "_names")
make_attribute_wrapper(MultiIndexType, "name", "_name")


@typeof_impl.register(pd.MultiIndex)
def typeof_multi_index(val, c):
    # using array type inference
    # TODO: avoid using .values if possible, since behavior of .values may change
    array_types = tuple(numba.typeof(val.levels[i].values) for i in range(val.nlevels))
    return MultiIndexType(
        array_types,
        tuple(get_val_type_maybe_str_literal(v) for v in val.names),
        numba.typeof(val.name),
    )


@box(MultiIndexType)
def box_multi_index(typ, val, c):
    """box into pd.MultiIndex object. using `from_arrays` since we are just storing
    arrays currently. TODO: support `levels` and `codes`
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    multi_index_class_obj = c.pyapi.object_getattr_string(class_obj, "MultiIndex")

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, types.Tuple(typ.array_types), index_val.data)
    data = c.pyapi.from_native_value(
        types.Tuple(typ.array_types), index_val.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, types.Tuple(typ.names_typ), index_val.names)
    names = c.pyapi.from_native_value(
        types.Tuple(typ.names_typ), index_val.names, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    sortorder = c.pyapi.make_none()
    index_obj = c.pyapi.call_method(
        multi_index_class_obj, "from_arrays", (data, sortorder, names)
    )
    c.pyapi.object_setattr_string(index_obj, "name", name)

    c.pyapi.decref(class_obj)
    c.pyapi.decref(multi_index_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@unbox(MultiIndexType)
def unbox_multi_index(typ, val, c):
    """ubox pd.MultiIndex object into native representation.
    using `get_level_values()` to get arrays out since we are just storing
    arrays currently. TODO: support `levels` and `codes`
    """
    data_arrs = []
    # save array objects to decref later since array may be created on demand and
    # cleaned up in Pandas
    arr_objs = []

    for i in range(typ.nlevels):
        # generate val.get_level_values(i)
        i_obj = c.pyapi.unserialize(c.pyapi.serialize_object(i))
        level_obj = c.pyapi.call_method(val, "get_level_values", (i_obj,))
        array_obj = c.pyapi.object_getattr_string(level_obj, "values")
        c.pyapi.decref(level_obj)
        c.pyapi.decref(i_obj)
        data_arr = c.pyapi.to_native_value(typ.array_types[i], array_obj).value
        data_arrs.append(data_arr)
        arr_objs.append(array_obj)

    # set data, names and name attributes
    # if array types are uniform, LLVM ArrayType should be used,
    # otherwise, LiteralStructType is needed
    if isinstance(types.Tuple(typ.array_types), types.UniTuple):
        data = cgutils.pack_array(c.builder, data_arrs)
    else:
        data = cgutils.pack_struct(c.builder, data_arrs)
    # names = tuple(val.names)
    names_obj = c.pyapi.object_getattr_string(val, "names")
    tuple_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tuple))
    names_tup_obj = c.pyapi.call_function_objargs(tuple_class_obj, (names_obj,))
    names = c.pyapi.to_native_value(types.Tuple(typ.names_typ), names_tup_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.names = names
    index_val.name = name

    for array_obj in arr_objs:
        c.pyapi.decref(array_obj)

    c.pyapi.decref(names_obj)
    c.pyapi.decref(tuple_class_obj)
    c.pyapi.decref(names_tup_obj)
    c.pyapi.decref(name_obj)
    return NativeValue(index_val._getvalue())


@intrinsic
def init_multi_index(typingctx, data, names, name=None):
    """Create a MultiIndex with provided data, names and name values."""
    name = types.none if name is None else name
    # recreate Tuple type to make sure UniTuple is created if types are homogeneous
    # instead of regular Tuple
    # happens in gatherv() implementation of MultiIndex for some reason
    names = types.Tuple(names.types)

    def codegen(context, builder, signature, args):
        data_val, names_val, name_val = args
        # create multi_index struct and store values
        multi_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        multi_index.data = data_val
        multi_index.names = names_val
        multi_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], names_val)
        context.nrt.incref(builder, signature.args[2], name_val)

        return multi_index._getvalue()

    ret_typ = MultiIndexType(data.types, names.types, name)
    return ret_typ(data, names, name), codegen
