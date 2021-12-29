# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Wrapper class for Tuples that supports tracking null entries.
This is primarily used for maintaining null information for
Series values used in df.apply
"""

import operator

from numba.core import cgutils, types
from numba.extending import (
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    register_model,
)


class NullableTupleType(types.Type):
    """
    Wrapper around various tuple classes that
    includes a null bitmap.

    Note this type is only intended for use with small
    number of values because it uses tuples internally.
    """

    def __init__(self, tuple_typ, null_typ):
        self._tuple_typ = tuple_typ
        # Null values is included to avoid requiring casting.
        self._null_typ = null_typ
        super(NullableTupleType, self).__init__(
            name=f"NullableTupleType({tuple_typ}, {null_typ})"
        )

    @property
    def tuple_typ(self):
        return self._tuple_typ

    @property
    def null_typ(self):
        return self._null_typ

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self._tuple_typ[i]

    @property
    def key(self):
        return self._tuple_typ

    @property
    def dtype(self):
        return self.tuple_typ.dtype

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(NullableTupleType)
class NullableTupleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", fe_type.tuple_typ), ("null_values", fe_type.null_typ)]
        super(NullableTupleModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(NullableTupleType, "data", "_data")
make_attribute_wrapper(NullableTupleType, "null_values", "_null_values")


@intrinsic
def build_nullable_tuple(typingctx, data_tuple, null_values):
    assert isinstance(
        data_tuple, types.BaseTuple
    ), "build_nullable_tuple 'data_tuple' argument must be a tuple"
    assert isinstance(
        null_values, types.BaseTuple
    ), "build_nullable_tuple 'null_values' argument must be a tuple"

    def codegen(context, builder, signature, args):
        data_tuple, null_values = args
        nullable_tuple = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )

        nullable_tuple.data = data_tuple
        nullable_tuple.null_values = null_values
        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_tuple)
        context.nrt.incref(builder, signature.args[1], null_values)
        return nullable_tuple._getvalue()

    sig = NullableTupleType(data_tuple, null_values)(data_tuple, null_values)
    return sig, codegen


@box(NullableTupleType)
def box_nullable_tuple(typ, val, c):
    """
    Boxes a nullable tuple as a regular Python tuple with the appropriate
    NA value based on the scalar types.
    """
    nullable_tuple_struct = cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, typ.tuple_typ, nullable_tuple_struct.data)
    c.context.nrt.incref(c.builder, typ.null_typ, nullable_tuple_struct.null_values)
    # box both the tuple and the null values.
    tuple_obj = c.pyapi.from_native_value(
        typ.tuple_typ, nullable_tuple_struct.data, c.env_manager
    )
    null_values_obj = c.pyapi.from_native_value(
        typ.null_typ, nullable_tuple_struct.null_values, c.env_manager
    )
    n_elems = c.context.get_constant(types.int64, len(typ.tuple_typ))
    # Create a list to convert to a tuple
    list_obj = c.pyapi.list_new(n_elems)
    with cgutils.for_range(c.builder, n_elems) as loop:
        i = loop.index
        py_index = c.pyapi.long_from_longlong(i)
        null_val = c.pyapi.object_getitem(null_values_obj, py_index)
        # TODO: Check True vs False with converting the value back
        null_bool_val = c.pyapi.to_native_value(types.bool_, null_val).value
        with c.builder.if_else(null_bool_val) as (then, orelse):
            with then:
                # TODO: Generate the correct null type for each type.
                # For example:
                # None for Strings
                # pd.NA for NullableIntegers
                # NaN for Float
                # NaT for Datetime64/Timedelta64
                c.pyapi.list_setitem(list_obj, i, c.pyapi.make_none())
            with orelse:
                tuple_val = c.pyapi.object_getitem(tuple_obj, py_index)
                c.pyapi.list_setitem(list_obj, i, tuple_val)
        # Decref py objects
        c.pyapi.decref(py_index)
        c.pyapi.decref(null_val)

    tuple_func = c.pyapi.unserialize(c.pyapi.serialize_object(tuple))
    final_tuple_obj = c.pyapi.call_function_objargs(tuple_func, (list_obj,))

    # Decref py objects
    c.pyapi.decref(tuple_obj)
    c.pyapi.decref(null_values_obj)
    c.pyapi.decref(tuple_func)
    c.pyapi.decref(list_obj)

    # Decref val
    c.context.nrt.decref(c.builder, typ, val)

    return final_tuple_obj


@overload(operator.getitem)
def overload_getitem(A, idx):
    if not isinstance(A, NullableTupleType):  # pragma: no cover
        return

    return lambda A, idx: A._data[idx]  # pragma: no cover
