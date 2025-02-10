"""
Wrapper class for Tuples that supports tracking null entries.
This is primarily used for maintaining null information for
Series values used in df.apply
"""

import operator

import numba
from numba.core import cgutils, types
from numba.extending import (
    box,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_method,
    register_model,
)


class NullableTupleType(types.IterableType):
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
        super().__init__(name=f"NullableTupleType({tuple_typ}, {null_typ})")

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

    @property
    def iterator_type(self):
        # just provide the iterator over the data
        # TODO: Support nullable section (likely optional issues)
        return self.tuple_typ.iterator_type

    def __len__(self):
        # Determine len based on tuple.
        return len(self.tuple_typ)


@register_model(NullableTupleType)
class NullableTupleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", fe_type.tuple_typ), ("null_values", fe_type.null_typ)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(NullableTupleType, "data", "_data")
make_attribute_wrapper(NullableTupleType, "null_values", "_null_values")


@intrinsic
def build_nullable_tuple(typingctx, data_tuple, null_values):
    assert isinstance(data_tuple, types.BaseTuple), (
        "build_nullable_tuple 'data_tuple' argument must be a tuple"
    )
    assert isinstance(null_values, types.BaseTuple), (
        "build_nullable_tuple 'null_values' argument must be a tuple"
    )

    # Unliteral to prevent mismatch when Null value may or may not be
    # known at compile time. This occurs when typing with dummy values.
    data_tuple = types.unliteral(data_tuple)
    null_values = types.unliteral(null_values)

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


@overload(len)
def overload_len(A):
    if not isinstance(A, NullableTupleType):  # pragma: no cover
        return

    return lambda A: len(A._data)  # pragma: no cover


@lower_builtin("getiter", NullableTupleType)
def nullable_tuple_getiter(context, builder, sig, args):
    """support getting an iterator object for NullableTupleType by calling 'getiter'
    on the underlying data tuple.
    """
    # TODO: Support include the null values in iteration
    nullable_tuple = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    impl = context.get_function("getiter", sig.return_type(sig.args[0].tuple_typ))
    return impl(builder, (nullable_tuple.data,))


@overload(operator.eq)
def nullable_tuple_eq(val1, val2):
    """
    Implementation for equality between 2 nullable
    tuples. If two tuples aren't exactly the same type
    they are considered unequal.
    """
    if not isinstance(val1, NullableTupleType) or not isinstance(
        val2, NullableTupleType
    ):
        # Only support equality between two nullable tuples
        return
    # Only compare equality if the types are considered the same.
    # In build_nullable_tuple (the only valid nullable tuple
    # constructor), we unliteral all types in the signature. As
    # a result, we never have to worry about types not matching
    # because of literals.
    if val1 != val2:
        return lambda val1, val2: False  # pragma: no cover

    func_text = "def impl(val1, val2):\n"
    func_text += "    data_tup1 = val1._data\n"
    func_text += "    null_tup1 = val1._null_values\n"
    func_text += "    data_tup2 = val2._data\n"
    func_text += "    null_tup2 = val2._null_values\n"
    tup_typ = val1._tuple_typ
    for i in range(len(tup_typ)):
        func_text += f"    null1_{i} = null_tup1[{i}]\n"
        func_text += f"    null2_{i} = null_tup2[{i}]\n"
        func_text += f"    data1_{i} = data_tup1[{i}]\n"
        func_text += f"    data2_{i} = data_tup2[{i}]\n"
        func_text += f"    if null1_{i} != null2_{i}:\n"
        func_text += "        return False\n"
        func_text += f"    if null1_{i} and (data1_{i} != data2_{i}):\n"
        func_text += "        return False\n"
    func_text += "    return True\n"
    local_vars = {}
    exec(func_text, {}, local_vars)
    impl = local_vars["impl"]
    return impl


@overload_method(NullableTupleType, "__hash__")
def nullable_tuple_hash(val):
    """
    Implementation of hash for nullable tuples.
    This implementation chooses to hash only the entries
    that are valid (since NULL values have undefined
    values).

    Note this heavily reuses the existing numba implementation
    """

    def impl(val):  # pragma: no cover
        return _nullable_tuple_hash(val)

    return impl


_PyHASH_XXPRIME_1 = numba.cpython.hashing._PyHASH_XXPRIME_1
_PyHASH_XXPRIME_2 = numba.cpython.hashing._PyHASH_XXPRIME_1
_PyHASH_XXPRIME_5 = numba.cpython.hashing._PyHASH_XXPRIME_1


@numba.generated_jit(nopython=True)
def _nullable_tuple_hash(nullable_tup):  # pragma: no cover
    """
    Copies the Numba base tuple implementation
    but skips any the data for any values
    that are null.

    Since we use multiple tuples it is necessary to generate code
    that iterates over the tuple elements.
    """
    func_text = "def impl(nullable_tup):\n"
    func_text += "    data_tup = nullable_tup._data\n"
    func_text += "    null_tup = nullable_tup._null_values\n"
    func_text += "    tl = numba.cpython.hashing._Py_uhash_t(len(data_tup))\n"
    func_text += "    acc = _PyHASH_XXPRIME_5\n"
    tup_typ = nullable_tup._tuple_typ
    for i in range(len(tup_typ)):
        func_text += f"    null_val_{i} = null_tup[{i}]\n"
        func_text += f"    null_lane_{i} = hash(null_val_{i})\n"
        func_text += f"    if null_lane_{i} == numba.cpython.hashing._Py_uhash_t(-1):\n"
        func_text += "        return -1\n"
        func_text += f"    acc += null_lane_{i} * _PyHASH_XXPRIME_2\n"
        func_text += "    acc = numba.cpython.hashing._PyHASH_XXROTATE(acc)\n"
        func_text += "    acc *= _PyHASH_XXPRIME_1\n"
        func_text += f"    if not null_val_{i}:\n"
        func_text += f"        lane_{i} = hash(data_tup[{i}])\n"
        func_text += f"        if lane_{i} == numba.cpython.hashing._Py_uhash_t(-1):\n"
        func_text += "            return -1\n"
        func_text += f"        acc += lane_{i} * _PyHASH_XXPRIME_2\n"
        func_text += "        acc = numba.cpython.hashing._PyHASH_XXROTATE(acc)\n"
        func_text += "        acc *= _PyHASH_XXPRIME_1\n"
    func_text += "    acc += tl ^ (_PyHASH_XXPRIME_5 ^ numba.cpython.hashing._Py_uhash_t(3527539))\n"
    func_text += "    if acc == numba.cpython.hashing._Py_uhash_t(-1):\n"
    func_text += "        return numba.cpython.hashing.process_return(1546275796)\n"
    func_text += "    return numba.cpython.hashing.process_return(acc)\n"
    local_vars = {}
    exec(
        func_text,
        {
            "numba": numba,
            "_PyHASH_XXPRIME_1": _PyHASH_XXPRIME_1,
            "_PyHASH_XXPRIME_2": _PyHASH_XXPRIME_2,
            "_PyHASH_XXPRIME_5": _PyHASH_XXPRIME_5,
        },
        local_vars,
    )
    impl = local_vars["impl"]
    return impl
