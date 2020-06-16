# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for structs of values.
Corresponds to Spark's StructType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Struct arrays: https://arrow.apache.org/docs/format/Columnar.html

The values are stored in contingous data arrays; one array per field. For example:
A:             ["AA", "B", "C"]
B:             [1, 2, 4]
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

from bodo.utils.typing import is_list_like_index_type, BodoError
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext

ll.add_symbol("struct_array_from_sequence", array_ext.struct_array_from_sequence)
ll.add_symbol("np_array_from_struct_array", array_ext.np_array_from_struct_array)


class StructArrayType(types.ArrayCompatible):
    """Data type for arrays of structs
    """

    def __init__(self, data, names=None):
        # data is tuple of Array types
        # names is a tuple of field names
        assert (
            isinstance(data, tuple)
            and len(data) > 0
            and all(bodo.utils.utils.is_array_typ(a, False) for a in data)
        )
        if names is not None:
            assert (
                isinstance(names, tuple)
                and all(isinstance(a, str) for a in names)
                and len(names) == len(data)
            )
        else:
            names = tuple("f{}".format(i) for i in range(len(data)))

        self.data = data
        self.names = names
        super(StructArrayType, self).__init__(
            name="StructArrayType({}, {})".format(data, names)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    # TODO: we need a dict-like type that allows heterogenous values
    # @property
    # def dtype(self):
    #     return types.DictType(self.data)

    @classmethod
    def from_dict(cls, d):
        """create a StructArrayType from dict where keys are names and values are dtypes
        """
        assert isinstance(d, dict)
        names = tuple(str(a) for a in d.keys())
        data = tuple(
            bodo.hiframes.pd_series_ext._get_series_array_type(t) for t in d.values()
        )
        return StructArrayType(data, names)


class StructArrayPayloadType(types.Type):
    def __init__(self, data):
        assert isinstance(data, tuple) and all(
            bodo.utils.utils.is_array_typ(a, False) for a in data
        )
        self.data = data
        super(StructArrayPayloadType, self).__init__(
            name="StructArrayPayloadType({})".format(data)
        )


@register_model(StructArrayPayloadType)
class StructArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.BaseTuple.from_types(fe_type.data)),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(StructArrayType)
class StructArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = StructArrayPayloadType(fe_type.data)
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


def define_struct_arr_dtor(context, builder, struct_arr_type, payload_type):
    """
    Define destructor for struct array type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = mod.get_or_insert_function(
        fnty,
        name=".dtor.struct_arr.{}.{}.".format(
            struct_arr_type.data, struct_arr_type.names
        ),
    )

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(
        builder, types.BaseTuple.from_types(struct_arr_type.data), payload.data
    )
    context.nrt.decref(builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap)

    builder.ret_void()
    return fn


def construct_struct_array(context, builder, struct_arr_type, n_structs):
    """Creates meminfo and sets dtor, and allocates buffers for struct array
    """
    # create payload type
    payload_type = StructArrayPayloadType(struct_arr_type.data)
    alloc_type = context.get_data_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_struct_arr_dtor(context, builder, struct_arr_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)

    # alloc data
    # TODO: general alloc, not just Numpy
    arrs = []
    arr_ptrs = []
    for arr_typ in struct_arr_type.data:
        arr = bodo.utils.utils._empty_nd_impl(context, builder, arr_typ, [n_structs])
        arr_ptrs.append(arr.data)
        arrs.append(arr._getvalue())

    payload.data = (
        cgutils.pack_array(builder, arrs)
        if types.is_homogeneous(*struct_arr_type.data)
        else cgutils.pack_struct(builder, arrs)
    )

    # alloc null bitmap
    n_bitmask_bytes = builder.udiv(
        builder.add(n_structs, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    null_bitmap = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
    )
    null_bitmap_ptr = null_bitmap.data
    payload.null_bitmap = null_bitmap._getvalue()

    builder.store(payload._getvalue(), meminfo_data_ptr)

    return meminfo, arr_ptrs, null_bitmap_ptr


def _get_C_API_ptrs(c, arr_ptrs, data, names):
    """convert struct array info into pointers to pass to C API
    """

    # get pointer to a tuple of data pointers to pass to C
    data_ptr_tup = (
        cgutils.pack_array(c.builder, arr_ptrs)
        if types.is_homogeneous(*data)
        else cgutils.pack_struct(c.builder, arr_ptrs)
    )
    data_ptr_tup_ptr = cgutils.alloca_once_value(c.builder, data_ptr_tup)
    # get pointer to a tuple of c type enums to pass to C
    c_types = [
        c.context.get_constant(types.int32, bodo.utils.utils.numba_to_c_type(a.dtype))
        for a in data
    ]
    c_types_ptr = cgutils.alloca_once_value(
        c.builder, cgutils.pack_array(c.builder, c_types)
    )
    # get pointer to a tuple of field names to pass to C
    field_names = cgutils.pack_array(
        c.builder, [c.context.insert_const_string(c.builder.module, a) for a in names]
    )
    field_names_ptr = cgutils.alloca_once_value(c.builder, field_names)
    return data_ptr_tup_ptr, c_types_ptr, field_names_ptr


@unbox(StructArrayType)
def unbox_struct_array(typ, val, c):
    """
    Unbox a numpy array with list of data values.
    """
    # get length
    n_structs = bodo.utils.utils.object_length(c, val)
    # create struct array
    meminfo, arr_ptrs, null_bitmap_ptr = construct_struct_array(
        c.context, c.builder, typ, n_structs
    )

    data_ptr_tup_ptr, c_types_ptr, field_names_ptr = _get_C_API_ptrs(
        c, arr_ptrs, typ.data, typ.names
    )

    # function signature of struct_array_from_sequence
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # obj
            lir.IntType(32),  # number of arrays
            lir.IntType(8).as_pointer(),  # data
            lir.IntType(8).as_pointer(),  # null_bitmap
            lir.IntType(8).as_pointer(),  # c types
            lir.IntType(8).as_pointer(),  # field names
        ],
    )
    fn = c.builder.module.get_or_insert_function(
        fnty, name="struct_array_from_sequence"
    )

    c.builder.call(
        fn,
        [
            val,
            c.context.get_constant(types.int32, len(typ.data)),
            c.builder.bitcast(data_ptr_tup_ptr, lir.IntType(8).as_pointer()),
            null_bitmap_ptr,
            c.builder.bitcast(c_types_ptr, lir.IntType(8).as_pointer()),
            c.builder.bitcast(field_names_ptr, lir.IntType(8).as_pointer()),
        ],
    )

    struct_array = c.context.make_helper(c.builder, typ)
    struct_array.meminfo = meminfo

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(struct_array._getvalue(), is_error=is_error)


def _get_struct_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a struct array value
    """
    struct_array = context.make_helper(builder, arr_typ, arr)
    payload_type = StructArrayPayloadType(arr_typ.data)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, struct_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_data_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


@box(StructArrayType)
def box_struct_arr(typ, val, c):
    """box packed native representation of list of item array into python objects
    """

    payload = _get_struct_arr_payload(c.context, c.builder, typ, val)
    data_ptrs = []
    assert len(typ.data) > 0
    # TODO: support non-Numpy arrays
    for i, arr_typ in enumerate(typ.data):
        arr_ptr = c.builder.extract_value(payload.data, i)
        arr = c.context.make_array(arr_typ)(c.context, c.builder, value=arr_ptr)
        data_ptrs.append(arr.data)
        # get length from first array
        if i == 0:
            length = c.builder.extract_value(arr.shape, 0)
    data_ptr_tup_ptr, c_types_ptr, field_names_ptr = _get_C_API_ptrs(
        c, data_ptrs, typ.data, typ.names
    )

    fnty = lir.FunctionType(
        c.context.get_argument_type(types.pyobject),
        [
            lir.IntType(64),  # length
            lir.IntType(32),  # number of arrays
            lir.IntType(8).as_pointer(),  # data
            lir.IntType(8).as_pointer(),  # null_bitmap
            lir.IntType(8).as_pointer(),  # c types
            lir.IntType(8).as_pointer(),  # field names
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(
        fnty, name="np_array_from_struct_array"
    )

    null_bitmap_ptr = c.context.make_helper(
        c.builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap
    ).data

    arr = c.builder.call(
        fn_get,
        [
            length,
            c.context.get_constant(types.int32, len(typ.data)),
            c.builder.bitcast(data_ptr_tup_ptr, lir.IntType(8).as_pointer()),
            null_bitmap_ptr,
            c.builder.bitcast(c_types_ptr, lir.IntType(8).as_pointer()),
            c.builder.bitcast(field_names_ptr, lir.IntType(8).as_pointer()),
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


class StructRecordType(types.Type):
    """Data type for struct records taken as scalars from struct arrays. A regular
    dictionary doesn't work in the general case since values can have different types.
    Very similar structure to StructArrayType, except that it holds scalar values and
    supports getitem/setitem of fields.
    """

    def __init__(self, data, names):
        # data is tuple of scalar types
        # names is a tuple of field names
        assert (
            isinstance(data, tuple)
            and len(data) > 0
            and all(not bodo.utils.utils.is_array_typ(a, False) for a in data)
        )
        assert (
            isinstance(names, tuple)
            and all(isinstance(a, str) for a in names)
            and len(names) == len(data)
        )

        self.data = data
        self.names = names
        super(StructRecordType, self).__init__(
            name="StructRecordType({}, {})".format(data, names)
        )


class StructRecordPayloadType(types.Type):
    def __init__(self, data):
        assert isinstance(data, tuple) and all(
            not bodo.utils.utils.is_array_typ(a, False) for a in data
        )
        self.data = data
        super(StructRecordPayloadType, self).__init__(
            name="StructRecordPayloadType({})".format(data)
        )


@register_model(StructRecordPayloadType)
class StructRecordPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.BaseTuple.from_types(fe_type.data)),
            ("null_bitmap", types.UniTuple(types.int8, len(fe_type.data))),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(StructRecordType)
class StructRecordModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = StructRecordPayloadType(fe_type.data)
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


def define_struct_rec_dtor(context, builder, struct_rec_type, payload_type):
    """
    Define destructor for struct record type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = mod.get_or_insert_function(
        fnty,
        name=".dtor.struct_rec.{}.{}.".format(
            struct_rec_type.data, struct_rec_type.names
        ),
    )

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(
        builder, types.BaseTuple.from_types(struct_rec_type.data), payload.data
    )
    # no need for null_bitmap since it is using primitive types

    builder.ret_void()
    return fn


def _get_struct_rec_payload(context, builder, typ, rec):
    """get payload struct proxy for a struct record value
    """
    struct_rec = context.make_helper(builder, typ, rec)
    payload_type = StructRecordPayloadType(typ.data)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, struct_rec.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_data_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


@box(StructRecordType)
def box_struct_rec(typ, val, c):
    """box struct records into python dictionary objects
    """
    out_dict = c.pyapi.dict_new(len(typ.data))
    payload = _get_struct_rec_payload(c.context, c.builder, typ, val)

    assert len(typ.data) > 0
    for i, val_typ in enumerate(typ.data):
        value = c.builder.extract_value(payload.data, i)
        val_obj = c.pyapi.from_native_value(val_typ, value, c.env_manager)
        c.pyapi.dict_setitem_string(out_dict, typ.names[i], val_obj)
        c.pyapi.decref(val_obj)

    c.context.nrt.decref(c.builder, typ, val)
    return out_dict


def construct_struct_record(context, builder, struct_rec_type, values, nulls):
    """Creates meminfo and sets dtor and data for struct record
    """
    # create payload type
    payload_type = StructRecordPayloadType(struct_rec_type.data)
    alloc_type = context.get_data_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_struct_rec_dtor(context, builder, struct_rec_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)

    payload.data = (
        cgutils.pack_array(builder, values)
        if types.is_homogeneous(*struct_rec_type.data)
        else cgutils.pack_struct(builder, values)
    )

    payload.null_bitmap = cgutils.pack_array(builder, nulls)

    builder.store(payload._getvalue(), meminfo_data_ptr)
    return meminfo


@intrinsic
def struct_array_get_record(typingctx, struct_arr_typ, ind_typ=None):
    """get struct record from struct array, e.g. A[i]
    Returns a dictionary of value types are the same, otherwise a StructRecordType
    """
    assert isinstance(struct_arr_typ, StructArrayType) and isinstance(
        ind_typ, types.Integer
    )
    n_fields = len(struct_arr_typ.data)
    data_types = tuple(d.dtype for d in struct_arr_typ.data)
    # return a regular dictionary if values have the same type, otherwise record
    if types.is_homogeneous(*struct_arr_typ.data):
        out_typ = types.DictType(bodo.string_type, data_types[0])
    else:
        out_typ = StructRecordType(data_types, struct_arr_typ.names)

    def codegen(context, builder, sig, args):
        struct_arr, ind = args

        payload = _get_struct_arr_payload(context, builder, struct_arr_typ, struct_arr)
        data_vals = []
        # TODO: set nulls from data arrays
        nulls = [context.get_constant(types.uint8, 1) for _ in range(n_fields)]
        # TODO: support non-Numpy arrays
        for i, arr_typ in enumerate(struct_arr_typ.data):
            arr_ptr = builder.extract_value(payload.data, i)
            arr = context.make_array(arr_typ)(context, builder, value=arr_ptr)
            data_vals.append(
                numba.np.arrayobj._getitem_array_single_int(
                    context, builder, arr_typ.dtype, arr_typ, arr, ind
                )
            )

        if isinstance(out_typ, types.DictType):
            names_consts = [
                context.insert_const_string(builder.module, name)
                for name in struct_arr_typ.names
            ]
            val_tup = cgutils.pack_array(builder, data_vals)
            names_tup = cgutils.pack_array(builder, names_consts)
            # TODO: support NA values as optional type?
            def impl(names, vals):
                d = {}
                for i, name in enumerate(names):
                    d[name] = vals[i]
                return d

            return context.compile_internal(
                builder,
                impl,
                out_typ(
                    types.Tuple(
                        tuple(
                            types.StringLiteral(name) for name in struct_arr_typ.names
                        )
                    ),
                    types.Tuple(data_types),
                ),
                [names_tup, val_tup],
            )

        meminfo = construct_struct_record(context, builder, out_typ, data_vals, nulls)
        struct_record = context.make_helper(builder, out_typ)
        struct_record.meminfo = meminfo
        return struct_record._getvalue()

    return out_typ(struct_arr_typ, ind_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def struct_arr_getitem(arr, ind):
    if not isinstance(arr, StructArrayType):
        return

    if isinstance(ind, types.Integer):
        # TODO: warning if value is NA?
        def struct_arr_getitem_impl(arr, ind):  # pragma: no cover
            return struct_array_get_record(arr, ind)

        return struct_arr_getitem_impl
