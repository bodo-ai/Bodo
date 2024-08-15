# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Array implementation for structs of values.
Corresponds to Spark's StructType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Struct arrays: https://arrow.apache.org/docs/format/Columnar.html

The values are stored in contiguous data arrays; one array per field. For example:
A:             ["AA", "B", "C"]
B:             [1, 2, 4]
"""
import operator

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba import generated_jit
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_cast,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis
from numba.typed.typedobjectutils import _cast

import bodo
from bodo.libs import array_ext  # noqa: F401
from bodo.utils.cg_helpers import (
    gen_allocate_array,
    is_na_value,
)
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    get_array_getitem_scalar_type,
    get_overload_const_int,
    get_overload_const_str,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    unwrap_typeref,
)


class StructArrayType(types.ArrayCompatible):
    """Data type for arrays of structs"""

    data: tuple[types.ArrayCompatible, ...]
    names: tuple[str, ...]

    def __init__(self, data, names=None):
        # data is tuple of Array types
        # names is a tuple of field names
        assert isinstance(data, tuple) and all(
            bodo.utils.utils.is_array_typ(a, False) for a in data
        ), "Internal error in StructArrayType: Data does not have the correct format"
        if names is not None:
            assert (
                isinstance(names, tuple)
                and all(isinstance(a, str) for a in names)
                and len(names) == len(data)
            )
        else:
            names = tuple(f"f{i}" for i in range(len(data)))

        self.data = data
        self.names = names
        super(StructArrayType, self).__init__(
            name="StructArrayType({}, {})".format(data, names)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        # TODO: consider enabling dict return if possible
        # if types.is_homogeneous(*self.data):
        #     return types.DictType(bodo.string_type, self.data[0].dtype)

        # NOTE: the scalar type of most arrays is the same as dtype (e.g. int64), except
        # DatetimeArrayType and null_array_type which have different dtype objects.
        # Therefore, we have to use get_array_getitem_scalar_type instead of dtype here
        return StructType(
            tuple(get_array_getitem_scalar_type(t) for t in self.data), self.names
        )

    @classmethod
    def from_dict(cls, d):
        """create a StructArrayType from dict where keys are names and values are dtypes"""
        assert isinstance(d, dict)
        names = tuple(str(a) for a in d.keys())
        data = tuple(dtype_to_array_type(t) for t in d.values())
        return StructArrayType(data, names)

    def copy(self):
        return StructArrayType(self.data, self.names)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


class StructArrayPayloadType(types.Type):
    def __init__(self, data):
        assert isinstance(data, tuple) and all(
            bodo.utils.utils.is_array_typ(a, False) for a in data
        )
        self.data = data
        super(StructArrayPayloadType, self).__init__(
            name="StructArrayPayloadType({})".format(data)
        )

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(StructArrayPayloadType)
class StructArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            # Keeping the number of rows is necessary for arrays with no fields
            ("n_structs", types.int64),
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
    fn = cgutils.get_or_insert_function(
        mod,
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
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(
        builder, types.BaseTuple.from_types(struct_arr_type.data), payload.data
    )
    context.nrt.decref(builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap)

    builder.ret_void()
    return fn


def construct_struct_array(
    context, builder, struct_arr_type, n_structs, n_elems, dict_ref_arr=None, c=None
):
    """Creates meminfo and sets dtor, and allocates buffers for struct array"""
    # create payload type
    payload_type = StructArrayPayloadType(struct_arr_type.data)
    alloc_type = context.get_value_type(payload_type)
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
    payload.n_structs = n_structs

    # alloc data
    arrs = []
    curr_count_ind = 0
    ref_data = (
        _get_struct_arr_payload(context, builder, struct_arr_type, dict_ref_arr).data
        if dict_ref_arr
        else None
    )
    for data_ind, arr_typ in enumerate(struct_arr_type.data):
        n_nested_count_t = bodo.utils.transform.get_type_alloc_counts(arr_typ.dtype)
        n_all_elems = cgutils.pack_array(
            builder,
            [n_structs]
            + [
                builder.extract_value(n_elems, i)
                for i in range(curr_count_ind, curr_count_ind + n_nested_count_t)
            ],
        )
        ref_arg = builder.extract_value(ref_data, data_ind) if ref_data else None
        arr = gen_allocate_array(context, builder, arr_typ, n_all_elems, ref_arg, c)
        arrs.append(arr)
        curr_count_ind += n_nested_count_t

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

    return meminfo, payload.data, null_bitmap_ptr


@unbox(StructArrayType)
def unbox_struct_array(typ, val, c):
    """
    Unbox an array with struct values.
    """
    return bodo.libs.array.unbox_nested_array(typ, val, c)


def _get_struct_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a struct array value"""
    struct_array = context.make_helper(builder, arr_typ, arr)
    payload_type = StructArrayPayloadType(arr_typ.data)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, struct_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


@box(StructArrayType)
def box_struct_arr(typ, val, c):
    """box struct array into python objects."""
    return bodo.libs.array.box_nested_array(typ, val, c)


def _fix_nested_counts(nested_counts, struct_arr_type, nested_counts_type, builder):
    """make sure 'nested_counts' has -1 for all unknown alloc counts"""
    # subtracting one to account for the number of rows of the struct array itself
    n_elem_alloc_counts = (
        bodo.utils.transform.get_type_alloc_counts(struct_arr_type) - 1
    )
    # creating a tuple is not necessary if no nested count is needed
    if n_elem_alloc_counts == 0:
        return nested_counts

    if not isinstance(nested_counts_type, types.UniTuple):  # pragma: no cover
        nested_counts = cgutils.pack_array(
            builder,
            [lir.Constant(lir.IntType(64), -1) for _ in range(n_elem_alloc_counts)],
        )
    elif nested_counts_type.count < n_elem_alloc_counts:
        nested_counts = cgutils.pack_array(
            builder,
            [
                builder.extract_value(nested_counts, i)
                for i in range(nested_counts_type.count)
            ]
            + [
                lir.Constant(lir.IntType(64), -1)
                for _ in range(n_elem_alloc_counts - nested_counts_type.count)
            ],
        )
    return nested_counts


@intrinsic(prefer_literal=True)
def pre_alloc_struct_array(
    typingctx,
    num_structs_typ,
    nested_counts_typ,
    dtypes_typ,
    names_typ,
    dict_ref_arr_typ,
):
    assert isinstance(num_structs_typ, types.Integer) and isinstance(
        dtypes_typ, types.BaseTuple
    )
    if is_overload_none(names_typ):
        names = tuple(f"f{i}" for i in range(len(dtypes_typ)))
    else:
        names = tuple(get_overload_const_str(t) for t in names_typ.types)
    arr_typs = tuple(t.instance_type for t in dtypes_typ.types)
    struct_arr_type = StructArrayType(arr_typs, names)

    def codegen(context, builder, sig, args):
        num_structs, nested_counts, _, _, dict_ref_arr = args

        nested_counts_type = sig.args[1]
        nested_counts = _fix_nested_counts(
            nested_counts, struct_arr_type, nested_counts_type, builder
        )

        dict_ref_arg = None if is_overload_none(dict_ref_arr_typ) else dict_ref_arr
        meminfo, _, _ = construct_struct_array(
            context, builder, struct_arr_type, num_structs, nested_counts, dict_ref_arg
        )
        struct_array = context.make_helper(builder, struct_arr_type)
        struct_array.meminfo = meminfo
        return struct_array._getvalue()

    return (
        struct_arr_type(
            num_structs_typ, nested_counts_typ, dtypes_typ, names_typ, dict_ref_arr_typ
        ),
        codegen,
    )


def pre_alloc_struct_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_struct_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) > 0
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_struct_arr_ext_pre_alloc_struct_array = (
    pre_alloc_struct_array_equiv
)


class StructType(types.Type):
    """Data type for structs taken as scalars from struct arrays. A regular
    dictionary doesn't work in the general case since values can have different types.
    Very similar structure to StructArrayType, except that it holds scalar values and
    supports getitem/setitem of fields.
    """

    def __init__(self, data, names):
        # data is tuple of scalar types
        # names is a tuple of field names
        assert isinstance(data, tuple)
        assert (
            isinstance(names, tuple)
            and all(isinstance(a, str) for a in names)
            and len(names) == len(data)
        )

        self.data = data
        self.names = names
        super(StructType, self).__init__(name="StructType({}, {})".format(data, names))

    def unify(self, typingctx, other):
        """Unify struct types with same field names but different data types
        (e.g. optional vs non-optional type in BodoSQL)
        """
        if isinstance(other, StructType) and other.names == self.names:
            data = []
            for t1, t2 in zip(self.data, other.data):
                out_t = t1.unify(typingctx, t2)
                if out_t is None and (t1 is not None or t2 is not None):
                    return
                data.append(out_t)
            return StructType(tuple(data), self.names)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


class StructPayloadType(types.Type):
    def __init__(self, data):
        assert isinstance(data, tuple)
        self.data = data
        super(StructPayloadType, self).__init__(
            name="StructPayloadType({})".format(data)
        )

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(StructPayloadType)
class StructPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.BaseTuple.from_types(fe_type.data)),
            ("null_bitmap", types.UniTuple(types.int8, len(fe_type.data))),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(StructType)
class StructModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = StructPayloadType(fe_type.data)
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


def define_struct_dtor(context, builder, struct_type, payload_type):
    """
    Define destructor for struct type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(
        mod,
        fnty,
        name=".dtor.struct.{}.{}.".format(struct_type.data, struct_type.names),
    )

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    # decref all non-NA values
    for i in range(len(struct_type.data)):
        null_mask = builder.extract_value(payload.null_bitmap, i)
        not_na_cond = builder.icmp_unsigned(
            "==", null_mask, lir.Constant(null_mask.type, 1)
        )

        with builder.if_then(not_na_cond):
            val = builder.extract_value(payload.data, i)
            context.nrt.decref(builder, struct_type.data[i], val)

    # no need for null_bitmap since it is using primitive types

    builder.ret_void()
    return fn


def _get_struct_payload(context, builder, typ, struct):
    """get payload struct proxy for a struct value"""
    struct = context.make_helper(builder, typ, struct)
    payload_type = StructPayloadType(typ.data)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, struct.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload, meminfo_data_ptr


@unbox(StructType)
def unbox_struct(typ, val, c):
    """
    Unbox a dict into a struct.
    """
    context = c.context
    builder = c.builder

    # get pd.NA object to check for new NA kind
    mod_name = context.insert_const_string(builder.module, "pandas")
    pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
    C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

    data_vals = []
    nulls = []
    for i, t in enumerate(typ.data):
        field_val_obj = c.pyapi.dict_getitem_string(val, typ.names[i])
        # use NA as default
        null_ptr = cgutils.alloca_once_value(
            c.builder, context.get_constant(types.uint8, 0)
        )
        data_ptr = cgutils.alloca_once_value(
            c.builder, cgutils.get_null_value(context.get_value_type(t))
        )
        # check for NA
        is_na = is_na_value(builder, context, field_val_obj, C_NA)
        not_na_cond = builder.icmp_unsigned("!=", is_na, lir.Constant(is_na.type, 1))
        with builder.if_then(not_na_cond):
            builder.store(context.get_constant(types.uint8, 1), null_ptr)
            field_val = c.pyapi.to_native_value(t, field_val_obj).value
            builder.store(field_val, data_ptr)
        # no need to decref field_val_obj, dict_getitem_string returns borrowed ref
        data_vals.append(builder.load(data_ptr))
        nulls.append(builder.load(null_ptr))

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)

    meminfo = construct_struct(context, builder, typ, data_vals, nulls)
    struct = context.make_helper(builder, typ)
    struct.meminfo = meminfo
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(struct._getvalue(), is_error=is_error)


@box(StructType)
def box_struct(typ, val, c):
    """box structs into python dictionary objects"""
    out_dict = c.pyapi.dict_new(len(typ.data))
    payload, _ = _get_struct_payload(c.context, c.builder, typ, val)

    for i, val_typ in enumerate(typ.data):
        # set None as default
        c.pyapi.dict_setitem_string(out_dict, typ.names[i], c.pyapi.borrow_none())
        # check for not NA
        null_mask = c.builder.extract_value(payload.null_bitmap, i)
        not_na_cond = c.builder.icmp_unsigned(
            "==", null_mask, lir.Constant(null_mask.type, 1)
        )
        with c.builder.if_then(not_na_cond):
            # out_dict['field_name'] = value
            value = c.builder.extract_value(payload.data, i)
            c.context.nrt.incref(c.builder, val_typ, value)
            val_obj = c.pyapi.from_native_value(val_typ, value, c.env_manager)
            c.pyapi.dict_setitem_string(out_dict, typ.names[i], val_obj)
            c.pyapi.decref(val_obj)

    c.context.nrt.decref(c.builder, typ, val)
    return out_dict


@intrinsic(prefer_literal=True)
def init_struct(typingctx, data_typ, names_typ):
    """create a new struct from input data tuple and names."""
    names = tuple(get_overload_const_str(t) for t in names_typ.types)
    struct_type = StructType(data_typ.types, names)

    def codegen(context, builder, sig, args):
        data, _names = args
        # TODO: refactor to avoid duplication with construct_struct
        # create payload type
        payload_type = StructPayloadType(struct_type.data)
        alloc_type = context.get_value_type(payload_type)
        alloc_size = context.get_abi_sizeof(alloc_type)

        # define dtor
        dtor_fn = define_struct_dtor(context, builder, struct_type, payload_type)

        # create meminfo
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, alloc_size), dtor_fn
        )
        meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
        meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

        # set values in payload
        payload = cgutils.create_struct_proxy(payload_type)(context, builder)
        payload.data = data
        # assuming all values are non-NA
        # TODO: support setting NA values in this function (maybe new arg for mask)
        # NOTE: passing type to pack_array() is necessary in case value list is empty
        payload.null_bitmap = cgutils.pack_array(
            builder,
            [context.get_constant(types.uint8, 1) for _ in range(len(data_typ.types))],
            context.get_data_type(types.uint8),
        )

        builder.store(payload._getvalue(), meminfo_data_ptr)
        context.nrt.incref(builder, data_typ, data)

        struct = context.make_helper(builder, struct_type)
        struct.meminfo = meminfo
        return struct._getvalue()

    return struct_type(data_typ, names_typ), codegen


@generated_jit(nopython=True)
def init_struct_with_nulls(values, nulls, names):
    """
    Creates a struct and sets certain fields to null.

    Args:
        values: tuple of data items to pack into a struct
        nulls: array of booleans where True indicates that the
        current field is null.
        names: tuple of the names of each struct field

    Returns:
        The values packed into a struct with the specified names
        and the specified indices as nulls.
    """
    names_unwrapped = unwrap_typeref(names)
    assert isinstance(
        names_unwrapped, bodo.utils.typing.ColNamesMetaType
    ), f"Internal error in init_struct_with_nulls: 'names' must be a ColNamesMetaType. Got: {names}"
    names_tup = names_unwrapped.meta
    func_text = "def impl(values, nulls, names):\n"
    func_text += f"  s = init_struct(values, {names_tup})\n"
    for i in range(len(names_tup)):
        func_text += f"  if nulls[{i}]:\n"
        func_text += f"    set_struct_field_to_null(s, {i})\n"
    func_text += "  return s\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "init_struct": init_struct,
            "set_struct_field_to_null": set_struct_field_to_null,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@intrinsic
def get_struct_data(typingctx, struct_typ):
    """get data values of struct as tuple"""
    assert isinstance(struct_typ, StructType)

    def codegen(context, builder, sig, args):
        (struct,) = args
        payload, _ = _get_struct_payload(context, builder, struct_typ, struct)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return types.BaseTuple.from_types(struct_typ.data)(struct_typ), codegen


@intrinsic
def get_struct_null_bitmap(typingctx, struct_typ):
    """get null bitmap tuple of struct value"""
    assert isinstance(struct_typ, StructType)

    def codegen(context, builder, sig, args):
        (struct,) = args
        payload, _ = _get_struct_payload(context, builder, struct_typ, struct)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    ret_typ = types.UniTuple(types.int8, len(struct_typ.data))
    return ret_typ(struct_typ), codegen


@intrinsic(prefer_literal=True)
def set_struct_data(typingctx, struct_typ, field_ind_typ, val_typ):
    """set a field in struct to value. needs to replace the whole payload."""
    assert isinstance(struct_typ, StructType) and is_overload_constant_int(
        field_ind_typ
    )
    field_ind = get_overload_const_int(field_ind_typ)

    def codegen(context, builder, sig, args):
        (struct, _, val) = args
        payload, meminfo_data_ptr = _get_struct_payload(
            context, builder, struct_typ, struct
        )
        old_data = payload.data
        new_data = builder.insert_value(old_data, val, field_ind)
        data_tup_typ = types.BaseTuple.from_types(struct_typ.data)
        context.nrt.decref(builder, data_tup_typ, old_data)
        context.nrt.incref(builder, data_tup_typ, new_data)
        payload.data = new_data
        builder.store(payload._getvalue(), meminfo_data_ptr)
        return context.get_dummy_value()

    return types.none(struct_typ, field_ind_typ, val_typ), codegen


@intrinsic(prefer_literal=True)
def set_struct_field_to_null(typingctx, struct_typ, field_ind_typ):
    """set a field in struct null."""
    assert isinstance(struct_typ, StructType) and is_overload_constant_int(
        field_ind_typ
    )
    field_ind = get_overload_const_int(field_ind_typ)
    null_tup_typ = types.UniTuple(types.int8, len(struct_typ.names))
    data_tup_typ = types.BaseTuple.from_types(struct_typ.data)
    zero = lir.Constant(lir.IntType(8), 0)

    def codegen(context, builder, sig, args):
        (struct, _) = args
        payload, meminfo_data_ptr = _get_struct_payload(
            context, builder, struct_typ, struct
        )
        old_nulls = payload.null_bitmap
        new_nulls = builder.insert_value(old_nulls, zero, field_ind)
        context.nrt.decref(builder, null_tup_typ, old_nulls)
        context.nrt.incref(builder, null_tup_typ, new_nulls)

        # Decref old data element to avoid memory leak since
        # destructor ignores null elements
        old_data = payload.data
        data_elem_null = context.get_constant_null(data_tup_typ[field_ind])
        new_data = builder.insert_value(old_data, data_elem_null, field_ind)
        context.nrt.decref(builder, data_tup_typ, old_data)
        context.nrt.incref(builder, data_tup_typ, new_data)

        payload.null_bitmap = new_nulls
        payload.data = new_data
        builder.store(payload._getvalue(), meminfo_data_ptr)
        return context.get_dummy_value()

    return types.none(struct_typ, field_ind_typ), codegen


@lower_cast(StructType, StructType)
def cast_struct_type(context, builder, fromty, toty, val):
    """Support casting struct values where data elements could have different types
    and need casting.
    """
    payload, _ = _get_struct_payload(context, builder, fromty, val)

    # Create data and null values for output struct
    null_vals = []
    data_vals = []
    for i, val_typ in enumerate(fromty.data):
        # check for not NA
        null_mask = builder.extract_value(payload.null_bitmap, i)
        null_vals.append(null_mask)
        not_na_cond = builder.icmp_unsigned(
            "==", null_mask, lir.Constant(null_mask.type, 1)
        )
        data_val_ptr = cgutils.alloca_once_value(
            builder, context.get_constant_null(toty.data[i])
        )
        with builder.if_then(not_na_cond):
            data_val = builder.extract_value(payload.data, i)
            # Cast data element to target type if necessary
            if val_typ != toty.data[i]:
                new_data_val = context.cast(builder, data_val, val_typ, toty.data[i])
            else:
                new_data_val = data_val
            builder.store(new_data_val, data_val_ptr)
        data_vals.append(builder.load(data_val_ptr))

    meminfo = construct_struct(context, builder, toty, data_vals, null_vals)
    struct = context.make_helper(builder, toty)
    struct.meminfo = meminfo
    return struct._getvalue()


def _get_struct_field_ind(struct, ind, op):
    """find struct field index for 'ind' (a const str type) for operation 'op'.
    Raise error if not possible.
    """
    if not is_overload_constant_str(ind):  # pragma: no cover
        raise BodoError(
            "structs (from struct array) only support constant strings for {}, not {}".format(
                op, ind
            )
        )

    ind_str = get_overload_const_str(ind)
    if ind_str not in struct.names:  # pragma: no cover
        raise BodoError("Field {} does not exist in struct {}".format(ind_str, struct))

    return struct.names.index(ind_str)


def is_field_value_null(s, field_name):  # pragma: no cover
    pass


@overload(is_field_value_null, no_unliteral=True)
def overload_is_field_value_null(s, field_name):
    """return True if struct field is NA"""
    field_ind = _get_struct_field_ind(s, field_name, "element access (getitem)")
    return (
        lambda s, field_name: get_struct_null_bitmap(s)[field_ind] == 0
    )  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def struct_getitem(struct, ind):
    if not isinstance(struct, StructType):
        return

    field_ind = _get_struct_field_ind(struct, ind, "element access (getitem)")
    # TODO: warning if value is NA?
    return lambda struct, ind: get_struct_data(struct)[field_ind]  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def struct_setitem(struct, ind, val):
    if not isinstance(struct, StructType):
        return

    field_ind = _get_struct_field_ind(struct, ind, "item assignment (setitem)")
    field_typ = struct.data[field_ind]

    # TODO: set NA
    return lambda struct, ind, val: set_struct_data(
        struct, field_ind, _cast(val, field_typ)
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_struct_arr_len(struct):
    if isinstance(struct, StructType):
        num_fields = len(struct.data)
        return lambda struct: num_fields  # pragma: no cover


def construct_struct(context, builder, struct_type, values, nulls):
    """Creates meminfo and sets dtor and data for struct"""
    # create payload type
    payload_type = StructPayloadType(struct_type.data)
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_struct_dtor(context, builder, struct_type, payload_type)

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
        if types.is_homogeneous(*struct_type.data)
        else cgutils.pack_struct(builder, values)
    )

    payload.null_bitmap = cgutils.pack_array(builder, nulls, lir.IntType(8))

    builder.store(payload._getvalue(), meminfo_data_ptr)
    return meminfo


@intrinsic
def struct_array_get_struct(typingctx, struct_arr_typ, ind_typ):
    """get struct from struct array, e.g. A[i]
    Returns a dictionary of value types are the same, otherwise a StructType
    """
    assert isinstance(struct_arr_typ, StructArrayType) and isinstance(
        ind_typ, types.Integer
    )
    # NOTE: the scalar type of most arrays is the same as dtype (e.g. int64), except
    # DatetimeArrayType and null_array_type which have different dtype objects.
    # Therefore, we have to use get_array_getitem_scalar_type instead of dtype here
    data_types = tuple(get_array_getitem_scalar_type(d) for d in struct_arr_typ.data)
    # TODO: consider enabling dict return if possible
    # # return a regular dictionary if values have the same type, otherwise struct
    # if types.is_homogeneous(*struct_arr_typ.data):
    #     out_typ = types.DictType(bodo.string_type, data_types[0])
    # else:
    #     out_typ = StructType(data_types, struct_arr_typ.names)
    out_typ = StructType(data_types, struct_arr_typ.names)

    def codegen(context, builder, sig, args):
        struct_arr, ind = args

        payload = _get_struct_arr_payload(context, builder, struct_arr_typ, struct_arr)
        data_vals = []
        null_vals = []
        for i, arr_typ in enumerate(struct_arr_typ.data):
            arr_ptr = builder.extract_value(payload.data, i)

            na_val = context.compile_internal(
                builder,
                lambda arr, ind: np.uint8(0)
                if bodo.libs.array_kernels.isna(arr, ind)
                else np.uint8(1),
                types.uint8(arr_typ, types.int64),
                [arr_ptr, ind],
            )
            null_vals.append(na_val)

            # NOTE: the scalar type of most arrays is the same as dtype (e.g. int64), except
            # DatetimeArrayType and null_array_type which have different dtype objects.
            # Therefore, we have to use get_array_getitem_scalar_type instead of dtype
            data_val_ptr = cgutils.alloca_once_value(
                builder,
                context.get_constant_null(get_array_getitem_scalar_type(arr_typ)),
            )
            # check for not NA
            not_na_cond = builder.icmp_unsigned(
                "==", na_val, lir.Constant(na_val.type, 1)
            )
            with builder.if_then(not_na_cond):
                data_val = context.compile_internal(
                    builder,
                    lambda arr, ind: arr[ind],
                    get_array_getitem_scalar_type(arr_typ)(arr_typ, types.int64),
                    [arr_ptr, ind],
                )
                builder.store(data_val, data_val_ptr)
            data_vals.append(builder.load(data_val_ptr))

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

            dict_out = context.compile_internal(
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
            # decref values after use
            context.nrt.decref(builder, types.BaseTuple.from_types(data_types), val_tup)
            return dict_out

        meminfo = construct_struct(context, builder, out_typ, data_vals, null_vals)
        struct = context.make_helper(builder, out_typ)
        struct.meminfo = meminfo
        return struct._getvalue()

    return out_typ(struct_arr_typ, ind_typ), codegen


@intrinsic
def get_data(typingctx, arr_typ):
    """get data arrays of struct array as tuple"""
    assert isinstance(arr_typ, StructArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return types.BaseTuple.from_types(arr_typ.data)(arr_typ), codegen


@intrinsic
def get_null_bitmap(typingctx, arr_typ):
    """get null bitmap array of struct array"""
    assert isinstance(arr_typ, StructArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    return types.Array(types.uint8, 1, "C")(arr_typ), codegen


@intrinsic
def get_n_structs(typingctx, arr_typ):
    """get length of struct array"""
    assert isinstance(arr_typ, StructArrayType), "get_n_structs: struct array expected"

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
        return payload.n_structs

    return types.int64(arr_typ), codegen


@intrinsic(prefer_literal=True)
def init_struct_arr(typingctx, n_structs_t, data_typ, null_bitmap_typ, names_typ):
    """create a new struct array from input data array tuple, null bitmap, and names."""
    names = tuple(get_overload_const_str(t) for t in names_typ.types)
    struct_arr_type = StructArrayType(data_typ.types, names)

    def codegen(context, builder, sig, args):
        n_structs, data, null_bitmap, _names = args
        # TODO: refactor to avoid duplication with construct_struct
        # create payload type
        payload_type = StructArrayPayloadType(struct_arr_type.data)
        alloc_type = context.get_value_type(payload_type)
        alloc_size = context.get_abi_sizeof(alloc_type)

        # define dtor
        dtor_fn = define_struct_arr_dtor(
            context, builder, struct_arr_type, payload_type
        )

        # create meminfo
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, alloc_size), dtor_fn
        )
        meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
        meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

        # set values in payload
        payload = cgutils.create_struct_proxy(payload_type)(context, builder)
        payload.data = data
        payload.null_bitmap = null_bitmap
        payload.n_structs = n_structs
        builder.store(payload._getvalue(), meminfo_data_ptr)
        context.nrt.incref(builder, data_typ, data)
        context.nrt.incref(builder, null_bitmap_typ, null_bitmap)

        struct_array = context.make_helper(builder, struct_arr_type)
        struct_array.meminfo = meminfo
        return struct_array._getvalue()

    return struct_arr_type(types.int64, data_typ, null_bitmap_typ, names_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def struct_arr_getitem(arr, ind):
    if not isinstance(arr, StructArrayType):
        return

    if isinstance(ind, types.Integer):
        # TODO: warning if value is NA?
        def struct_arr_getitem_impl(arr, ind):  # pragma: no cover
            if ind < 0:
                ind += len(arr)
            return struct_array_get_struct(arr, ind)

        return struct_arr_getitem_impl

    # Boolean array handling.
    if is_list_like_index_type(ind) or isinstance(ind, types.SliceType):
        # other getitem cases return an array, so just call getitem on underlying arrays
        n_fields = len(arr.data)
        func_text = "def impl(arr, ind):\n"
        func_text += "  data = get_data(arr)\n"
        func_text += "  null_bitmap = get_null_bitmap(arr)\n"
        func_text += "  n = len(arr)\n"
        if is_list_like_index_type(ind) and ind.dtype == types.bool_:
            func_text += "  out_null_bitmap = get_new_null_mask_bool_index(null_bitmap, ind, n)\n"
            func_text += "  n_out = pd.Series(ind).sum()\n"
        elif is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
            func_text += (
                "  out_null_bitmap = get_new_null_mask_int_index(null_bitmap, ind, n)\n"
            )
            func_text += "  n_out = len(ind)\n"
        elif isinstance(ind, types.SliceType):
            func_text += "  out_null_bitmap = get_new_null_mask_slice_index(null_bitmap, ind, n)\n"
            func_text += (
                "  slice_idx = numba.cpython.unicode._normalize_slice(ind, n)\n"
            )
            func_text += "  n_out = numba.cpython.unicode._slice_span(slice_idx)\n"
        else:  # pragma: no cover
            raise BodoError("invalid index {} in struct array indexing".format(ind))
        func_text += (
            "  return init_struct_arr(n_out, ({}{}), out_null_bitmap, ({}{}))\n".format(
                ", ".join(
                    "ensure_contig_if_np(data[{}][ind])".format(i)
                    for i in range(n_fields)
                ),
                "," if n_fields else "",
                ", ".join("'{}'".format(name) for name in arr.names),
                "," if n_fields else "",
            )
        )
        loc_vars = {}
        exec(
            func_text,
            {
                "pd": pd,
                "numba": numba,
                "init_struct_arr": init_struct_arr,
                "get_data": get_data,
                "get_null_bitmap": get_null_bitmap,
                "ensure_contig_if_np": bodo.utils.conversion.ensure_contig_if_np,
                "get_new_null_mask_bool_index": bodo.utils.indexing.get_new_null_mask_bool_index,
                "get_new_null_mask_int_index": bodo.utils.indexing.get_new_null_mask_int_index,
                "get_new_null_mask_slice_index": bodo.utils.indexing.get_new_null_mask_slice_index,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    raise BodoError(
        f"getitem for StructArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def struct_arr_setitem(arr, ind, val):
    if not isinstance(arr, StructArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    if isinstance(ind, types.Integer):
        n_fields = len(arr.data)
        func_text = "def impl(arr, ind, val):\n"
        func_text += "  data = get_data(arr)\n"
        func_text += "  null_bitmap = get_null_bitmap(arr)\n"
        func_text += "  set_bit_to_arr(null_bitmap, ind, 1)\n"
        for i in range(n_fields):
            if isinstance(val, StructType):
                func_text += "  if is_field_value_null(val, '{}'):\n".format(
                    arr.names[i]
                )
                func_text += (
                    "    bodo.libs.array_kernels.setna(data[{}], ind)\n".format(i)
                )
                func_text += "  else:\n"
                func_text += "    data[{}][ind] = val['{}']\n".format(i, arr.names[i])
            else:
                func_text += "  data[{}][ind] = val['{}']\n".format(i, arr.names[i])

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "get_data": get_data,
                "get_null_bitmap": get_null_bitmap,
                "set_bit_to_arr": bodo.libs.int_arr_ext.set_bit_to_arr,
                "is_field_value_null": is_field_value_null,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    # slice case (used in unboxing)
    if isinstance(ind, types.SliceType):
        # set data arrays and null bitmap
        n_fields = len(arr.data)
        func_text = "def impl(arr, ind, val):\n"
        func_text += "  data = get_data(arr)\n"
        func_text += "  null_bitmap = get_null_bitmap(arr)\n"
        func_text += "  val_data = get_data(val)\n"
        func_text += "  val_null_bitmap = get_null_bitmap(val)\n"
        func_text += "  setitem_slice_index_null_bits(null_bitmap, val_null_bitmap, ind, len(arr))\n"
        for i in range(n_fields):
            func_text += "  data[{0}][ind] = val_data[{0}]\n".format(i)

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "get_data": get_data,
                "get_null_bitmap": get_null_bitmap,
                "set_bit_to_arr": bodo.libs.int_arr_ext.set_bit_to_arr,
                "setitem_slice_index_null_bits": bodo.utils.indexing.setitem_slice_index_null_bits,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    raise BodoError(
        "only setitem with scalar/slice index is currently supported for struct arrays"
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_struct_arr_len(A):
    if isinstance(A, StructArrayType):
        if len(A.data) == 0:
            return lambda A: get_n_structs(A)  # pragma: no cover
        return lambda A: len(get_data(A)[0])  # pragma: no cover


@overload_attribute(StructArrayType, "shape")
def overload_struct_arr_shape(A):
    if len(A.data) == 0:
        return lambda A: (get_n_structs(A),)  # pragma: no cover
    return lambda A: (len(get_data(A)[0]),)  # pragma: no cover


@overload_attribute(StructArrayType, "dtype")
def overload_struct_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@overload_attribute(StructArrayType, "ndim")
def overload_struct_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(StructArrayType, "nbytes")
def overload_struct_arr_nbytes(A):
    func_text = "def impl(A):\n"
    func_text += "  total_nbytes = 0\n"
    func_text += "  data = get_data(A)\n"
    for i in range(len(A.data)):
        func_text += f"  total_nbytes += data[{i}].nbytes\n"
    func_text += "  total_nbytes += get_null_bitmap(A).nbytes\n"
    func_text += "  return total_nbytes\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "get_data": get_data,
            "get_null_bitmap": get_null_bitmap,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


@overload_method(StructArrayType, "copy", no_unliteral=True)
def overload_struct_arr_copy(A):
    names = A.names

    def copy_impl(A):  # pragma: no cover
        data = get_data(A)
        null_bitmap = get_null_bitmap(A)
        out_data_arrs = bodo.libs.struct_arr_ext.copy_arr_tup(data)
        out_null_bitmap = null_bitmap.copy()

        return init_struct_arr(len(A), out_data_arrs, out_null_bitmap, names)

    return copy_impl


def copy_arr_tup(arrs):  # pragma: no cover
    return tuple(a.copy() for a in arrs)


@overload(copy_arr_tup, no_unliteral=True)
def copy_arr_tup_overload(arrs):
    """
    Generate copy on a tuple of arrays and return the result.
    """
    count = arrs.count
    func_text = "def f(arrs):\n"
    func_text += "  return ({}{})\n".format(
        ",".join("arrs[{}].copy()".format(i) for i in range(count)),
        "," if count else "",
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["f"]
    return impl


def scalar_to_struct_array(scalar_val, length, _arr_typ):
    pass


@overload(scalar_to_struct_array)
def overload_scalar_to_struct_array(scalar_val, length, _arr_typ):
    """
    Create an StructArray of length `length` by repeating scalar_val `length` times

    Args:
        scalar_val (StructType): The struct value to be repeated
        length (int): Length of the output StructArray
        _arr_typ (types.Type): StructArrayType for output
    Returns:
        An StructArray of length `length`
    """

    arr_type = unwrap_typeref(_arr_typ)
    dtypes = arr_type.data
    names = arr_type.names

    def impl(scalar_val, length, _arr_typ):  # pragma: no cover
        out_arr = pre_alloc_struct_array(length, (-1,), dtypes, names, None)
        for i in range(length):
            out_arr[i] = scalar_val
        return out_arr

    return impl
