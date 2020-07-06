# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for structs of values.
Corresponds to Spark's StructType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Struct arrays: https://arrow.apache.org/docs/format/Columnar.html

The values are stored in contiguous data arrays; one array per field. For example:
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
from numba.typed.typedobjectutils import _cast

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

    @property
    def dtype(self):
        return StructType(tuple(t.dtype for t in self.data), self.names)

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
    context, builder, struct_arr_type, n_structs, n_elems, c=None
):
    """Creates meminfo and sets dtor, and allocates buffers for struct array
    """
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

    # alloc data
    arrs = []
    curr_count_ind = 0
    for arr_typ in struct_arr_type.data:
        n_nested_count_t = bodo.utils.transform.get_type_alloc_counts(arr_typ.dtype)
        n_all_elems = cgutils.pack_array(
            builder,
            [n_structs]
            + [
                builder.extract_value(n_elems, i)
                for i in range(curr_count_ind, curr_count_ind + n_nested_count_t)
            ],
        )
        arr = gen_allocate_array(context, builder, arr_typ, n_all_elems, c)
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


def _get_C_API_ptrs(c, data_tup, data_typ, names):
    """convert struct array info into pointers to pass to C API
    """

    data_ptrs = []
    assert len(data_typ) > 0
    # TODO: support non-Numpy arrays
    for i, arr_typ in enumerate(data_typ):
        arr_ptr = c.builder.extract_value(data_tup, i)
        arr = c.context.make_array(arr_typ)(c.context, c.builder, value=arr_ptr)
        data_ptrs.append(arr.data)

    # get pointer to a tuple of data pointers to pass to C
    data_ptr_tup = (
        cgutils.pack_array(c.builder, data_ptrs)
        if types.is_homogeneous(*data_typ)
        else cgutils.pack_struct(c.builder, data_ptrs)
    )
    data_ptr_tup_ptr = cgutils.alloca_once_value(c.builder, data_ptr_tup)
    # get pointer to a tuple of c type enums to pass to C
    c_types = [
        c.context.get_constant(types.int32, bodo.utils.utils.numba_to_c_type(a.dtype))
        for a in data_typ
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

    # can be handled in C if all data arrays are Numpy and in handled dtypes
    handle_in_c = all(
        isinstance(t, types.Array)
        and t.dtype in (types.int64, types.float64, types.bool_, datetime_date_type,)
        for t in typ.data
    )

    if handle_in_c:
        n_elems = cgutils.pack_array(c.builder, [], lir.IntType(64))
    else:
        n_elems_all = get_array_elem_counts(c, c.builder, c.context, val, typ)
        # ignore first value in tuple which is array length
        n_elems = cgutils.pack_array(
            c.builder,
            [
                c.builder.extract_value(n_elems_all, i)
                for i in range(1, n_elems_all.type.count)
            ],
            lir.IntType(64),
        )

    # create struct array
    meminfo, data_tup, null_bitmap_ptr = construct_struct_array(
        c.context, c.builder, typ, n_structs, n_elems, c
    )

    if handle_in_c:
        data_ptr_tup_ptr, c_types_ptr, field_names_ptr = _get_C_API_ptrs(
            c, data_tup, typ.data, typ.names
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
    else:
        _unbox_struct_array_generic(typ, val, c, n_structs, data_tup, null_bitmap_ptr)

    struct_array = c.context.make_helper(c.builder, typ)
    struct_array.meminfo = meminfo

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(struct_array._getvalue(), is_error=is_error)


def _unbox_struct_array_generic(typ, val, c, n_structs, data_tup, null_bitmap_ptr):
    """unbox struct array using generic Numba unboxing to handle all item types
    that can be unboxed.
    """
    context = c.context
    builder = c.builder

    # TODO: error checking for pyapi calls
    # get pd.NA object to check for new NA kind
    mod_name = context.insert_const_string(builder.module, "pandas")
    pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
    C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

    # pseudocode for code generation:
    # for i in range(len(A)):
    #   dict_obj = A[i]
    #   if isna(dict_obj):
    #     set null_bitmap i'th bit to 0
    #   else:
    #     set null_bitmap i'th bit to 1
    #     for j, name in enumerate(field_names):
    #        val_obj = dict_obj[name]
    #        A.data[j] = unbox(val_obj)

    # for each struct
    with cgutils.for_range(builder, n_structs) as loop:
        struct_ind = loop.index

        # dict_obj = A[i]
        dict_obj = seq_getitem(builder, context, val, struct_ind)

        # set NA bit to 0
        set_bitmap_bit(builder, null_bitmap_ptr, struct_ind, 0)
        # set field values to NA (will be overwritten below if not NA)
        # this approach is used to avoid if/else which can be buggy
        for j in range(len(typ.data)):
            arr_typ = typ.data[j]
            data_arr = builder.extract_value(data_tup, j)

            def set_na(data_arr, i):
                bodo.ir.join.setitem_arr_nan(data_arr, i)

            sig = types.none(arr_typ, types.int64)
            _is_error, _res = c.pyapi.call_jit_code(set_na, sig, [data_arr, struct_ind])

        # check for NA
        is_na = is_na_value(builder, context, dict_obj, C_NA)
        is_na_cond = builder.icmp_unsigned("!=", is_na, lir.Constant(is_na.type, 1))
        with builder.if_then(is_na_cond):
            # set NA bit to 1
            set_bitmap_bit(builder, null_bitmap_ptr, struct_ind, 1)
            for j in range(len(typ.data)):
                arr_typ = typ.data[j]
                val_obj = c.pyapi.dict_getitem_string(dict_obj, typ.names[j])
                # check for NA
                is_na = is_na_value(builder, context, val_obj, C_NA)
                is_na_cond = builder.icmp_unsigned(
                    "!=", is_na, lir.Constant(is_na.type, 1)
                )
                with builder.if_then(is_na_cond):
                    val_obj = to_arr_obj_if_list_obj(
                        c, context, builder, val_obj, arr_typ.dtype
                    )
                    field_val = c.pyapi.to_native_value(arr_typ.dtype, val_obj).value
                    data_arr = builder.extract_value(data_tup, j)

                    def set_data(data_arr, i, field_val):
                        data_arr[i] = field_val

                    sig = types.none(arr_typ, types.int64, arr_typ.dtype)
                    _is_error, _res = c.pyapi.call_jit_code(
                        set_data, sig, [data_arr, struct_ind, field_val]
                    )
                    c.context.nrt.decref(builder, arr_typ.dtype, field_val)
                # no need to decref val_obj, dict_getitem_string returns borrowed ref
        c.pyapi.decref(dict_obj)

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)


def _get_struct_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a struct array value
    """
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
    """box packed native representation of list of item array into python objects
    """

    payload = _get_struct_arr_payload(c.context, c.builder, typ, val)
    _is_error, length = c.pyapi.call_jit_code(lambda A: len(A), types.int64(typ), [val])
    null_bitmap_ptr = c.context.make_helper(
        c.builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap
    ).data

    # can be handled in C if all data arrays are Numpy and in handled dtypes
    handle_in_c = all(
        isinstance(t, types.Array)
        and t.dtype in (types.int64, types.float64, types.bool_, datetime_date_type,)
        for t in typ.data
    )

    if handle_in_c:
        data_ptr_tup_ptr, c_types_ptr, field_names_ptr = _get_C_API_ptrs(
            c, payload.data, typ.data, typ.names
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
    else:
        arr = _box_struct_array_generic(typ, c, length, payload.data, null_bitmap_ptr)

    c.context.nrt.decref(c.builder, typ, val)
    return arr


def _box_struct_array_generic(typ, c, length, data_arrs_tup, null_bitmap_ptr):
    """box struct array using generic Numba boxing to handle all item types
    that can be boxed.
    """
    context = c.context
    builder = c.builder
    # TODO: error checking for pyapi calls

    # pseudocode for code generation:
    # out_arr = np.ndarray(n, np.object_)
    # for i in range(n):
    #   if isna(A[i]):
    #     out_arr[i] = np.nan
    #   else:
    #     dict_obj = dict_new(n_items)
    #     for j in range(n_items):
    #        dict_obj[field_names[i]] = A.data[j]

    # create array of objects with num_items shape
    mod_name = context.insert_const_string(builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype_obj = c.pyapi.object_getattr_string(np_class_obj, "object_")
    num_items_obj = c.pyapi.long_from_longlong(length)
    out_arr = c.pyapi.call_method(np_class_obj, "ndarray", (num_items_obj, dtype_obj))
    # get np.nan to set NA
    nan_obj = c.pyapi.object_getattr_string(np_class_obj, "nan")

    # for each struct
    with cgutils.for_range(builder, length) as loop:
        struct_ind = loop.index
        # A[i] = np.nan
        pyarray_setitem(builder, context, out_arr, struct_ind, nan_obj)
        # check for NA
        na_bit = get_bitmap_bit(builder, null_bitmap_ptr, struct_ind)
        not_na_cond = builder.icmp_unsigned(
            "!=", na_bit, lir.Constant(lir.IntType(8), 0)
        )
        with builder.if_then(not_na_cond):
            # create dict obj
            dict_obj = c.pyapi.dict_new(len(typ.data))

            # set field values
            for i, arr_typ in enumerate(typ.data):
                # set NA as default
                c.pyapi.dict_setitem_string(dict_obj, typ.names[i], nan_obj)

                # is_not_na_val = not isna(data_arr, struct_ind)
                data_arr = c.builder.extract_value(data_arrs_tup, i)
                _is_error, is_not_na_val = c.pyapi.call_jit_code(
                    lambda data_arr, ind: not bodo.libs.array_kernels.isna(
                        data_arr, ind
                    ),
                    types.bool_(arr_typ, types.int64),
                    [data_arr, struct_ind],
                )

                # if value is not NA
                with builder.if_then(is_not_na_val):
                    # field_val = data_arr[struct_ind]
                    _is_error, field_val = c.pyapi.call_jit_code(
                        lambda data_arr, ind: data_arr[ind],
                        (arr_typ.dtype)(arr_typ, types.int64),
                        [data_arr, struct_ind],
                    )

                    # dict_obj[field_name] = field_val
                    field_val_obj = c.pyapi.from_native_value(
                        arr_typ.dtype, field_val, c.env_manager
                    )
                    c.pyapi.dict_setitem_string(dict_obj, typ.names[i], field_val_obj)
                    c.pyapi.decref(field_val_obj)
            pyarray_setitem(builder, context, out_arr, struct_ind, dict_obj)
            c.pyapi.decref(dict_obj)

    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(num_items_obj)
    c.pyapi.decref(nan_obj)
    return out_arr


@intrinsic
def pre_alloc_struct_array(
    typingctx, num_structs_typ, nested_counts_typ, dtypes_typ, names_typ=None
):
    assert isinstance(num_structs_typ, types.Integer) and isinstance(
        dtypes_typ, types.BaseTuple
    )
    names = tuple(get_overload_const_str(t) for t in names_typ.types)
    arr_typs = tuple(t.instance_type for t in dtypes_typ.types)
    struct_arr_type = StructArrayType(arr_typs, names)

    def codegen(context, builder, sig, args):
        num_structs, nested_counts, _, _ = args
        meminfo, _, _ = construct_struct_array(
            context, builder, struct_arr_type, num_structs, nested_counts
        )
        struct_array = context.make_helper(builder, struct_arr_type)
        struct_array.meminfo = meminfo
        return struct_array._getvalue()

    return (
        struct_arr_type(num_structs_typ, nested_counts_typ, dtypes_typ, names_typ),
        codegen,
    )


def pre_alloc_struct_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_struct_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 4 and not kws
    return args[0], []


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
        super(StructType, self).__init__(name="StructType({}, {})".format(data, names))


class StructPayloadType(types.Type):
    def __init__(self, data):
        assert isinstance(data, tuple) and all(
            not bodo.utils.utils.is_array_typ(a, False) for a in data
        )
        self.data = data
        super(StructPayloadType, self).__init__(
            name="StructPayloadType({})".format(data)
        )


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
    fn = mod.get_or_insert_function(
        fnty, name=".dtor.struct.{}.{}.".format(struct_type.data, struct_type.names),
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
    """get payload struct proxy for a struct value
    """
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
    """box structs into python dictionary objects
    """
    out_dict = c.pyapi.dict_new(len(typ.data))
    payload, _ = _get_struct_payload(c.context, c.builder, typ, val)

    assert len(typ.data) > 0
    # TODO: support NAs
    for i, val_typ in enumerate(typ.data):
        value = c.builder.extract_value(payload.data, i)
        val_obj = c.pyapi.from_native_value(val_typ, value, c.env_manager)
        c.pyapi.dict_setitem_string(out_dict, typ.names[i], val_obj)
        c.pyapi.decref(val_obj)

    c.context.nrt.decref(c.builder, typ, val)
    return out_dict


@intrinsic
def init_struct(typingctx, data_typ, names_typ=None):
    """create a new struct from input data tuple and names.
    """
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
        payload.null_bitmap = cgutils.pack_array(
            builder,
            [context.get_constant(types.uint8, 1) for _ in range(len(data_typ.types))],
        )

        builder.store(payload._getvalue(), meminfo_data_ptr)
        context.nrt.incref(builder, data_typ, data)

        struct = context.make_helper(builder, struct_type)
        struct.meminfo = meminfo
        return struct._getvalue()

    return struct_type(data_typ, names_typ), codegen


@intrinsic
def get_struct_data(typingctx, struct_typ=None):
    """get data values of struct as tuple
    """
    assert isinstance(struct_typ, StructType)

    def codegen(context, builder, sig, args):
        (struct,) = args
        payload, _ = _get_struct_payload(context, builder, struct_typ, struct)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return types.BaseTuple.from_types(struct_typ.data)(struct_typ), codegen


@intrinsic
def get_struct_null_bitmap(typingctx, struct_typ=None):
    """get null bitmap tuple of struct value
    """
    assert isinstance(struct_typ, StructType)

    def codegen(context, builder, sig, args):
        (struct,) = args
        payload, _ = _get_struct_payload(context, builder, struct_typ, struct)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    ret_typ = types.UniTuple(types.int8, len(struct_typ.data))
    return ret_typ(struct_typ), codegen


@intrinsic
def set_struct_data(typingctx, struct_typ, field_ind_typ, val_typ=None):
    """set a field in struct to value. needs to replace the whole payload.
    """
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


@overload(is_field_value_null)
def overload_is_field_value_null(s, field_name):
    """return True if struct field is NA
    """
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
def struct_getitem(struct, ind, val):
    if not isinstance(struct, StructType):
        return

    field_ind = _get_struct_field_ind(struct, ind, "item assignment (setitem)")
    field_typ = struct.data[field_ind]

    # TODO: set NA
    return lambda struct, ind, val: set_struct_data(
        struct, field_ind, _cast(val, field_typ)
    )  # pragma: no cover


def construct_struct(context, builder, struct_type, values, nulls):
    """Creates meminfo and sets dtor and data for struct
    """
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

    payload.null_bitmap = cgutils.pack_array(builder, nulls)

    builder.store(payload._getvalue(), meminfo_data_ptr)
    return meminfo


@intrinsic
def struct_array_get_struct(typingctx, struct_arr_typ, ind_typ=None):
    """get struct from struct array, e.g. A[i]
    Returns a dictionary of value types are the same, otherwise a StructType
    """
    assert isinstance(struct_arr_typ, StructArrayType) and isinstance(
        ind_typ, types.Integer
    )
    data_types = tuple(d.dtype for d in struct_arr_typ.data)
    # return a regular dictionary if values have the same type, otherwise struct
    if types.is_homogeneous(*struct_arr_typ.data):
        out_typ = types.DictType(bodo.string_type, data_types[0])
    else:
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

            data_val = context.compile_internal(
                builder,
                lambda arr, ind: arr[ind],
                arr_typ.dtype(arr_typ, types.int64),
                [arr_ptr, ind],
            )
            data_vals.append(data_val)

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

        meminfo = construct_struct(context, builder, out_typ, data_vals, null_vals)
        struct = context.make_helper(builder, out_typ)
        struct.meminfo = meminfo
        return struct._getvalue()

    return out_typ(struct_arr_typ, ind_typ), codegen


@intrinsic
def get_data(typingctx, arr_typ=None):
    """get data arrays of struct array as tuple
    """
    assert isinstance(arr_typ, StructArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return types.BaseTuple.from_types(arr_typ.data)(arr_typ), codegen


@intrinsic
def get_null_bitmap(typingctx, arr_typ=None):
    """get null bitmap array of struct array
    """
    assert isinstance(arr_typ, StructArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    return types.Array(types.uint8, 1, "C")(arr_typ), codegen


@intrinsic
def init_struct_arr(typingctx, data_typ, null_bitmap_typ, names_typ=None):
    """create a new struct array from input data array tuple, null bitmap, and names.
    """
    names = tuple(get_overload_const_str(t) for t in names_typ.types)
    struct_arr_type = StructArrayType(data_typ.types, names)

    def codegen(context, builder, sig, args):
        data, null_bitmap, _names = args
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
        builder.store(payload._getvalue(), meminfo_data_ptr)
        context.nrt.incref(builder, data_typ, data)
        context.nrt.incref(builder, null_bitmap_typ, null_bitmap)

        struct_array = context.make_helper(builder, struct_arr_type)
        struct_array.meminfo = meminfo
        return struct_array._getvalue()

    return struct_arr_type(data_typ, null_bitmap_typ, names_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def struct_arr_getitem(arr, ind):
    if not isinstance(arr, StructArrayType):
        return

    if isinstance(ind, types.Integer):
        # TODO: warning if value is NA?
        def struct_arr_getitem_impl(arr, ind):  # pragma: no cover
            return struct_array_get_struct(arr, ind)

        return struct_arr_getitem_impl

    # other getitem cases return an array, so just call getitem on underlying arrays
    n_fields = len(arr.data)
    func_text = "def impl(arr, ind):\n"
    func_text += "  data = get_data(arr)\n"
    func_text += "  null_bitmap = get_null_bitmap(arr)\n"
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        func_text += "  out_null_bitmap = get_new_null_mask_bool_index(null_bitmap, ind, len(data[0]))\n"
    elif is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        func_text += "  out_null_bitmap = get_new_null_mask_int_index(null_bitmap, ind, len(data[0]))\n"
    elif isinstance(ind, types.SliceType):
        func_text += "  out_null_bitmap = get_new_null_mask_slice_index(null_bitmap, ind, len(data[0]))\n"
    else:  # pragma: no cover
        raise BodoError("invalid index {} in struct array indexing".format(ind))
    func_text += "  return init_struct_arr(({},), out_null_bitmap, ({},))\n".format(
        ", ".join(
            "ensure_contig_if_np(data[{}][ind])".format(i) for i in range(n_fields)
        ),
        ", ".join("'{}'".format(name) for name in arr.names),
    )
    loc_vars = {}
    exec(
        func_text,
        {
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


@overload(operator.setitem, no_unliteral=True)
def struct_arr_setitem(arr, ind, val):
    if not isinstance(arr, StructArrayType):
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
                func_text += "    bodo.ir.join.setitem_arr_nan(data[{}], ind)\n".format(
                    i
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


@overload(len, no_unliteral=True)
def overload_struct_arr_len(A):
    if isinstance(A, StructArrayType):
        return lambda A: len(get_data(A)[0])


@overload_attribute(StructArrayType, "shape")
def overload_struct_arr_shape(A):
    return lambda A: (len(get_data(A)[0]),)


@overload_attribute(StructArrayType, "ndim")
def overload_struct_arr_ndim(A):
    return lambda A: 1


@overload_method(StructArrayType, "copy", no_unliteral=True)
def overload_struct_arr_copy(A):
    names = A.names

    def copy_impl(A):  # pragma: no cover
        data = get_data(A)
        null_bitmap = get_null_bitmap(A)
        out_data_arrs = bodo.ir.join.copy_arr_tup(data)
        out_null_bitmap = null_bitmap.copy()

        return init_struct_arr(out_data_arrs, out_null_bitmap, names)

    return copy_impl
