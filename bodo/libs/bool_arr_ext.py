# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Nullable boolean array that stores data in Numpy format (1 byte per value)
but nulls are stored in bit arrays (1 bit per value) similar to Arrow's nulls.
Pandas converts boolean array to object when NAs are introduced.
"""
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import hstr_ext
from bodo.libs.str_arr_ext import string_array_type
from bodo.utils.typing import is_list_like_index_type

ll.add_symbol("is_bool_array", hstr_ext.is_bool_array)
ll.add_symbol("is_pd_boolean_array", hstr_ext.is_pd_boolean_array)
ll.add_symbol("unbox_bool_array_obj", hstr_ext.unbox_bool_array_obj)
from bodo.utils.indexing import (
    array_getitem_bool_index,
    array_getitem_int_index,
    array_getitem_slice_index,
    array_setitem_bool_index,
    array_setitem_int_index,
    array_setitem_slice_index,
)
from bodo.utils.typing import (
    BodoError,
    is_iterable_type,
    is_overload_false,
    is_overload_true,
    parse_dtype,
)


class BooleanArrayType(types.ArrayCompatible):
    def __init__(self):
        super(BooleanArrayType, self).__init__(name="BooleanArrayType()")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.bool_

    def copy(self):
        return BooleanArrayType()


boolean_array = BooleanArrayType()


@typeof_impl.register(pd.arrays.BooleanArray)
def typeof_boolean_array(val, c):
    return boolean_array


data_type = types.Array(types.bool_, 1, "C")
nulls_type = types.Array(types.uint8, 1, "C")


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(BooleanArrayType)
class BooleanArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", data_type),
            ("null_bitmap", nulls_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BooleanArrayType, "data", "_data")
make_attribute_wrapper(BooleanArrayType, "null_bitmap", "_null_bitmap")


# dtype object for pd.BooleanDtype()
class BooleanDtype(types.Number):
    """
    Type class associated with pandas Boolean dtype pd.BooleanDtype()
    """

    def __init__(self):
        self.dtype = types.bool_
        super(BooleanDtype, self).__init__("BooleanDtype")


boolean_dtype = BooleanDtype()


register_model(BooleanDtype)(models.OpaqueModel)


@box(BooleanDtype)
def box_boolean_dtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    res = c.pyapi.call_method(pd_class_obj, "BooleanDtype", ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(BooleanDtype)
def unbox_boolean_dtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


typeof_impl.register(pd.BooleanDtype)(lambda a, b: boolean_dtype)
type_callable(pd.BooleanDtype)(lambda c: lambda: boolean_dtype)
lower_builtin(pd.BooleanDtype)(lambda c, b, s, a: c.get_dummy_value())


@numba.njit
def gen_full_bitmap(n):  # pragma: no cover
    n_bytes = (n + 7) >> 3
    return np.full(n_bytes, 255, np.uint8)


def call_func_in_unbox(func, args, arg_typs, c):
    func_typ = c.context.typing_context.resolve_value_type(func)
    func_sig = func_typ.get_call_type(c.context.typing_context, arg_typs, {})
    func_impl = c.context.get_function(func_typ, func_sig)

    # XXX: workaround wrapper must be used due to call convention changes
    fnty = c.context.call_conv.get_function_type(func_sig.return_type, func_sig.args)
    mod = c.builder.module
    fn = lir.Function(mod, fnty, name=mod.get_unique_name(".func_conv"))
    fn.linkage = "internal"
    inner_builder = lir.IRBuilder(fn.append_basic_block())
    inner_args = c.context.call_conv.decode_arguments(inner_builder, func_sig.args, fn)
    h = func_impl(inner_builder, inner_args)
    c.context.call_conv.return_value(inner_builder, h)

    status, retval = c.context.call_conv.call_function(
        c.builder, fn, func_sig.return_type, func_sig.args, args
    )
    # TODO: check status?
    return retval


@unbox(BooleanArrayType)
def unbox_bool_array(typ, obj, c):
    """
    Convert a pd.arrays.BooleanArray or a Numpy array object to a native BooleanArray
    structure. The array's dtype can be bool or object, depending on the presense of
    nans.
    """
    n_obj = c.pyapi.call_method(obj, "__len__", ())
    n = c.pyapi.long_as_longlong(n_obj)
    c.pyapi.decref(n_obj)

    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn_bool = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="is_bool_array"
    )

    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="is_pd_boolean_array"
    )

    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    is_pd_bool = c.builder.call(fn, [obj])
    cond_pd = c.builder.icmp_unsigned("!=", is_pd_bool, is_pd_bool.type(0))
    with c.builder.if_else(cond_pd) as (pd_then, pd_otherwise):
        with pd_then:
            data_obj = c.pyapi.object_getattr_string(obj, "_data")
            bool_arr.data = c.pyapi.to_native_value(
                types.Array(types.bool_, 1, "C"), data_obj
            ).value

            mask_arr_obj = c.pyapi.object_getattr_string(obj, "_mask")
            mask_arr = c.pyapi.to_native_value(
                types.Array(types.bool_, 1, "C"), mask_arr_obj
            ).value
            n_bytes = c.builder.udiv(
                c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
                lir.Constant(lir.IntType(64), 8),
            )
            mask_arr_struct = c.context.make_array(types.Array(types.bool_, 1, "C"))(
                c.context, c.builder, mask_arr
            )
            bitmap_arr_struct = bodo.utils.utils._empty_nd_impl(
                c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bytes]
            )

            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(64),
                ],
            )
            fn = cgutils.get_or_insert_function(
                c.builder.module, fnty, name="mask_arr_to_bitmap"
            )
            c.builder.call(fn, [bitmap_arr_struct.data, mask_arr_struct.data, n])
            bool_arr.null_bitmap = bitmap_arr_struct._getvalue()
            # clean up native mask array after creating bitmap from it
            c.context.nrt.decref(c.builder, types.Array(types.bool_, 1, "C"), mask_arr)
            c.pyapi.decref(data_obj)
            c.pyapi.decref(mask_arr_obj)

        with pd_otherwise:
            is_bool_dtype = c.builder.call(fn_bool, [obj])
            cond = c.builder.icmp_unsigned("!=", is_bool_dtype, is_bool_dtype.type(0))
            with c.builder.if_else(cond) as (then, otherwise):
                with then:
                    # array is bool
                    bool_arr.data = c.pyapi.to_native_value(
                        types.Array(types.bool_, 1, "C"), obj
                    ).value
                    bool_arr.null_bitmap = call_func_in_unbox(
                        gen_full_bitmap, (n,), (types.int64,), c
                    )
                with otherwise:
                    # array is object
                    # allocate data
                    bool_arr.data = bodo.utils.utils._empty_nd_impl(
                        c.context, c.builder, types.Array(types.bool_, 1, "C"), [n]
                    )._getvalue()
                    # allocate bitmap
                    n_bytes = c.builder.udiv(
                        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
                        lir.Constant(lir.IntType(64), 8),
                    )
                    bool_arr.null_bitmap = bodo.utils.utils._empty_nd_impl(
                        c.context,
                        c.builder,
                        types.Array(types.uint8, 1, "C"),
                        [n_bytes],
                    )._getvalue()
                    # get array pointers for data and bitmap
                    data_ptr = c.context.make_array(types.Array(types.bool_, 1, "C"))(
                        c.context, c.builder, bool_arr.data
                    ).data
                    bitmap_ptr = c.context.make_array(types.Array(types.uint8, 1, "C"))(
                        c.context, c.builder, bool_arr.null_bitmap
                    ).data
                    fnty = lir.FunctionType(
                        lir.VoidType(),
                        [
                            lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(64),
                        ],
                    )
                    fn = cgutils.get_or_insert_function(
                        c.builder.module, fnty, name="unbox_bool_array_obj"
                    )
                    c.builder.call(fn, [obj, data_ptr, bitmap_ptr, n])

    return NativeValue(bool_arr._getvalue())


@box(BooleanArrayType)
def box_bool_arr(typ, val, c):
    """Box bool array into pd.arrays.BooleanArray object. Null bitmap is converted to
    mask array.
    """
    # TODO: refactor with integer array
    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, "C"), bool_arr.data, c.env_manager
    )
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, bool_arr.null_bitmap
    ).data

    # allocate mask array
    n_obj = c.pyapi.call_method(data, "__len__", ())
    n = c.pyapi.long_as_longlong(n_obj)
    mod_name = c.context.insert_const_string(c.builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    bool_dtype = c.pyapi.object_getattr_string(np_class_obj, "bool_")
    mask_arr = c.pyapi.call_method(np_class_obj, "empty", (n_obj, bool_dtype))
    mask_arr_ctypes = c.pyapi.object_getattr_string(mask_arr, "ctypes")
    mask_arr_data = c.pyapi.object_getattr_string(mask_arr_ctypes, "data")
    mask_arr_ptr = c.builder.inttoptr(
        c.pyapi.long_as_longlong(mask_arr_data), lir.IntType(8).as_pointer()
    )

    # fill mask array
    with cgutils.for_range(c.builder, n) as loop:
        # (bits[i >> 3] >> (i & 0x07)) & 1
        i = loop.index
        byte_ind = c.builder.lshr(i, lir.Constant(lir.IntType(64), 3))
        byte = c.builder.load(cgutils.gep(c.builder, bitmap_arr_data, byte_ind))
        mask = c.builder.trunc(
            c.builder.and_(i, lir.Constant(lir.IntType(64), 7)), lir.IntType(8)
        )
        val = c.builder.and_(
            c.builder.lshr(byte, mask), lir.Constant(lir.IntType(8), 1)
        )
        # flip value since bitmap uses opposite convention
        val = c.builder.xor(val, lir.Constant(lir.IntType(8), 1))
        ptr = cgutils.gep(c.builder, mask_arr_ptr, i)
        c.builder.store(val, ptr)

    # clean up bitmap after mask array is created
    c.context.nrt.decref(
        c.builder, types.Array(types.uint8, 1, "C"), bool_arr.null_bitmap
    )

    # create BooleanArray
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arr_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, "arrays")
    res = c.pyapi.call_method(arr_mod_obj, "BooleanArray", (data, mask_arr))

    # clean up references (TODO: check for potential refcount issues)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(n_obj)
    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(bool_dtype)
    c.pyapi.decref(mask_arr_ctypes)
    c.pyapi.decref(mask_arr_data)
    c.pyapi.decref(arr_mod_obj)
    c.pyapi.decref(data)
    c.pyapi.decref(mask_arr)
    return res


@lower_constant(BooleanArrayType)
def lower_constant_bool_arr(context, builder, typ, pyval):

    n = len(pyval)
    data_arr = np.empty(n, np.bool_)
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)
    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            data_arr[i] = s

    data_const_arr = context.get_constant_generic(builder, data_type, data_arr)

    nulls_const_arr = context.get_constant_generic(builder, nulls_type, nulls_arr)

    bool_array = context.make_helper(builder, typ)
    bool_array.data = data_const_arr
    bool_array.null_bitmap = nulls_const_arr
    return bool_array._getvalue()


def lower_init_bool_array(context, builder, signature, args):
    data_val, bitmap_val = args
    # create bool_arr struct and store values
    bool_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
    bool_arr.data = data_val
    bool_arr.null_bitmap = bitmap_val

    # increase refcount of stored values
    context.nrt.incref(builder, signature.args[0], data_val)
    context.nrt.incref(builder, signature.args[1], bitmap_val)

    return bool_arr._getvalue()


@intrinsic
def init_bool_array(typingctx, data, null_bitmap=None):
    """Create a BooleanArray with provided data and null bitmap values."""
    assert data == types.Array(types.bool_, 1, "C")
    assert null_bitmap == types.Array(types.uint8, 1, "C")
    sig = boolean_array(data, null_bitmap)
    return sig, lower_init_bool_array


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_data(A):
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_bitmap(A):
    return lambda A: A._null_bitmap


# array analysis extension
def get_bool_arr_data_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_get_bool_arr_data = (
    get_bool_arr_data_equiv
)


def init_bool_array_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_init_bool_array = (
    init_bool_array_equiv
)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_bool_array(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 2
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_bool_array", "bodo.libs.bool_arr_ext")
] = alias_ext_init_bool_array
numba.core.ir_utils.alias_func_extensions[
    ("get_bool_arr_data", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_bool_arr_bitmap", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func


# high-level allocation function for boolean arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_bool_array(n):  # pragma: no cover
    data_arr = np.empty(n, dtype=np.bool_)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_bool_array(data_arr, nulls)


def alloc_bool_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_bool_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 1 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_alloc_bool_array = (
    alloc_bool_array_equiv
)


@overload(operator.getitem, no_unliteral=True)
def bool_arr_getitem(A, ind):
    if A != boolean_array:
        return

    # TODO: refactor with int arr since almost same code

    if isinstance(types.unliteral(ind), types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        return lambda A, ind: A._data[ind]

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_bool_index(A, ind)
            return init_bool_array(new_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_int_index(A, ind)
            return init_bool_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_slice_index(A, ind)
            return init_bool_array(new_data, new_mask)

        return impl_slice

    # This should be the only BooleanArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for BooleanArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def bool_arr_setitem(A, idx, val):
    if A != boolean_array:
        return

    # TODO: refactor with int arr since almost same code

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for BooleanArray with indexing type {idx} received an incorrect 'value' type {val}."

    # scalar case
    if isinstance(idx, types.Integer):

        if types.unliteral(val) == types.bool_:

            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = val
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            return impl_scalar
        else:
            raise BodoError(typ_err_msg)

    if not (
        (is_iterable_type(val) and val.dtype == types.bool_)
        or types.unliteral(val) == types.bool_
    ):
        raise BodoError(typ_err_msg)

    # array of int indices
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):

        def impl_arr_ind_mask(A, idx, val):  # pragma: no cover
            array_setitem_int_index(A, idx, val)

        return impl_arr_ind_mask

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:

        def impl_bool_ind_mask(A, idx, val):  # pragma: no cover
            array_setitem_bool_index(A, idx, val)

        return impl_bool_ind_mask

    # slice case
    if isinstance(idx, types.SliceType):

        def impl_slice_mask(A, idx, val):  # pragma: no cover
            array_setitem_slice_index(A, idx, val)

        return impl_slice_mask

    # This should be the only BooleanArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for BooleanArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_bool_arr_len(A):
    if A == boolean_array:
        return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(BooleanArrayType, "shape")
def overload_bool_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(BooleanArrayType, "dtype")
def overload_bool_arr_dtype(A):
    return lambda A: pd.BooleanDtype()  # pragma: no cover


@overload_attribute(BooleanArrayType, "ndim")
def overload_bool_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(BooleanArrayType, "nbytes")
def bool_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


@overload_method(BooleanArrayType, "copy", no_unliteral=True)
def overload_bool_arr_copy(A):
    return lambda A: bodo.libs.bool_arr_ext.init_bool_array(
        bodo.libs.bool_arr_ext.get_bool_arr_data(A).copy(),
        bodo.libs.bool_arr_ext.get_bool_arr_bitmap(A).copy(),
    )  # pragma: no cover


@overload_method(BooleanArrayType, "sum", no_unliteral=True, inline="always")
def overload_bool_sum(A):
    """
    Support for .sum() method for BooleanArrays. This is not yet supported
    in Pandas, but can be used to resolve issues where .sum() may be called on
    boolean array that would otherwise be a numpy array.

    We don't accept any arguments at this time as the common case is just A.sum()
    """

    def impl(A):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0
            if not bodo.libs.array_kernels.isna(A, i):
                val = A[i]
            s += val
        return s

    return impl


@overload_method(BooleanArrayType, "astype", no_unliteral=True)
def overload_bool_arr_astype(A, dtype, copy=True):
    # same dtype case
    if dtype == types.bool_:
        # copy=False
        if is_overload_false(copy):
            return lambda A, dtype, copy=True: A
        # copy=True
        elif is_overload_true(copy):
            return lambda A, dtype, copy=True: A.copy()
        # copy value is dynamic
        else:

            def impl(A, dtype, copy=True):  # pragma: no cover
                if copy:
                    return A.copy()
                else:
                    return A

            return impl

    # numpy dtypes
    nb_dtype = parse_dtype(dtype)
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):

        def impl_float(A, dtype, copy=True):  # pragma: no cover
            data = bodo.libs.bool_arr_ext.get_bool_arr_data(A)
            n = len(data)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                B[i] = data[i]
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = np.nan
            return B

        return impl_float

    # TODO: raise error like Pandas when NAs are assigned to integers
    return lambda A, dtype, copy=True: bodo.libs.bool_arr_ext.get_bool_arr_data(
        A
    ).astype(nb_dtype)


@overload(str, no_unliteral=True)
def overload_str_bool(val):
    if val == types.bool_:

        def impl(val):  # pragma: no cover
            if val:
                return "True"
            return "False"

        return impl


# XXX: register all operators just in case they are supported on bool
# TODO: apply null masks if needed
############################### numpy ufuncs #################################


ufunc_aliases = {
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
}


def create_op_overload(op, n_inputs):
    op_name = op.__name__
    op_name = ufunc_aliases.get(op_name, op_name)

    if n_inputs == 1:

        def overload_bool_arr_op_nin_1(A):
            if isinstance(A, BooleanArrayType):
                return bodo.libs.int_arr_ext.get_nullable_array_unary_impl(op, A)

        return overload_bool_arr_op_nin_1
    elif n_inputs == 2:

        def overload_bool_arr_op_nin_2(lhs, rhs):
            # if any input is BooleanArray
            if lhs == boolean_array or rhs == boolean_array:
                return bodo.libs.int_arr_ext.get_nullable_array_binary_impl(
                    op, lhs, rhs
                )

        return overload_bool_arr_op_nin_2
    else:  # pragma: no cover
        raise RuntimeError(
            "Don't know how to register ufuncs from ufunc_db with arity > 2"
        )


def _install_np_ufuncs():
    import numba.np.ufunc_db

    for ufunc in numba.np.ufunc_db.get_ufuncs():
        overload_impl = create_op_overload(ufunc, ufunc.nin)
        overload(ufunc, no_unliteral=True)(overload_impl)


_install_np_ufuncs()


####################### binary operators ###############################

skips = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    # operator.or_ and operator.and_ are
    # handled manually because the null
    # behavior differs from other kernels
    operator.or_,
    operator.and_,
]


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys():
        if op in skips:
            continue
        overload_impl = create_op_overload(op, 2)
        overload(op, no_unliteral=True)(overload_impl)


_install_binary_ops()


####################### binary inplace operators #############################


def _install_inplace_binary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys():
        overload_impl = create_op_overload(op, 2)
        overload(op, no_unliteral=True)(overload_impl)


_install_inplace_binary_ops()


########################## unary operators ###############################


def _install_unary_ops():
    # install unary operators: ~, -, +
    for op in (operator.neg, operator.invert, operator.pos):
        overload_impl = create_op_overload(op, 1)
        overload(op, no_unliteral=True)(overload_impl)


_install_unary_ops()


@overload_method(BooleanArrayType, "unique", no_unliteral=True)
def overload_unique(A):
    def impl_bool_arr(A):  # pragma: no cover
        # preserve order
        data = []
        mask = []
        na_found = False  # write NA only once
        true_found = False
        false_found = False
        for i in range(len(A)):
            if bodo.libs.array_kernels.isna(A, i):
                if not na_found:
                    data.append(False)
                    mask.append(False)
                    na_found = True
                continue
            val = A[i]
            if val and not true_found:
                data.append(True)
                mask.append(True)
                true_found = True
            if not val and not false_found:
                data.append(False)
                mask.append(True)
                false_found = True
            if na_found and true_found and false_found:
                break

        new_data = np.array(data)
        n = len(new_data)
        n_bytes = 1
        new_mask = np.empty(n_bytes, np.uint8)
        for j in range(n):
            bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, j, mask[j])
        return init_bool_array(new_data, new_mask)

    return impl_bool_arr


@overload(operator.getitem, no_unliteral=True)
def bool_arr_ind_getitem(A, ind):
    # getitem for array indexed by BooleanArray
    # TODO: other array types for A?
    if ind == boolean_array and (
        isinstance(A, (types.Array, bodo.libs.int_arr_ext.IntegerArrayType))
        or isinstance(A, bodo.libs.struct_arr_ext.StructArrayType)
        or isinstance(A, bodo.libs.array_item_arr_ext.ArrayItemArrayType)
        or isinstance(A, bodo.libs.map_arr_ext.MapArrayType)
        or A
        in (
            string_array_type,
            bodo.hiframes.split_impl.string_array_split_view_type,
            boolean_array,
        )
    ):
        # XXX assuming data value for NAs is False
        return lambda A, ind: A[ind._data]


@lower_cast(types.Array(types.bool_, 1, "C"), boolean_array)
def cast_np_bool_arr_to_bool_arr(context, builder, fromty, toty, val):
    func = lambda A: bodo.libs.bool_arr_ext.init_bool_array(
        A, np.full((len(A) + 7) >> 3, 255, np.uint8)
    )
    res = context.compile_internal(builder, func, toty(fromty), [val])
    return impl_ret_borrowed(context, builder, toty, res)


@overload(operator.setitem, no_unliteral=True)
def overload_np_array_setitem_bool_arr(A, idx, val):
    """Support setitem of Arrays with boolean_array"""
    if isinstance(A, types.Array) and idx == boolean_array:

        def impl(A, idx, val):  # pragma: no cover
            # TODO(ehsan): consider NAs in idx?
            A[idx._data] = val

        return impl


def create_nullable_logical_op_overload(op):
    is_or = op == operator.or_

    def bool_array_impl(val1, val2):
        """
        Support for operator.or_ and operator.and_
        for nullable boolean arrays. This overload
        only supports two arrays and
        1 array with 1 scalar.
        """
        # 1 input must be a nullable boolean array and the other either a nullable boolean
        # array, a numpy boolean array, or a bool
        is_valid = (
            (val1 == bodo.boolean_array or val2 == bodo.boolean_array)
            and (
                (
                    bodo.utils.utils.is_array_typ(val1, False)
                    and val1.dtype == types.bool_
                )
                or val1 == types.bool_
            )
            and (
                (
                    bodo.utils.utils.is_array_typ(val2, False)
                    and val2.dtype == types.bool_
                )
                or val2 == types.bool_
            )
        )
        if not is_valid:
            return

        # All implementations are almost identical, differing only
        # on the logical check and array vs scalar code. To simplfy
        # the code being generate we allocate these output variables
        # once at the start based on if the inputs are arrays and
        # if we are doing AND or OR.
        is_val1_arr = bodo.utils.utils.is_array_typ(val1, False)
        is_val2_arr = bodo.utils.utils.is_array_typ(val2, False)
        truth_prefix = "not " if not is_or else ""
        len_arr = "val1" if is_val1_arr else "val2"
        val1_elem = "val1[i]" if is_val1_arr else "val1"
        val2_elem = "val2[i]" if is_val2_arr else "val2"

        func_text = "def impl(val1, val2):\n"
        func_text += "  numba.parfors.parfor.init_prange()\n"
        func_text += f"  n = len({len_arr})\n"
        func_text += (
            "  out_arr = bodo.utils.utils.alloc_type(n, bodo.boolean_array, (-1,))\n"
        )
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        if is_val1_arr and is_val2_arr:
            func_text += "    if bodo.libs.array_kernels.isna(val1, i) and bodo.libs.array_kernels.isna(val2, i):\n"
            func_text += "      bodo.libs.array_kernels.setna(out_arr, i)\n"
            func_text += "      continue\n"
        if is_val1_arr:
            func_text += "    if bodo.libs.array_kernels.isna(val1, i):\n"
            func_text += f"      if {truth_prefix}{val2_elem}:\n"
            func_text += f"       out_arr[i] = {val2_elem}\n"
            func_text += "      else:\n"
            func_text += "        bodo.libs.array_kernels.setna(out_arr, i)\n"
            func_text += "      continue\n"
        if is_val2_arr:
            func_text += "    if bodo.libs.array_kernels.isna(val2, i):\n"
            func_text += f"      if {truth_prefix}{val1_elem}:\n"
            func_text += f"       out_arr[i] = {val1_elem}\n"
            func_text += "      else:\n"
            func_text += "        bodo.libs.array_kernels.setna(out_arr, i)\n"
            func_text += "      continue\n"
        func_text += f"    out_arr[i] = operator.{'or_' if is_or else 'and_'}({val1_elem}, {val2_elem})\n"
        func_text += "  return out_arr\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "numba": numba, "operator": operator}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    return bool_array_impl


def _install_nullable_logical_ops():
    # install unary operators: &, |
    for op in (operator.and_, operator.or_):
        overload_impl = create_nullable_logical_op_overload(op)
        overload(op)(overload_impl)


_install_nullable_logical_ops()
