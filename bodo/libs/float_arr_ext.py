# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Nullable float array corresponding to Pandas FloatingArray.
However, nulls are stored in bit arrays similar to Arrow's arrays.
"""
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
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

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import array_ext, hstr_ext  # noqa: F401  # isort:skip

ll.add_symbol("is_pd_float_array", array_ext.is_pd_float_array)
ll.add_symbol("float_array_from_sequence", array_ext.float_array_from_sequence)

from bodo.utils.indexing import (
    array_getitem_bool_index,
    array_getitem_int_index,
    array_getitem_slice_index,
    array_setitem_bool_index,
    array_setitem_int_index,
    array_setitem_slice_index,
)
from bodo.utils.typing import (
    BodoArrayIterator,
    BodoError,
    check_unsupported_args,
    is_iterable_type,
    is_list_like_index_type,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    parse_dtype,
    raise_bodo_error,
)


class FloatingArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super(FloatingArrayType, self).__init__(name=f"FloatingArrayType({dtype})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):  # pragma: no cover
        return FloatingArrayType(self.dtype)

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)

    @property
    def get_pandas_scalar_type_instance(self):  # pragma: no cover
        """
        Get the Pandas dtype instance that matches stored
        scalars.
        """
        return pd.Float64Dtype() if self.dtype == types.float64 else pd.Float32Dtype()

    def unify(self, typingctx, other):
        """Allow casting Numpy float arrays to nullable float arrays"""
        if isinstance(other, types.Array) and other.ndim == 1:
            # If dtype matches or other.dtype is undefined (inferred)
            # Similar to Numba array unify:
            # https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/core/types/npytypes.py#L491
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return self


# store data and nulls as regular numpy arrays without payload machinery
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(FloatingArrayType)
class FloatingArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(fe_type.dtype, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(FloatingArrayType, "data", "_data")
make_attribute_wrapper(FloatingArrayType, "null_bitmap", "_null_bitmap")


lower_builtin("getiter", FloatingArrayType)(numba.np.arrayobj.getiter_array)


@typeof_impl.register(pd.arrays.FloatingArray)
def _typeof_pd_float_array(val, c):
    dtype = types.float32 if val.dtype == pd.Float32Dtype() else types.float64
    return FloatingArrayType(dtype)


# dtype object for pd.Float64Dtype() etc.
class FloatDtype(types.Number):
    """
    Type class associated with pandas Float dtypes (Float32Dtype, Float64Dtype).
    """

    def __init__(self, dtype):
        assert isinstance(dtype, types.Float)
        self.dtype = dtype
        name = f"Float{dtype.bitwidth}Dtype()"
        super(FloatDtype, self).__init__(name)


register_model(FloatDtype)(models.OpaqueModel)


@box(FloatDtype)
def box_floatdtype(typ, val, c):  # pragma: no cover
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    res = c.pyapi.call_method(pd_class_obj, str(typ)[:-2], ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(FloatDtype)
def unbox_floatdtype(typ, val, c):  # pragma: no cover
    return NativeValue(c.context.get_dummy_value())


def typeof_pd_float_dtype(val, c):
    dtype = types.float32 if val == pd.Float32Dtype() else types.float64
    return FloatDtype(dtype)


def _register_float_dtype(t):  # pragma: no cover
    typeof_impl.register(t)(typeof_pd_float_dtype)
    float_dtype = typeof_pd_float_dtype(t(), None)
    type_callable(t)(lambda c: lambda: float_dtype)
    lower_builtin(t)(lambda c, b, s, a: c.get_dummy_value())


_register_float_dtype(pd.Float32Dtype)
_register_float_dtype(pd.Float64Dtype)


@unbox(FloatingArrayType)
def unbox_float_array(typ, obj, c):  # pragma: no cover
    """
    Convert a pd.arrays.FloatingArray object to a native FloatingArray structure.
    """
    n_obj = c.pyapi.call_method(obj, "__len__", ())
    n = c.pyapi.long_as_longlong(n_obj)
    c.pyapi.decref(n_obj)

    # TODO: handle or disallow reflection
    float_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    n_bytes = c.builder.udiv(
        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    bitmap_arr_struct = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bytes]
    )

    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="is_pd_float_array"
    )

    is_pd_float = c.builder.call(fn, [obj])
    cond_pd = c.builder.icmp_unsigned("!=", is_pd_float, is_pd_float.type(0))
    with c.builder.if_else(cond_pd) as (pd_then, pd_otherwise):
        with pd_then:
            data_obj = c.pyapi.object_getattr_string(obj, "_data")
            float_arr.data = c.pyapi.to_native_value(
                types.Array(typ.dtype, 1, "C"), data_obj
            ).value

            mask_arr_obj = c.pyapi.object_getattr_string(obj, "_mask")
            mask_arr = c.pyapi.to_native_value(
                types.Array(types.bool_, 1, "C"), mask_arr_obj
            ).value
            c.pyapi.decref(data_obj)
            c.pyapi.decref(mask_arr_obj)

            mask_arr_struct = c.context.make_array(types.Array(types.bool_, 1, "C"))(
                c.context, c.builder, mask_arr
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

            # clean up native mask array after creating bitmap from it
            c.context.nrt.decref(c.builder, types.Array(types.bool_, 1, "C"), mask_arr)
        with pd_otherwise:
            # assuming objects are all float64 (TODO: handle corner cases like np.float32)
            data_arr_struct = bodo.utils.utils._empty_nd_impl(
                c.context, c.builder, types.Array(typ.dtype, 1, "C"), [n]
            )
            fnty = lir.FunctionType(
                lir.IntType(32),
                [
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            unbox_fn = cgutils.get_or_insert_function(
                c.builder.module, fnty, name="float_array_from_sequence"
            )
            c.builder.call(
                unbox_fn,
                [
                    obj,
                    c.builder.bitcast(
                        data_arr_struct.data, lir.IntType(8).as_pointer()
                    ),
                    bitmap_arr_struct.data,
                ],
            )
            float_arr.data = data_arr_struct._getvalue()

    float_arr.null_bitmap = bitmap_arr_struct._getvalue()
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(float_arr._getvalue(), is_error=is_error)


@box(FloatingArrayType)
def box_float_array(typ, val, c):  # pragma: no cover
    """Box float array into pandas FloatingArray object. Null bitmap is converted
    to mask array.
    """
    # box float array's data and bitmap
    float_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, "C"), float_arr.data, c.env_manager
    )
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, float_arr.null_bitmap
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
        c.builder, types.Array(types.uint8, 1, "C"), float_arr.null_bitmap
    )

    # create FloatingArray
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arr_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, "arrays")
    res = c.pyapi.call_method(arr_mod_obj, "FloatingArray", (data, mask_arr))

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


@intrinsic
def init_float_array(typingctx, data, null_bitmap):  # pragma: no cover
    """Create a FloatingArray with provided data and null bitmap values."""
    assert isinstance(data, types.Array)
    assert null_bitmap == types.Array(types.uint8, 1, "C")

    def codegen(context, builder, signature, args):
        data_val, bitmap_val = args
        # create float_arr struct and store values
        float_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        float_arr.data = data_val
        float_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], bitmap_val)

        return float_arr._getvalue()

    ret_typ = FloatingArrayType(data.dtype)
    sig = ret_typ(data, null_bitmap)
    return sig, codegen


@lower_constant(FloatingArrayType)
def lower_constant_float_arr(context, builder, typ, pyval):  # pragma: no cover
    n = len(pyval)
    data_arr = np.empty(n, pyval.dtype.type)
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)
    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            data_arr[i] = s

    data_const_arr = context.get_constant_generic(
        builder, types.Array(typ.dtype, 1, "C"), data_arr
    )

    nulls_const_arr = context.get_constant_generic(
        builder, types.Array(types.uint8, 1, "C"), nulls_arr
    )

    # create float arr struct
    return lir.Constant.literal_struct([data_const_arr, nulls_const_arr])


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_float_arr_data(A):  # pragma: no cover
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_float_arr_bitmap(A):  # pragma: no cover
    return lambda A: A._null_bitmap


# array analysis extension
def get_float_arr_data_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_float_arr_ext_get_float_arr_data = (
    get_float_arr_data_equiv
)


def init_float_array_equiv(self, scope, equiv_set, loc, args, kws):  # pragma: no cover
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_float_arr_ext_init_float_array = (
    init_float_array_equiv
)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):  # pragma: no cover
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_float_array(
    lhs_name, args, alias_map, arg_aliases
):  # pragma: no cover
    assert len(args) == 2
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_float_array", "bodo.libs.float_arr_ext")
] = alias_ext_init_float_array
numba.core.ir_utils.alias_func_extensions[
    ("get_float_arr_data", "bodo.libs.float_arr_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_float_arr_bitmap", "bodo.libs.float_arr_ext")
] = alias_ext_dummy_func


# high-level allocation function for float arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_float_array(n, dtype):  # pragma: no cover
    data_arr = np.empty(n, dtype)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_float_array(data_arr, nulls)


def alloc_float_array_equiv(self, scope, equiv_set, loc, args, kws):  # pragma: no cover
    """Array analysis function for alloc_float_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_float_arr_ext_alloc_float_array = (
    alloc_float_array_equiv
)


@overload(operator.getitem, no_unliteral=True)
def float_arr_getitem(A, ind):  # pragma: no cover
    if not isinstance(A, FloatingArrayType):
        return

    if isinstance(ind, types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        return lambda A, ind: A._data[ind]

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_bool_index(A, ind)
            return init_float_array(new_data, new_mask)

        return impl_bool

    # float arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_int_index(A, ind)
            return init_float_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_slice_index(A, ind)
            return init_float_array(new_data, new_mask)

        return impl_slice

    # This should be the only FloatingArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for FloatingArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def float_arr_setitem(A, idx, val):  # pragma: no cover
    if not isinstance(A, FloatingArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for FloatingArray with indexing type {idx} received an incorrect 'value' type {val}."

    is_scalar = isinstance(
        val, (types.Integer, types.Boolean, types.Float, bodo.Decimal128Type)
    )

    # scalar case
    if isinstance(idx, types.Integer):
        if is_scalar:

            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = val
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            return impl_scalar

        else:
            raise BodoError(typ_err_msg)

    if not (
        (
            is_iterable_type(val)
            and isinstance(val.dtype, (types.Integer, types.Boolean, types.Float))
        )
        or is_scalar
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

    # This should be the only FloatingArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for FloatingArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def numpy_arr_setitem(A, idx, val):
    """Support setitem of Numpy arrays with nullable float arrays"""
    if not (
        isinstance(A, types.Array)
        and isinstance(A.dtype, types.Float)
        and isinstance(val, FloatingArrayType)
    ):
        return

    def impl_np_setitem_float_arr(A, idx, val):  # pragma: no cover
        # NOTE: NAs are lost in this operation if present for SQL so upstream operations
        # should make sure this is safe. For example, BodoSQL may know output is
        # non-nullable but internal operations may use nullable types by default.
        # See test_literals.py::test_array_literals_case"[integer_literals]"

        # Make sure data elements of NA values are NaN to pass the NAs to output
        data = val._data
        bitmap = val._null_bitmap
        for i in range(len(val)):
            if not bodo.libs.int_arr_ext.get_bit_bitmap_arr(bitmap, i):
                data[i] = np.nan

        A[idx] = data

    return impl_np_setitem_float_arr


@overload(len, no_unliteral=True)
def overload_float_arr_len(A):  # pragma: no cover
    if isinstance(A, FloatingArrayType):
        return lambda A: len(A._data)


@overload_attribute(FloatingArrayType, "shape")
def overload_float_arr_shape(A):  # pragma: no cover
    return lambda A: (len(A._data),)


@overload_attribute(FloatingArrayType, "dtype")
def overload_float_arr_dtype(A):  # pragma: no cover
    dtype_class = pd.Float32Dtype if A.dtype == types.float32 else pd.Float64Dtype
    return lambda A: dtype_class()


@overload_attribute(FloatingArrayType, "ndim")
def overload_float_arr_ndim(A):  # pragma: no cover
    return lambda A: 1


@overload_attribute(FloatingArrayType, "size")
def overload_float_size(A):
    return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(FloatingArrayType, "nbytes")
def float_arr_nbytes_overload(A):  # pragma: no cover
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


@overload_method(FloatingArrayType, "copy", no_unliteral=True)
def overload_float_arr_copy(A, dtype=None):  # pragma: no cover
    # TODO: Update dtype to do proper parsing with supported types.
    if not is_overload_none(dtype):
        return lambda A, dtype=None: A.astype(dtype, copy=True)  # pragma: no cover
    else:
        return lambda A, dtype=None: bodo.libs.float_arr_ext.init_float_array(
            bodo.libs.float_arr_ext.get_float_arr_data(A).copy(),
            bodo.libs.float_arr_ext.get_float_arr_bitmap(A).copy(),
        )  # pragma: no cover


@overload_method(FloatingArrayType, "astype", no_unliteral=True)
def overload_float_arr_astype(A, dtype, copy=True):  # pragma: no cover
    # dtype becomes NumberClass if type reference is passed
    # see convert_to_nullable_tup in array_kernels.py
    # see test_series_concat_convert_to_nullable

    # If dtype is a string, force it to be a literal
    if dtype == types.unicode_type:
        raise_bodo_error(
            "FloatingArray.astype(): 'dtype' when passed as string must be a constant value"
        )

    if isinstance(dtype, types.NumberClass):
        dtype = dtype.dtype

    # same dtype case
    if isinstance(dtype, FloatDtype) and A.dtype == dtype.dtype:
        # copy=False
        if is_overload_false(copy):
            return lambda A, dtype, copy=True: A  # pragma: no cover
        # copy=True
        elif is_overload_true(copy):
            return lambda A, dtype, copy=True: A.copy()  # pragma: no cover
        # copy value is dynamic
        else:

            def impl(A, dtype, copy=True):  # pragma: no cover
                if copy:
                    return A.copy()
                else:
                    return A

            return impl

    # other FloatDtype value, needs copy (TODO: copy mask?)
    if isinstance(dtype, FloatDtype):
        np_dtype = dtype.dtype
        return lambda A, dtype, copy=True: bodo.libs.float_arr_ext.init_float_array(
            bodo.libs.float_arr_ext.get_float_arr_data(A).astype(np_dtype),
            bodo.libs.float_arr_ext.get_float_arr_bitmap(A).copy(),
        )

    # Nullable integer type
    if isinstance(dtype, bodo.libs.int_arr_ext.IntDtype):
        np_dtype = dtype.dtype
        return lambda A, dtype, copy=True: bodo.libs.int_arr_ext.init_integer_array(
            bodo.libs.float_arr_ext.get_float_arr_data(A).astype(np_dtype),
            bodo.libs.float_arr_ext.get_float_arr_bitmap(A).copy(),
        )

    # numpy dtypes
    nb_dtype = parse_dtype(dtype, "FloatingArray.astype")
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):

        def impl_float(A, dtype, copy=True):  # pragma: no cover
            data = bodo.libs.float_arr_ext.get_float_arr_data(A)
            n = len(data)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                B[i] = data[i]
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = np.nan
            return B

        return impl_float

    # TODO: raise error like Pandas when NAs are assigned to floats
    return lambda A, dtype, copy=True: bodo.libs.float_arr_ext.get_float_arr_data(
        A
    ).astype(nb_dtype)


@lower_cast(types.Array, FloatingArrayType)
def cast_float_array(context, builder, fromty, toty, val):
    """cast regular float array to nullable float array"""
    f = lambda A: bodo.utils.conversion.coerce_to_array(
        A, use_nullable_array=True
    )  # pragma: no cover
    return context.compile_internal(builder, f, toty(fromty), [val])


@lower_cast(FloatingArrayType, types.Array)
def cast_float_array(context, builder, fromty, toty, val):
    """cast nullable float array to regular float array"""
    dtype = toty.dtype
    f = lambda A: A.astype(dtype)  # pragma: no cover
    return context.compile_internal(builder, f, toty(fromty), [val])


############################### numpy ufuncs #################################


ufunc_aliases = {
    "subtract": "sub",
    "multiply": "mul",
    "floor_divide": "floordiv",
    "true_divide": "truediv",
    "power": "pow",
    "remainder": "mod",
    "divide": "div",
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
}


def create_op_overload(op, n_inputs):  # pragma: no cover
    """creates overloads for operations on Floating arrays"""
    if n_inputs == 1:

        def overload_float_arr_op_nin_1(A):
            if isinstance(A, FloatingArrayType):
                return bodo.libs.int_arr_ext.get_nullable_array_unary_impl(op, A)

        return overload_float_arr_op_nin_1
    elif n_inputs == 2:

        def overload_series_op_nin_2(lhs, rhs):
            if isinstance(lhs, FloatingArrayType) or isinstance(rhs, FloatingArrayType):
                return bodo.libs.int_arr_ext.get_nullable_array_binary_impl(
                    op, lhs, rhs
                )

        return overload_series_op_nin_2
    else:
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
]


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys():
        if op in skips:
            continue
        overload_impl = create_op_overload(op, 2)
        overload(op)(overload_impl)


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


@overload(np.var, inline="always")
def overload_var(A):
    """Implements np.var() for nullable float arrays.
    Unlike Numpy arrays, this currently skips NAs to match SQL behavior since it's
    used in BodoSQL groupby:
    https://github.com/Bodo-inc/Bodo/blob/9384eee70c35eb16fd88e70456b4e6a89a485059/BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/BodoSQLCodeGen/AggCodeGen.java#L48
    """
    if not isinstance(A, FloatingArrayType):
        return

    def impl(A):
        return bodo.libs.array_ops.array_op_var(A, True, 0)

    return impl


@overload(np.std, inline="always")
def overload_std(A):
    """Implements np.std() for nullable float arrays.
    Unlike Numpy arrays, this currently skips NAs to match SQL behavior since it's
    used in BodoSQL groupby:
    https://github.com/Bodo-inc/Bodo/blob/9384eee70c35eb16fd88e70456b4e6a89a485059/BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/BodoSQLCodeGen/AggCodeGen.java#L48
    """
    if not isinstance(A, FloatingArrayType):
        return

    def impl(A):
        return bodo.libs.array_ops.array_op_std(A, True, 0)

    return impl


# inlining in Series pass but avoiding inline="always" since there are Numba-only cases
# that don't need inlining such as repeats.sum() in repeat_kernel()
@overload_method(FloatingArrayType, "sum", no_unliteral=True)
def overload_float_arr_sum(A, skipna=True, min_count=0):  # pragma: no cover
    """A.sum() for nullable float arrays"""
    unsupported_args = dict(skipna=skipna, min_count=min_count)
    arg_defaults = dict(skipna=True, min_count=0)
    check_unsupported_args("FloatingArray.sum", unsupported_args, arg_defaults)

    def impl(A, skipna=True, min_count=0):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = 0.0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0.0
            if not bodo.libs.array_kernels.isna(A, i):
                val = A[i]
            s += val
        return s

    return impl


@overload_method(FloatingArrayType, "unique", no_unliteral=True)
def overload_unique(A):  # pragma: no cover
    dtype = A.dtype

    def impl_float_arr(A):
        # preserve order
        data = []
        mask = []
        na_found = False  # write NA only once
        s = set()
        for i in range(len(A)):
            val = A[i]
            if bodo.libs.array_kernels.isna(A, i):
                if not na_found:
                    data.append(dtype(1))
                    mask.append(False)
                    na_found = True
                continue
            if val not in s:
                s.add(val)
                data.append(val)
                mask.append(True)
        new_data = np.array(data)
        n = len(new_data)
        n_bytes = (n + 7) >> 3
        new_mask = np.empty(n_bytes, np.uint8)
        for j in range(n):
            bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, j, mask[j])
        return init_float_array(new_data, new_mask)

    return impl_float_arr
