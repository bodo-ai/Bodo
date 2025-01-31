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
from numba.core.typing.templates import AbstractTemplate, infer_global
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
    register_jitable,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import hstr_ext
from bodo.utils.typing import is_list_like_index_type

ll.add_symbol("bool_arr_to_bitmap", hstr_ext.bool_arr_to_bitmap)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    is_iterable_type,
    is_overload_false,
    is_overload_true,
    parse_dtype,
    raise_bodo_error,
)


class BooleanArrayType(types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="BooleanArrayType()")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.bool_

    def copy(self):
        return BooleanArrayType()

    def unify(self, typingctx, other):
        """Allow casting Numpy bool arrays to nullable bool arrays"""
        if isinstance(other, types.Array) and other.ndim == 1:
            # If dtype matches or other.dtype is undefined (inferred)
            # Similar to Numba array unify:
            # https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/core/types/npytypes.py#L491
            if other.dtype == types.bool_ or not other.dtype.is_precise():
                return self


boolean_array_type = BooleanArrayType()


@typeof_impl.register(pd.arrays.BooleanArray)
def typeof_boolean_array(val, c):
    return boolean_array_type


data_type = types.Array(types.uint8, 1, "C")
nulls_type = types.Array(types.uint8, 1, "C")


# store data and nulls as regular numpy arrays without payload machinery
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(BooleanArrayType)
class BooleanArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", data_type),
            ("null_bitmap", nulls_type),
            # We need to store the actual length because the length
            # of the data array doesn't match the actual length.
            ("length", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BooleanArrayType, "data", "_data")
make_attribute_wrapper(BooleanArrayType, "null_bitmap", "_null_bitmap")
make_attribute_wrapper(BooleanArrayType, "length", "_length")


# dtype object for pd.BooleanDtype()
class BooleanDtype(types.Number):
    """
    Type class associated with pandas Boolean dtype pd.BooleanDtype()
    """

    def __init__(self):
        self.dtype = types.bool_
        super().__init__("BooleanDtype")


boolean_dtype = BooleanDtype()


register_model(BooleanDtype)(models.OpaqueModel)


@box(BooleanDtype)
def box_boolean_dtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)
    res = c.pyapi.call_method(pd_class_obj, "BooleanDtype", ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(BooleanDtype)
def unbox_boolean_dtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


typeof_impl.register(pd.BooleanDtype)(lambda a, b: boolean_dtype)
type_callable(pd.BooleanDtype)(lambda c: lambda: boolean_dtype)
lower_builtin(pd.BooleanDtype)(lambda c, b, s, a: c.get_dummy_value())


@numba.njit(cache=True)
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
    structure. The array's dtype can be bool or object, depending on the presence of
    NAs.
    """
    return bodo.libs.array.unbox_array_using_arrow(typ, obj, c)


@box(BooleanArrayType)
def box_bool_arr(typ, val, c):
    """Box bool array into Pandas ArrowExtensionArray."""
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@lower_constant(BooleanArrayType)
def lower_constant_bool_arr(context, builder, typ, pyval):
    n = len(pyval)
    nbytes = (n + 7) >> 3
    data_arr = np.empty(nbytes, np.uint8)
    nulls_arr = np.empty(nbytes, np.uint8)
    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            bodo.libs.int_arr_ext.set_bit_to_arr(data_arr, i, int(s))

    data_const_arr = context.get_constant_generic(builder, data_type, data_arr)

    nulls_const_arr = context.get_constant_generic(builder, nulls_type, nulls_arr)
    len_const = context.get_constant(types.int64, n)

    # create bool arr struct
    return lir.Constant.literal_struct([data_const_arr, nulls_const_arr, len_const])


def lower_init_bool_array(context, builder, signature, args):
    data_val, bitmap_val, length = args
    # create bool_arr struct and store values
    bool_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
    bool_arr.data = data_val
    bool_arr.null_bitmap = bitmap_val
    bool_arr.length = length

    # increase refcount of stored values
    context.nrt.incref(builder, signature.args[0], data_val)
    context.nrt.incref(builder, signature.args[1], bitmap_val)

    return bool_arr._getvalue()


@intrinsic
def init_bool_array(typingctx, data, null_bitmap, length):
    """Create a BooleanArray with provided data and null bitmap values."""
    assert data == types.Array(types.uint8, 1, "C")
    assert null_bitmap == types.Array(types.uint8, 1, "C")
    assert length == types.int64
    sig = boolean_array_type(data, null_bitmap, length)
    return sig, lower_init_bool_array


@register_jitable(inline="always")
def get_boolean_array_bytes_from_length(length):
    """
    Determine the number of bytes used for a boolean
    array of length `length`. Boolean arrays store 1
    bit per value.
    """
    return (length + 7) >> 3


# high-level allocation function for boolean arrays
@numba.njit(cache=True, no_cpython_wrapper=True)
def alloc_bool_array(n):  # pragma: no cover
    num_bytes = get_boolean_array_bytes_from_length(n)
    data_arr = np.empty(num_bytes, dtype=np.uint8)
    nulls = np.empty(num_bytes, dtype=np.uint8)
    return init_bool_array(data_arr, nulls, np.int64(n))


# allocate a boolean array of all false values
@numba.njit(cache=True, no_cpython_wrapper=True)
def alloc_false_bool_array(n):  # pragma: no cover
    num_bytes = get_boolean_array_bytes_from_length(n)
    data_arr = np.zeros(num_bytes, dtype=np.uint8)
    # Initialize null bitmap to all 1s
    nulls = np.full(num_bytes, 255, dtype=np.uint8)
    return init_bool_array(data_arr, nulls, np.int64(n))


# allocate a boolean array of all true values
@numba.njit(cache=True, no_cpython_wrapper=True)
def alloc_true_bool_array(n):  # pragma: no cover
    num_bytes = get_boolean_array_bytes_from_length(n)
    data_arr = np.full(num_bytes, 255, dtype=np.uint8)
    # Initialize null bitmap to all 1s
    nulls = np.full(num_bytes, 255, dtype=np.uint8)
    return init_bool_array(data_arr, nulls, np.int64(n))


def alloc_bool_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_bool_array(), alloc_false_bool_array(),
    and alloc_true_bool_array() passed to Numba's array analysis extension.
    Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 1 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_alloc_bool_array = (
    alloc_bool_array_equiv
)

ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_alloc_false_bool_array = (
    alloc_bool_array_equiv
)

ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_alloc_true_bool_array = (
    alloc_bool_array_equiv
)


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
def bool_arr_getitem(A, ind):
    if A != boolean_array_type:
        return

    if isinstance(types.unliteral(ind), types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        # Note: We fetch a single bit because we store 1 bit per byte.
        return lambda A, ind: bool(
            bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._data, ind)
        )

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            # Calculate the total number of values in the output.
            # This is effectively what the BooleanArrayIndexer does.
            # https://github.com/numba/numba/blob/2829f1f108d3350f456984036f9742c95ce41bfc/numba/np/arrayobj.py#L824
            old_len = len(A)
            new_len = 0
            for i in range(old_len):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    new_len += 1

            # Now that we know the length, we can allocate the output.
            output_arr = alloc_bool_array(new_len)
            # Note: This is very similar to setitem but doesn't match
            # the supported pattern.
            dest_ind = 0
            for i in range(old_len):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    # set the data value
                    data_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._data, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        output_arr._data, dest_ind, data_bit
                    )
                    # Set the null bit
                    null_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        A._null_bitmap, i
                    )
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        output_arr._null_bitmap, dest_ind, null_bit
                    )
                    # Update the destination index
                    dest_ind += 1

            return output_arr

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            n = len(ind)
            output_arr = alloc_bool_array(n)
            # Note: This is very similar to setitem but doesn't match
            # the supported pattern.
            for i in range(n):
                # set the data value
                data_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._data, ind[i])
                bodo.libs.int_arr_ext.set_bit_to_arr(output_arr._data, i, data_bit)
                # Set the null bit
                null_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                    A._null_bitmap, ind[i]
                )
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    output_arr._null_bitmap, i, null_bit
                )

            return output_arr

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            slice_ind = numba.cpython.unicode._normalize_slice(ind, len(A))
            out_length = numba.cpython.unicode._slice_span(slice_ind)
            output_arr = alloc_bool_array(out_length)
            # Note: This is very similar to setitem but doesn't match
            # the supported pattern.
            dest_idx = 0
            for i in range(slice_ind.start, slice_ind.stop, slice_ind.step):
                # set the data value
                data_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._data, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    output_arr._data, dest_idx, data_bit
                )
                # Set the null bit
                null_bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._null_bitmap, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    output_arr._null_bitmap, dest_idx, null_bit
                )
                # Update dest_idx
                dest_idx += 1

            return output_arr

        return impl_slice

    # This should be the only BooleanArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for BooleanArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True, jit_options={"cache": True})
def bool_arr_setitem(A, idx, val):
    if A != boolean_array_type:
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
                # Set the bit in the data array to val
                bodo.libs.int_arr_ext.set_bit_to_arr(A._data, idx, val)
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
        if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(
            val
        ):

            def impl_arr_ind_mask(A, idx, val):  # pragma: no cover
                # Note this is a largely inlined implementation of array_setitem_int_index.
                # We don't reuse the code because the "set" semantics are different.
                n = len(idx)
                for i in range(n):
                    # Abstract away the data access behind getitem
                    data_bit = val[i]
                    # set the data value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._data, idx[i], data_bit)
                    # Abstract away the null access behind isna
                    null_bit = not bodo.libs.array_kernels.isna(val, i)
                    # Set the null bit
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        A._null_bitmap, idx[i], null_bit
                    )

            return impl_arr_ind_mask

        elif val == types.bool_:

            def impl_scalar(A, idx, val):  # pragma: no cover
                n = len(idx)
                for i in range(n):
                    # set the data value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._data, idx[i], val)
                    # Set the null bit
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], 1)

            return impl_scalar

        # Safeguard against gaps in Array setitem to avoid failing in compilation.
        raise BodoError(
            f"setitem not supported for BooleanArray with value {val}"
        )  # pragma: no cover

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        # Note this is a largely inlined implementation of array_setitem_bool_index.
        # We don't reuse the code because the "set" semantics are different.

        if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(
            val
        ):

            def impl_bool_ind_mask(A, idx, val):  # pragma: no cover
                n = len(idx)
                val_ind = 0
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        # Abstract away the data access behind getitem
                        data_bit = val[val_ind]
                        # set the data value
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._data, i, data_bit)
                        # Abstract away the null access behind isna
                        null_bit = not bodo.libs.array_kernels.isna(val, val_ind)
                        # Set the null bit
                        bodo.libs.int_arr_ext.set_bit_to_arr(
                            A._null_bitmap, i, null_bit
                        )
                        val_ind += 1

            return impl_bool_ind_mask

        elif val == types.bool_:

            def impl_scalar(A, idx, val):  # pragma: no cover
                n = len(idx)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        # set the data value
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._data, i, val)
                        # Set the null bit
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_scalar

        # Safeguard against gaps in Array setitem to avoid failing in compilation.
        raise BodoError(
            f"setitem not supported for BooleanArray with value {val}"
        )  # pragma: no cover

    # slice case
    if isinstance(idx, types.SliceType):
        # Note we inline array_setitem_slice_index because
        # the semantics for setitem differ.
        if val == types.boolean:

            def impl_scalar(A, idx, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    # set the data value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._data, i, val)
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_scalar
        else:

            def impl_array(A, idx, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
                val_idx = 0
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    data_bit = val[val_idx]
                    # set the data value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._data, i, data_bit)
                    # Abstract away the null access behind isna
                    null_bit = not bodo.libs.array_kernels.isna(val, val_idx)
                    # Set the null bit
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, null_bit)
                    # Update val_idx
                    val_idx += 1

            return impl_array

    # This should be the only BooleanArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for BooleanArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_bool_arr_len(A):
    if A == boolean_array_type:
        return lambda A: A._length  # pragma: no cover


@overload_attribute(BooleanArrayType, "size", jit_options={"cache": True})
def overload_bool_arr_size(A):
    return lambda A: A._length  # pragma: no cover


@overload_attribute(BooleanArrayType, "shape", jit_options={"cache": True})
def overload_bool_arr_shape(A):
    return lambda A: (A._length,)  # pragma: no cover


@overload_attribute(BooleanArrayType, "dtype", jit_options={"cache": True})
def overload_bool_arr_dtype(A):
    return lambda A: pd.BooleanDtype()  # pragma: no cover


@overload_attribute(BooleanArrayType, "ndim", jit_options={"cache": True})
def overload_bool_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(BooleanArrayType, "nbytes", jit_options={"cache": True})
def bool_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


@overload_method(
    BooleanArrayType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_bool_arr_copy(A):
    return lambda A: bodo.libs.bool_arr_ext.init_bool_array(
        A._data.copy(),
        A._null_bitmap.copy(),
        len(A),
    )  # pragma: no cover


@overload_method(
    BooleanArrayType,
    "sum",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_bool_sum(A):
    """
    Support for .sum() method for BooleanArrays. We don't accept any arguments
    at this time as the common case is just A.sum()
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


@overload_method(
    BooleanArrayType,
    "any",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_bool_any(A):
    """
    Support for .any() method for BooleanArrays. We don't accept any arguments
    at this time as the common case is just A.any().
    """

    def impl(A):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        result = False
        for i in numba.parfors.parfor.internal_prange(len(A)):
            if not bodo.libs.array_kernels.isna(A, i) and A[i]:
                result = True
        return result

    return impl


@overload_method(
    BooleanArrayType, "astype", no_unliteral=True, jit_options={"cache": True}
)
def overload_bool_arr_astype(A, dtype, copy=True):
    # If dtype is a string, force it to be a literal
    if dtype == types.unicode_type:
        raise_bodo_error(
            "BooleanArray.astype(): 'dtype' when passed as string must be a constant value"
        )

    # same dtype case
    if dtype in (types.bool_, boolean_dtype):
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
    nb_dtype = parse_dtype(dtype, "BooleanArray.astype")
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):

        def impl_float(A, dtype, copy=True):  # pragma: no cover
            n = len(A)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                B[i] = A[i]
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = np.nan
            return B

        return impl_float

    # TODO: raise error like Pandas when NAs are assigned to integers
    return lambda A, dtype, copy=True: A.to_numpy().astype(nb_dtype)


@overload_method(
    BooleanArrayType, "fillna", no_unliteral=True, jit_options={"cache": True}
)
def overload_bool_fillna(A, value=None, method=None, limit=None):
    def impl(A, value=None, method=None, limit=None):  # pragma: no cover
        n = len(A)
        B = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, i):
                B[i] = value
            else:
                B[i] = A[i]
        return B

    return impl


@overload_method(BooleanArrayType, "all", jit_options={"cache": True})
def overload_bool_arr_all(A, skipna=True):
    unsupported_args = {"skipna": skipna}
    default_args = {"skipna": True}
    check_unsupported_args(
        "BooleanArray.to_numpy",
        unsupported_args,
        default_args,
        package_name="pandas",
        # Note: We don't have docs from array yet.
        module_name="Array",
    )

    def impl(A, skipna=True):  # pragma: no cover
        result = True
        n = len(A)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, i):
                continue
            elif not A[i]:
                result = False
                break
        # This returns the output for a single rank. Distributed
        # pass handles the reduction.
        return result

    return impl


@overload_method(
    BooleanArrayType, "to_numpy", no_unliteral=True, jit_options={"cache": True}
)
def overload_bool_arr_to_numpy(A, dtype=None, copy=False, na_value=None):
    # TODO: support the proper default value for dtype and na_value
    unsupported_args = {"dtype": dtype, "copy": copy, "na_value": na_value}
    default_args = {"dtype": None, "copy": False, "na_value": None}
    check_unsupported_args(
        "BooleanArray.to_numpy",
        unsupported_args,
        default_args,
        package_name="pandas",
        # Note: We don't have docs from array yet.
        module_name="Array",
    )

    def impl(A, dtype=None, copy=False, na_value=None):  # pragma: no cover
        n = len(A)
        out_array = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            # TODO: handle NA values via na_value
            # For now we default to False because the common use case
            # is to use a boolean array as a filter (where NA is falsy), and we want to match
            # those semantics.
            out_array[i] = not bodo.libs.array_kernels.isna(A, i) and A[i]
        return out_array

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
            if lhs == boolean_array_type or rhs == boolean_array_type:
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


@overload_method(
    BooleanArrayType, "unique", no_unliteral=True, jit_options={"cache": True}
)
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

        n = len(data)
        output_array = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for j in range(n):
            if mask[j]:
                output_array[j] = data[j]
            else:
                bodo.libs.array_kernels.setna(output_array, j)
        return output_array

    return impl_bool_arr


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
def bool_arr_ind_getitem(A, ind):
    # getitem for Numpy arrays indexed by BooleanArray
    if ind == boolean_array_type and isinstance(A, types.Array):
        _dtype = A.dtype

        def impl(A, ind):  # pragma: no cover
            old_len = len(A)
            new_len = 0
            for i in range(old_len):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    new_len += 1

            # Now that we know the length, we can allocate the output.
            output_arr = np.empty(new_len, _dtype)
            # Note: This is very similar to setitem but doesn't match
            # the supported pattern.
            dest_ind = 0
            for i in range(old_len):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    # set the data value
                    output_arr[dest_ind] = A[i]
                    # Update the destination index
                    dest_ind += 1

            return output_arr

        return impl


@lower_cast(types.Array(types.bool_, 1, "C"), boolean_array_type)
def cast_np_bool_arr_to_bool_arr(context, builder, fromty, toty, val):
    def func(A):  # pragma: no cover
        n = len(A)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in range(n):
            out_arr[i] = A[i]
        return out_arr

    res = context.compile_internal(builder, func, toty(fromty), [val])
    return impl_ret_borrowed(context, builder, toty, res)


@overload(operator.setitem, no_unliteral=True, jit_options={"cache": True})
def overload_np_array_setitem_bool_arr(A, idx, val):
    """Support setitem of Arrays with boolean_array_type"""
    if isinstance(A, types.Array) and idx == boolean_array_type:
        if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(
            val
        ):

            def impl(A, idx, val):  # pragma: no cover
                n = len(idx)
                val_ind = 0
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        A[i] = val[val_ind]
                        val_ind += 1

            return impl

        else:

            def impl(A, idx, val):  # pragma: no cover
                n = len(idx)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        A[i] = val

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
        if not is_valid_boolean_array_logical_op(val1, val2):
            return

        # To simplify the code being generate we allocate these output
        # variables once at the start based on if the inputs are arrays.
        is_val1_arr = bodo.utils.utils.is_array_typ(val1, False)
        is_val2_arr = bodo.utils.utils.is_array_typ(val2, False)
        len_arr = "val1" if is_val1_arr else "val2"

        func_text = "def impl(val1, val2):\n"
        func_text += f"  n = len({len_arr})\n"
        func_text += "  out_arr = bodo.utils.utils.alloc_type(n, bodo.boolean_array_type, (-1,))\n"
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        if is_val1_arr:
            null1 = "bodo.libs.array_kernels.isna(val1, i)\n"
            inner_val1 = "val1[i]"
        else:
            null1 = "False\n"
            inner_val1 = "val1"
        if is_val2_arr:
            null2 = "bodo.libs.array_kernels.isna(val2, i)\n"
            inner_val2 = "val2[i]"
        else:
            null2 = "False\n"
            inner_val2 = "val2"
        if is_or:
            func_text += f"    result, isna_val = compute_or_body({null1}, {null2}, {inner_val1}, {inner_val2})\n"
        else:
            func_text += f"    result, isna_val = compute_and_body({null1}, {null2}, {inner_val1}, {inner_val2})\n"
        # We need to place the setna in the first block for setitem/getitem elimination to work properly
        # in the parfor handling in aggregate.py. See test_groupby.py::test_groupby_agg_nullable_or
        # https://github.com/numba/numba/blob/bce065548dd3cb0a3540dde73673c378ad8d37fc/numba/parfors/parfor.py#L4110
        func_text += "    out_arr[i] = result\n"
        func_text += "    if isna_val:\n"
        func_text += "      bodo.libs.array_kernels.setna(out_arr, i)\n"
        func_text += "      continue\n"
        func_text += "  return out_arr\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "numba": numba,
                "compute_and_body": compute_and_body,
                "compute_or_body": compute_or_body,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return bool_array_impl


def compute_or_body(null1, null2, val1, val2):  # pragma: no cover
    pass


@overload(compute_or_body, jit_options={"cache": True})
def overload_compute_or_body(null1, null2, val1, val2):
    """
    Separate function to compute the body of an OR.
    This is used to reduce the amount of IR generated.

    This returns a tuple of values (RESULT, ISNA)
    matching the result if the result should be null.
    """
    # Null sematics have the following behavior:
    # NULL | NULL -> NULL
    # NULL | True -> True
    # NULL | False -> NULL

    def impl(null1, null2, val1, val2):  # pragma: no cover
        if null1 and null2:
            return (False, True)
        elif null1:
            return (val2, val2 == False)
        elif null2:
            return (val1, val1 == False)
        else:
            return (val1 | val2, False)

    return impl


def compute_and_body(null1, null2, val1, val2):  # pragma: no cover
    pass


@overload(compute_and_body, jit_options={"cache": True})
def overload_compute_and_body(null1, null2, val1, val2):
    """
    Separate function to compute the body of an AND.
    This is used to reduce the amount of IR generated.

    This returns a tuple of values (RESULT, ISNA)
    matching the result if the result should be null.
    """
    # Null sematics have the following behavior:
    # NULL & NULL -> NULL
    # NULL & True -> NULL
    # NULL & False -> False

    def impl(null1, null2, val1, val2):  # pragma: no cover
        if null1 and null2:
            return (False, True)
        elif null1:
            return (val2, val2 == True)
        elif null2:
            return (val1, val1 == True)
        else:
            return (val1 & val2, False)

    return impl


def create_boolean_array_logical_lower_impl(op):
    """
    Returns a lowering implementation for the specified operand (Or/And),
    To be used with lower_builtin
    """

    def logical_lower_impl(context, builder, sig, args):
        impl = create_nullable_logical_op_overload(op)(*sig.args)
        return context.compile_internal(builder, impl, sig, args)

    return logical_lower_impl


class BooleanArrayLogicalOperatorTemplate(AbstractTemplate):
    """
    Operator template used for doing typing for nullable logical operators (And/Or)
    between boolean arrays.
    """

    def generic(self, args, kws):
        assert len(args) == 2
        # No kws supported, as builtin operators do not accept them
        assert not kws

        if not is_valid_boolean_array_logical_op(args[0], args[1]):
            return

        # Return type is always boolean array
        ret = boolean_array_type
        # Return the signature
        return ret(*args)


def is_valid_boolean_array_logical_op(typ1, typ2):
    """Helper function that determines if we a valid logical and/or operation
    on a boolean array type"""

    is_valid = (
        (typ1 == bodo.boolean_array_type or typ2 == bodo.boolean_array_type)
        and (
            (bodo.utils.utils.is_array_typ(typ1, False) and typ1.dtype == types.bool_)
            or typ1 == types.bool_
        )
        and (
            (bodo.utils.utils.is_array_typ(typ2, False) and typ2.dtype == types.bool_)
            or typ2 == types.bool_
        )
    )
    return is_valid


def _install_nullable_logical_lowering():
    # install unary operators: &, |
    for op in (operator.and_, operator.or_):
        lower_impl = create_boolean_array_logical_lower_impl(op)
        infer_global(op)(BooleanArrayLogicalOperatorTemplate)
        for typ1, typ2 in [
            (boolean_array_type, boolean_array_type),
            (boolean_array_type, types.bool_),
            (boolean_array_type, types.Array(types.bool_, 1, "C")),
        ]:
            lower_builtin(op, typ1, typ2)(lower_impl)

            if typ1 != typ2:
                lower_builtin(op, typ2, typ1)(lower_impl)


_install_nullable_logical_lowering()
