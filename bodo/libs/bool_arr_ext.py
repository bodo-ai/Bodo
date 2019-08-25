"""Nullable boolean array that stores data in Numpy format (1 byte per value)
but nulls are stored in bit arrays (1 bit per value) similar to Arrow's nulls.
Pandas converts boolean array to object when NAs are introduced.
"""
import operator
import pandas as pd
import numpy as np
import numba
import bodo
from numba import types
from numba import cgutils
from numba.extending import (typeof_impl, type_callable, models,
    register_model, NativeValue, make_attribute_wrapper, lower_builtin, box,
    unbox, lower_getattr, intrinsic, overload_method, overload,
    overload_attribute)
from numba.array_analysis import ArrayAnalysis
from bodo.libs.str_arr_ext import kBitmask
from bodo.utils.typing import is_list_like_index_type

from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext
ll.add_symbol('set_nulls_bool_array', hstr_ext.set_nulls_bool_array)
ll.add_symbol('is_bool_array', hstr_ext.is_bool_array)
ll.add_symbol('unbox_bool_array_obj', hstr_ext.unbox_bool_array_obj)
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false, parse_dtype)


class BooleanArrayType(types.ArrayCompatible):
    def __init__(self):
        super(BooleanArrayType, self).__init__(
            name='BooleanArrayType()')

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, 'C')

    @property
    def dtype(self):
        return types.bool_

    def copy(self):
        return BooleanArrayType()


boolean_array = BooleanArrayType()


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(BooleanArrayType)
class BooleanArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.Array(types.bool_, 1, 'C')),
            ('null_bitmap', types.Array(types.uint8, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BooleanArrayType, 'data', '_data')
make_attribute_wrapper(BooleanArrayType, 'null_bitmap', '_null_bitmap')


@numba.njit
def gen_full_bitmap(n):
    n_bytes = (n + 7) >> 3
    return np.full(n_bytes, 255, np.uint8)


def call_func_in_unbox(func, args, arg_typs, c):
    func_typ = c.context.typing_context.resolve_value_type(func)
    func_sig = func_typ.get_call_type(
        c.context.typing_context, arg_typs, {})
    func_impl = c.context.get_function(func_typ, func_sig)

    # XXX: workaround wrapper must be used due to call convention changes
    fnty = c.context.call_conv.get_function_type(
        func_sig.return_type, func_sig.args)
    mod = c.builder.module
    fn = lir.Function(
        mod, fnty, name=mod.get_unique_name('.func_conv'),
    )
    fn.linkage = 'internal'
    inner_builder = lir.IRBuilder(fn.append_basic_block())
    inner_args = c.context.call_conv.decode_arguments(
        inner_builder, func_sig.args, fn,
    )
    h = func_impl(inner_builder, inner_args)
    c.context.call_conv.return_value(inner_builder, h)

    status, retval = c.context.call_conv.call_function(
        c.builder, fn, func_sig.return_type, func_sig.args, args,
    )
    # TODO: check status?
    return retval


@unbox(BooleanArrayType)
def unbox_bool_array(typ, obj, c):
    """
    Convert a Numpy array object to a native BooleanArray structure.
    The array's dtype can be bool or object, depending on the presense of nans.
    """
    n = c.pyapi.long_as_longlong(c.pyapi.call_method(obj, '__len__', ()))
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="is_bool_array")
    is_bool_dtype = c.builder.call(fn, [obj])
    # cgutils.printf(c.builder, 'is bool %d\n', is_bool_dtype)

    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    cond = c.builder.icmp_unsigned('!=', is_bool_dtype, is_bool_dtype.type(0))
    with c.builder.if_else(cond) as (then, otherwise):
        with then:
            # array is bool
            bool_arr.data = c.pyapi.to_native_value(
                types.Array(types.bool_, 1, 'C'), obj).value
            bool_arr.null_bitmap = call_func_in_unbox(
                gen_full_bitmap, (n,),
                (types.int64,), c)
        with otherwise:
            # array is object
            # allocate data
            bool_arr.data = numba.targets.arrayobj._empty_nd_impl(
                c.context, c.builder, types.Array(types.bool_, 1, 'C'),
                [n])._getvalue()
            # allocate bitmap
            n_bytes = c.builder.udiv(c.builder.add(n,
                lir.Constant(lir.IntType(64), 7)),
                lir.Constant(lir.IntType(64), 8))
            bool_arr.null_bitmap = numba.targets.arrayobj._empty_nd_impl(
                c.context, c.builder, types.Array(types.uint8, 1, 'C'),
                [n_bytes])._getvalue()
            # get array pointers for data and bitmap
            data_ptr = c.context.make_array(types.Array(types.bool_, 1, 'C'))(
                c.context, c.builder, bool_arr.data).data
            bitmap_ptr = c.context.make_array(
                types.Array(types.uint8, 1, 'C'))(
                c.context, c.builder, bool_arr.null_bitmap).data
            fnty = lir.FunctionType(lir.VoidType(),
                [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(), lir.IntType(64)])
            fn = c.builder.module.get_or_insert_function(
                fnty, name="unbox_bool_array_obj")
            c.builder.call(fn, [obj, data_ptr, bitmap_ptr, n])

    return NativeValue(bool_arr._getvalue())


@box(BooleanArrayType)
def box_bool_arr(typ, val, c):
    """Box int array into Numpy boolean array if there are no NAs. Otherwise,
    use object array with NAs converted to np.nan.
    """
    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    bool_arr_obj =  c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, 'C'), bool_arr.data, c.env_manager)
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, 'C'))(
        c.context, c.builder, bool_arr.null_bitmap).data

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
        [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="set_nulls_bool_array")
    res = c.builder.call(fn, [bool_arr_obj, bitmap_arr_data])
    return res


@intrinsic
def init_bool_array(typingctx, data, null_bitmap=None):
    """Create a BooleanArray with provided data and null bitmap values.
    """
    assert data == types.Array(types.bool_, 1, 'C')
    assert null_bitmap == types.Array(types.uint8, 1, 'C')

    def codegen(context, builder, signature, args):
        data_val, bitmap_val = args
        # create bool_arr struct and store values
        bool_arr = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        bool_arr.data = data_val
        bool_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], bitmap_val)

        return bool_arr._getvalue()

    sig = boolean_array(data, null_bitmap)
    return sig, codegen


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_data(A):
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_bitmap(A):
    return lambda A: A._null_bitmap


# array analysis extension
def get_bool_arr_data_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_get_bool_arr_data = \
    get_bool_arr_data_equiv


def init_bool_array_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_init_bool_array = \
    init_bool_array_equiv


@overload(operator.getitem)
def bool_arr_getitem(A, ind):
    if A != boolean_array:
        return

    # TODO: refactor with int arr since almost same code

    if isinstance(ind, types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        return lambda A, ind: A._data[ind]

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        def impl_bool(A, ind):
            ind = bodo.utils.conversion.coerce_to_ndarray(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind]
            n = len(new_data)
            n_bytes = (n + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in numba.parfor.internal_prange(len(ind)):
                if ind[i]:
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        old_mask, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        new_mask, curr_bit, bit)
                    curr_bit += 1
            return init_bool_array(new_data, new_mask)
        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        def impl(A, ind):
            ind_t = bodo.utils.conversion.coerce_to_ndarray(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind_t]
            n = len(new_data)
            n_bytes = (n + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in range(len(ind)):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                    old_mask, ind_t[i])
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    new_mask, curr_bit, bit)
                curr_bit += 1
            return init_bool_array(new_data, new_mask)
        return impl

    # slice case
    if isinstance(ind, types.SliceType):
        def impl_slice(A, ind):
            n = len(A._data)
            old_mask = A._null_bitmap
            new_data = A._data[ind]
            slice_idx = numba.unicode._normalize_slice(ind, n)
            span = numba.unicode._slice_span(slice_idx)
            n_bytes = (span + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                curr_bit += 1
            return init_bool_array(new_data, new_mask)
        return impl_slice


@overload(operator.setitem)
def bool_arr_setitem(A, idx, val):
    if A != boolean_array:
        return

    # TODO: refactor with int arr since almost same code

    # scalar case
    if isinstance(idx, types.Integer):
        assert val == types.bool_
        def impl_scalar(A, idx, val):
            A._data[idx] = val
            bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)
        return impl_scalar

    # array of int indices
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):
        # value is BooleanArray
        if val == boolean_array:
            def impl_arr_ind_mask(A, idx, val):
                n = len(val._data)
                for i in range(n):
                    A._data[idx[i]] = val._data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        val._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        A._null_bitmap, idx[i], bit)
            return impl_arr_ind_mask
        # value is Array/List
        def impl_arr_ind(A, idx, val):
            for i in range(len(val)):
                A._data[idx[i]] = val[i]
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], 1)
        return impl_arr_ind

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        # value is BooleanArray
        if val == boolean_array:
            def impl_bool_ind_mask(A, idx, val):
                n = len(idx)
                val_ind = 0
                for i in range(n):
                    if idx[i]:
                        A._data[i] = val[val_ind]
                        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                            val._null_bitmap, val_ind)
                        bodo.libs.int_arr_ext.set_bit_to_arr(
                            A._null_bitmap, i, bit)
                        val_ind += 1
            return impl_bool_ind_mask
        # value is Array/List
        def impl_bool_ind(A, idx, val):
            n = len(idx)
            val_ind = 0
            for i in range(n):
                if idx[i]:
                    A._data[i] = val[val_ind]
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)
                    val_ind += 1
        return impl_bool_ind

    # slice case
    if isinstance(idx, types.SliceType):
        # value is BooleanArray
        if val == boolean_array:
            def impl_slice_mask(A, idx, val):
                n = len(A._data)
                # using setitem directly instead of copying in loop since
                # Array setitem checks for memory overlap and copies source
                A._data[idx] = val._data
                # XXX: conservative copy of bitmap in case there is overlap
                # TODO: check for overlap and copy only if necessary
                src_bitmap = val._null_bitmap.copy()
                slice_idx = numba.unicode._normalize_slice(idx, n)
                val_ind = 0
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        src_bitmap, val_ind)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        A._null_bitmap, i, bit)
                    val_ind += 1
            return impl_slice_mask
        def impl_slice(A, idx, val):
            n = len(A._data)
            A._data[idx] = val
            slice_idx = numba.unicode._normalize_slice(idx, n)
            val_ind = 0
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)
                val_ind += 1
        return impl_slice


@overload(len)
def overload_bool_arr_len(A):
    if A == boolean_array:
        return lambda A: len(A._data)


@overload_attribute(BooleanArrayType, 'shape')
def overload_bool_arr_shape(A):
    return lambda A: (len(A._data),)


@overload_attribute(BooleanArrayType, 'dtype')
def overload_bool_arr_dtype(A):
    return lambda A: np.bool_


@overload_attribute(BooleanArrayType, 'ndim')
def overload_bool_arr_ndim(A):
    return lambda A: 1


@overload_method(BooleanArrayType, 'copy')
def overload_bool_arr_copy(A):
    return lambda A: bodo.libs.bool_arr_ext.init_bool_array(
        bodo.libs.bool_arr_ext.get_bool_arr_data(A).copy(),
        bodo.libs.bool_arr_ext.get_bool_arr_bitmap(A).copy())


@overload_method(BooleanArrayType, 'astype')
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
            def impl(A, dtype, copy=True):
                if copy:
                    return A.copy()
                else:
                    return A
            return impl

    # numpy dtypes
    nb_dtype = parse_dtype(dtype)
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):
        def impl_float(A, dtype, copy=True):
            data = bodo.libs.bool_arr_ext.get_bool_arr_data(A)
            n = len(data)
            B = np.empty(n, nb_dtype)
            for i in numba.parfor.internal_prange(n):
                B[i] = data[i]
                if bodo.hiframes.api.isna(A, i):
                    B[i] = np.nan
            return B
        return impl_float

    # TODO: raise error like Pandas when NAs are assigned to integers
    return lambda A, dtype, copy=True: \
            bodo.libs.bool_arr_ext.get_bool_arr_data(A).astype(nb_dtype)


@overload(str)
def overload_str_bool(val):
    if val == types.bool_:
        def impl(val):
            if val:
                return 'True'
            return 'False'
        return impl


# XXX: register all operators just in case they are supported on bool
# TODO: apply null masks if needed
############################### numpy ufuncs #################################


def create_op_overload(op, n_inputs):

    if n_inputs == 1:
        def overload_bool_arr_op_nin_1(A):
            if isinstance(A, BooleanArrayType):
                def impl(A):
                    arr = bodo.libs.bool_arr_ext.get_bool_arr_data(A)
                    out_arr = op(arr)
                    return out_arr
                return impl
        return overload_bool_arr_op_nin_1
    elif n_inputs == 2:
        def overload_series_op_nin_2(A1, A2):
            # both are BooleanArray
            if A1 == boolean_array and A2 == boolean_array:
                def impl_both(A1, A2):
                    arr1 = bodo.libs.bool_arr_ext.get_bool_arr_data(A1)
                    arr2 = bodo.libs.bool_arr_ext.get_bool_arr_data(A2)
                    out_arr = op(arr1, arr2)
                    return out_arr
                return impl_both
            # left arg is BooleanArray
            if A1 == boolean_array:
                def impl_left(A1, A2):
                    arr1 = bodo.libs.bool_arr_ext.get_bool_arr_data(A1)
                    out_arr = op(arr1, A2)
                    return out_arr
                return impl_left
            # right arg is BooleanArray
            if A2 == boolean_array:
                def impl_right(A1, A2):
                    arr2 = bodo.libs.bool_arr_ext.get_bool_arr_data(A2)
                    out_arr = op(A1, arr2)
                    return out_arr
                return impl_right
        return overload_series_op_nin_2
    else:
        raise RuntimeError(
            "Don't know how to register ufuncs from ufunc_db with arity > 2")


def _install_np_ufuncs():
    import numba.targets.ufunc_db
    for ufunc in numba.targets.ufunc_db.get_ufuncs():
        overload_impl = create_op_overload(ufunc, ufunc.nin)
        overload(ufunc)(overload_impl)


_install_np_ufuncs()


####################### binary operators ###############################


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in numba.typing.npydecl.NumpyRulesArrayOperator._op_map.keys():
        overload_impl = create_op_overload(op, 2)
        overload(op)(overload_impl)


_install_binary_ops()


####################### binary inplace operators #############################


def _install_inplace_binary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys():
        overload_impl = create_op_overload(op, 2)
        overload(op)(overload_impl)


_install_inplace_binary_ops()


########################## unary operators ###############################


def _install_unary_ops():
    # install unary operators: ~, -, +
    for op in (operator.neg, operator.invert, operator.pos):
        overload_impl = create_op_overload(op, 1)
        overload(op)(overload_impl)


_install_unary_ops()


@overload_method(BooleanArrayType, 'unique')
def overload_unique(A):
    def impl_bool_arr(A):
        # preserve order
        data = []
        mask = []
        na_found = False  # write NA only once
        true_found = False
        false_found = False
        for i in range(len(A)):
            if bodo.hiframes.api.isna(A, i):
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
