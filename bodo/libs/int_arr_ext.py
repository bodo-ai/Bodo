"""Nullable integer array corresponding to Pandas IntegerArray.
However, nulls are stored in bit arrays similar to Arrow's arrays.
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
ll.add_symbol('mask_arr_to_bitmap', hstr_ext.mask_arr_to_bitmap)
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false, parse_dtype)


class IntegerArrayType(types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super(IntegerArrayType, self).__init__(
            name='IntegerArrayType({})'.format(dtype))

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, 'C')

    def copy(self):
        return IntegerArrayType(self.dtype)


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(IntegerArrayType)
class IntegerArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.Array(fe_type.dtype, 1, 'C')),
            ('null_bitmap', types.Array(types.uint8, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntegerArrayType, 'data', '_data')
make_attribute_wrapper(IntegerArrayType, 'null_bitmap', '_null_bitmap')


@typeof_impl.register(pd.arrays.IntegerArray)
def _typeof_pd_int_array(val, c):
    bitwidth = 8 * val.dtype.itemsize
    kind = '' if val.dtype.kind == 'i' else 'u'
    dtype = getattr(types, '{}int{}'.format(kind, bitwidth))
    return IntegerArrayType(dtype)


# dtype object for pd.Int64Dtype() etc.
class IntDtype(types.Type):
    """
    Type class associated with pandas Integer dtypes (e.g. pd.Int64Dtype,
    pd.UInt64Dtype).
    """
    def __init__(self, dtype):
        assert isinstance(dtype, types.Integer)
        self.dtype = dtype
        name = "{}Int{}Dtype()".format(
            '' if dtype.signed else 'U', dtype.bitwidth)
        super(IntDtype, self).__init__(name)


register_model(IntDtype)(models.OpaqueModel)


@box(IntDtype)
def box_intdtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    res = c.pyapi.call_method(pd_class_obj, str(typ)[:-2], ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(IntDtype)
def unbox_intdtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


def typeof_pd_int_dtype(val, c):
    bitwidth = 8 * val.itemsize
    kind = '' if val.kind == 'i' else 'u'
    dtype = getattr(types, '{}int{}'.format(kind, bitwidth))
    return IntDtype(dtype)


def _register_int_dtype(t):
    typeof_impl.register(t)(typeof_pd_int_dtype)
    int_dtype = typeof_pd_int_dtype(t(), None)
    type_callable(t)(lambda c: lambda: int_dtype)
    lower_builtin(t)(lambda c, b, s, a: c.get_dummy_value())


pd_int_dtype_classes = (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype,
    pd.Int64Dtype, pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype,
    pd.UInt64Dtype)


for t in pd_int_dtype_classes:
    _register_int_dtype(t)


@numba.extending.register_jitable
def mask_arr_to_bitmap(mask_arr):
    n = len(mask_arr)
    n_bytes = (n + 7) >> 3
    bit_arr = np.empty(n_bytes, np.uint8)
    for i in range(n):
        b_ind = i // 8
        bit_arr[b_ind] ^= np.uint8(
            -np.uint8(not mask_arr[i]) ^ bit_arr[b_ind]) & kBitmask[i % 8]

    return bit_arr


@unbox(IntegerArrayType)
def unbox_int_array(typ, obj, c):
    """
    Convert a pd.arrays.IntegerArray object to a native IntegerArray structure.
    """
    # TODO: handle or disallow reflection
    int_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data_obj = c.pyapi.object_getattr_string(obj, "_data")
    int_arr.data = c.pyapi.to_native_value(
        types.Array(typ.dtype, 1, 'C'), data_obj).value

    mask_arr_obj = c.pyapi.object_getattr_string(obj, "_mask")
    mask_arr = c.pyapi.to_native_value(
        types.Array(types.bool_, 1, 'C'), mask_arr_obj).value

    # TODO: use this when Numba's #4435 is resolved
    # bt_typ = c.context.typing_context.resolve_value_type(mask_arr_to_bitmap)
    # bt_sig = bt_typ.get_call_type(
    #     c.context.typing_context, (types.Array(types.bool_, 1, 'C'),), {})
    # bt_impl = c.context.get_function(bt_typ, bt_sig)
    # int_arr.null_bitmap = bt_impl(c.builder, (mask_arr,))

    # XXX: workaround wrapper can be used
    # fnty = c.context.call_conv.get_function_type(bt_sig.return_type, bt_sig.args)
    # mod = c.builder.module
    # fn = lir.Function(
    #     mod, fnty, name=mod.get_unique_name('.mask_arr_conv'),
    # )
    # fn.linkage = 'internal'
    # inner_builder = lir.IRBuilder(fn.append_basic_block())
    # [inner_item] = c.context.call_conv.decode_arguments(
    #     inner_builder, bt_sig.args, fn,
    # )
    # h = bt_impl(inner_builder, (inner_item,))
    # c.context.call_conv.return_value(inner_builder, h)

    # status, retval = c.context.call_conv.call_function(
    #     c.builder, fn, bt_sig.return_type, bt_sig.args, [mask_arr],
    # )

    # int_arr.null_bitmap = retval

    n = c.pyapi.long_as_longlong(c.pyapi.call_method(obj, '__len__', ()))
    n_bytes = c.builder.udiv(c.builder.add(n,
            lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 8))
    mask_arr_struct = c.context.make_array(types.Array(types.bool_, 1, 'C'))(
        c.context, c.builder, mask_arr)
    bitmap_arr_struct = numba.targets.arrayobj._empty_nd_impl(
        c.context, c.builder, types.Array(types.bool_, 1, 'C'), [n_bytes])

    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
        lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="mask_arr_to_bitmap")
    c.builder.call(fn, [bitmap_arr_struct.data, mask_arr_struct.data, n])

    int_arr.null_bitmap = bitmap_arr_struct._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(int_arr._getvalue(), is_error=is_error)


@box(IntegerArrayType)
def box_int_arr(typ, val, c):
    """Box int array into pandas IntegerArray object. Null bitmap is converted
    to mask array.
    """
    # box integer array's data and bitmap
    int_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data =  c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, 'C'), int_arr.data, c.env_manager)
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, 'C'))(
        c.context, c.builder, int_arr.null_bitmap).data

    # allocate mask array
    n_obj = c.pyapi.call_method(data, '__len__', ())
    n = c.pyapi.long_as_longlong(n_obj)
    mod_name = c.context.insert_const_string(c.builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    bool_dtype = c.pyapi.object_getattr_string(np_class_obj, 'bool_')
    mask_arr = c.pyapi.call_method(np_class_obj, 'empty', (n_obj, bool_dtype))
    mask_arr_ctypes = c.pyapi.object_getattr_string(mask_arr, 'ctypes')
    mask_arr_data = c.pyapi.object_getattr_string(mask_arr_ctypes, 'data')
    mask_arr_ptr = c.builder.inttoptr(
        c.pyapi.long_as_longlong(mask_arr_data), lir.IntType(8).as_pointer())

    # fill mask array
    with cgutils.for_range(c.builder, n) as loop:
        # (bits[i >> 3] >> (i & 0x07)) & 1
        i = loop.index
        byte_ind = c.builder.lshr(i, lir.Constant(lir.IntType(64), 3))
        byte = c.builder.load(
            cgutils.gep(c.builder, bitmap_arr_data, byte_ind))
        mask = c.builder.trunc(
            c.builder.and_(i, lir.Constant(lir.IntType(64), 7)),
            lir.IntType(8))
        val = c.builder.and_(
            c.builder.lshr(byte, mask), lir.Constant(lir.IntType(8), 1))
        # flip value since bitmap uses opposite convention
        val = c.builder.xor(val, lir.Constant(lir.IntType(8), 1))
        ptr = cgutils.gep(c.builder, mask_arr_ptr, i)
        c.builder.store(val, ptr)

    # create IntegerArray
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arr_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, 'arrays')
    res = c.pyapi.call_method(arr_mod_obj, 'IntegerArray', (data, mask_arr))

    # clean up references (TODO: check for potential refcount issues)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(n_obj)
    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(bool_dtype)
    c.pyapi.decref(mask_arr_ctypes)
    c.pyapi.decref(mask_arr_data)
    c.pyapi.decref(arr_mod_obj)
    return res


@intrinsic
def init_integer_array(typingctx, data, null_bitmap=None):
    """Create a IntegerArray with provided data and null bitmap values.
    """
    assert isinstance(data, types.Array)
    assert null_bitmap == types.Array(types.uint8, 1, 'C')

    def codegen(context, builder, signature, args):
        data_val, bitmap_val = args
        # create int_arr struct and store values
        int_arr = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        int_arr.data = data_val
        int_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], bitmap_val)

        return int_arr._getvalue()

    ret_typ = IntegerArrayType(data.dtype)
    sig = ret_typ(data, null_bitmap)
    return sig, codegen


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_int_arr_data(A):
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_int_arr_bitmap(A):
    return lambda A: A._null_bitmap


# array analysis extension
def get_int_arr_data_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_int_arr_ext_get_int_arr_data = \
    get_int_arr_data_equiv


def init_integer_array_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_int_arr_ext_init_integer_array = \
    init_integer_array_equiv


@numba.extending.register_jitable
def set_bit_to_arr(bits, i, bit_is_set):
    bits[i // 8] ^= np.uint8(-np.uint8(bit_is_set) ^ bits[i // 8]) & kBitmask[i % 8]


@numba.extending.register_jitable
def get_bit_bitmap_arr(bits, i):
    return (bits[i >> 3] >> (i & 0x07)) & 1


@overload(operator.getitem)
def int_arr_getitem(A, ind):
    if not isinstance(A, IntegerArrayType):
        return

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
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        new_mask, curr_bit, bit)
                    curr_bit += 1
            return init_integer_array(new_data, new_mask)
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
            return init_integer_array(new_data, new_mask)
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
            return init_integer_array(new_data, new_mask)
        return impl_slice


@overload(operator.setitem)
def int_arr_setitem(A, idx, val):
    if not isinstance(A, IntegerArrayType):
        return

    # scalar case
    if isinstance(idx, types.Integer):
        assert isinstance(val, types.Integer)
        def impl_scalar(A, idx, val):
            A._data[idx] = val
            bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)
        return impl_scalar

    # array of int indices
    if isinstance(idx, types.Array) and isinstance(idx.dtype, types.Integer):
        # value is IntegerArray
        if isinstance(val, IntegerArrayType):
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
    if idx == types.Array(types.bool_, 1, 'C'):
        # value is IntegerArray
        if isinstance(val, IntegerArrayType):
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
        # value is IntegerArray
        if isinstance(val, IntegerArrayType):
            def impl_slice_mask(A, idx, val):
                n = len(A._data)
                slice_idx = numba.unicode._normalize_slice(idx, n)
                val_ind = 0
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A._data[i] = val._data[val_ind]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        val._null_bitmap, val_ind)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        A._null_bitmap, i, bit)
                    val_ind += 1
            return impl_slice_mask
        def impl_slice(A, idx, val):
            n = len(A._data)
            slice_idx = numba.unicode._normalize_slice(idx, n)
            val_ind = 0
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                A._data[i] = val[val_ind]
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)
                val_ind += 1
        return impl_slice


@overload(len)
def overload_int_arr_len(A):
    if isinstance(A, IntegerArrayType):
        return lambda A: len(A._data)


@overload_attribute(IntegerArrayType, 'shape')
def overload_int_arr_shape(A):
    return lambda A: (len(A._data),)


@overload_attribute(IntegerArrayType, 'dtype')
def overload_int_arr_dtype(A):
    dtype_class = getattr(pd, "{}Int{}Dtype".format(
            '' if A.dtype.signed else 'U', A.dtype.bitwidth))
    return lambda A: dtype_class()


@overload_attribute(IntegerArrayType, 'ndim')
def overload_int_arr_ndim(A):
    return lambda A: 1


@overload_method(IntegerArrayType, 'copy')
def overload_int_arr_copy(A):
    return lambda A: bodo.libs.int_arr_ext.init_integer_array(
        bodo.libs.int_arr_ext.get_int_arr_data(A).copy(),
        bodo.libs.int_arr_ext.get_int_arr_bitmap(A).copy())


@overload_method(IntegerArrayType, 'astype')
def overload_int_arr_astype(A, dtype, copy=True):
    # same dtype case
    if isinstance(dtype, IntDtype) and A.dtype == dtype.dtype:
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

    # other IntDtype value, needs copy (TODO: copy mask?)
    if isinstance(dtype, IntDtype):
        np_dtype = dtype.dtype
        return lambda A, dtype, copy=True: bodo.libs.int_arr_ext.init_integer_array(
            bodo.libs.int_arr_ext.get_int_arr_data(A).astype(np_dtype),
            bodo.libs.int_arr_ext.get_int_arr_bitmap(A).copy())

    # numpy dtypes
    nb_dtype = parse_dtype(dtype)
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):
        def impl_float(A, dtype, copy=True):
            data = bodo.libs.int_arr_ext.get_int_arr_data(A)
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
            bodo.libs.int_arr_ext.get_int_arr_data(A).astype(nb_dtype)


############################### numpy ufuncs #################################


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def apply_null_mask(arr, bitmap, mask_fill, inplace):
    assert isinstance(arr, types.Array)

    # Integer output becomes IntegerArray
    if isinstance(arr.dtype, types.Integer):
        if is_overload_none(inplace):
            return lambda arr, bitmap, mask_fill, inplace: \
                bodo.libs.int_arr_ext.init_integer_array(arr, bitmap.copy())
        else:
            return lambda arr, bitmap, mask_fill, inplace: \
                bodo.libs.int_arr_ext.init_integer_array(arr, bitmap)

    # NAs are applied to Float output
    if isinstance(arr.dtype, types.Float):
        def impl(arr, bitmap, mask_fill, inplace):
            n = len(arr)
            for i in numba.parfor.internal_prange(n):
                if not bodo.libs.int_arr_ext.get_bit_bitmap_arr(bitmap, i):
                    arr[i] = np.nan
            return arr
        return impl

    if arr.dtype == types.bool_:
        def impl_bool(arr, bitmap, mask_fill, inplace):
            n = len(arr)
            for i in numba.parfor.internal_prange(n):
                if not bodo.libs.int_arr_ext.get_bit_bitmap_arr(bitmap, i):
                    arr[i] = mask_fill
            return arr
        return impl_bool
    # TODO: handle other possible types
    return lambda arr, bitmap, mask_fill, inplace: arr


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def merge_bitmaps(B1, B2, n, inplace):
    assert B1 == types.Array(types.uint8, 1, 'C')
    assert B2 == types.Array(types.uint8, 1, 'C')

    if not is_overload_none(inplace):
        def impl_inplace(B1, B2, n, inplace):
            # looping over bits individually to hopefully enable more fusion
            # TODO: evaluate and improve
            for i in numba.parfor.internal_prange(n):
                bit1 = bodo.libs.int_arr_ext.get_bit_bitmap_arr(B1, i)
                bit2 = bodo.libs.int_arr_ext.get_bit_bitmap_arr(B2, i)
                bit = bit1 & bit2
                bodo.libs.int_arr_ext.set_bit_to_arr(B1, i, bit)
            return B1
        return impl_inplace

    def impl(B1, B2, n, inplace):
        numba.parfor.init_prange()
        n_bytes = (n + 7) >> 3
        B = np.empty(n_bytes, np.uint8)
        # looping over bits individually to hopefully enable more fusion
        # TODO: evaluate and improve
        for i in numba.parfor.internal_prange(n):
            bit1 = bodo.libs.int_arr_ext.get_bit_bitmap_arr(B1, i)
            bit2 = bodo.libs.int_arr_ext.get_bit_bitmap_arr(B2, i)
            bit = bit1 & bit2
            bodo.libs.int_arr_ext.set_bit_to_arr(B, i, bit)
        return B
    return impl


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


def create_op_overload(op, n_inputs):
    # see __array_ufunc__() of pd.arrays.IntegerArray
    # XXX: pandas assigns np.nan to NA positions which translates to True
    # for bool output of ufuncs, except the ones that are mapped to comparison
    # operators
    # TODO: use nullable Bool type
    # https://github.com/pandas-dev/pandas/blob/5de4e55d60bf8487a2ce64a440b6d5d92345a4bc/pandas/core/arrays/integer.py#L408
    op_name = op.__name__
    op_name = ufunc_aliases.get(op_name, op_name)
    # comparison operators assign False except 'ne'
    # https://github.com/pandas-dev/pandas/blob/5de4e55d60bf8487a2ce64a440b6d5d92345a4bc/pandas/core/arrays/integer.py#L631
    mask_fill = op_name not in ("eq", "lt", "gt", "le", "ge")
    # TODO: 1 ** np.nan is 1. So we have to unmask those.
    inplace = None
    if op in numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys():
        inplace = True

    if n_inputs == 1:
        def overload_int_arr_op_nin_1(A):
            if isinstance(A, IntegerArrayType):
                def impl(A):
                    arr = bodo.libs.int_arr_ext.get_int_arr_data(A)
                    bitmap = bodo.libs.int_arr_ext.get_int_arr_bitmap(A)
                    out_arr = op(arr)
                    return bodo.libs.int_arr_ext.apply_null_mask(
                                           out_arr, bitmap, mask_fill, inplace)
                return impl
        return overload_int_arr_op_nin_1
    elif n_inputs == 2:
        def overload_series_op_nin_2(A1, A2):
            # both are IntegerArray
            if isinstance(A1, IntegerArrayType) and isinstance(
                                                         A2, IntegerArrayType):
                def impl_both(A1, A2):
                    arr1 = bodo.libs.int_arr_ext.get_int_arr_data(A1)
                    bitmap1 = bodo.libs.int_arr_ext.get_int_arr_bitmap(A1)
                    arr2 = bodo.libs.int_arr_ext.get_int_arr_data(A2)
                    bitmap2 = bodo.libs.int_arr_ext.get_int_arr_bitmap(A2)
                    out_arr = op(arr1, arr2)
                    bitmap = bodo.libs.int_arr_ext.merge_bitmaps(
                                          bitmap1, bitmap2, len(arr1), inplace)
                    return bodo.libs.int_arr_ext.apply_null_mask(
                                           out_arr, bitmap, mask_fill, inplace)
                return impl_both
            # left arg is IntegerArray
            if isinstance(A1, IntegerArrayType):
                def impl_left(A1, A2):
                    arr1 = bodo.libs.int_arr_ext.get_int_arr_data(A1)
                    bitmap = bodo.libs.int_arr_ext.get_int_arr_bitmap(A1)
                    out_arr = op(arr1, A2)
                    return bodo.libs.int_arr_ext.apply_null_mask(
                                        out_arr, bitmap, mask_fill, inplace)
                return impl_left
            # right arg is IntegerArray
            if isinstance(A2, IntegerArrayType):
                def impl_right(A1, A2):
                    arr2 = bodo.libs.int_arr_ext.get_int_arr_data(A2)
                    bitmap = bodo.libs.int_arr_ext.get_int_arr_bitmap(A2)
                    out_arr = op(A1, arr2)
                    return bodo.libs.int_arr_ext.apply_null_mask(
                                        out_arr, bitmap, mask_fill, inplace)
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
