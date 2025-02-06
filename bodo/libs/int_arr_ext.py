"""Nullable integer array corresponding to Pandas IntegerArray.
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
from bodo.libs.str_arr_ext import kBitmask

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hstr_ext  # isort:skip

ll.add_symbol("mask_arr_to_bitmap", hstr_ext.mask_arr_to_bitmap)


from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
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
    to_nullable_type,
)


class IntegerArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(name=f"IntegerArrayType({dtype})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return IntegerArrayType(self.dtype)

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)

    def unify(self, typingctx, other):
        """Allow casting Numpy int arrays to nullable int arrays"""
        if isinstance(other, types.Array) and other.ndim == 1:
            # If dtype matches or other.dtype is undefined (inferred)
            # Similar to Numba array unify:
            # https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/core/types/npytypes.py#L491
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return self

    @property
    def get_pandas_scalar_type_instance(self):
        """
        Get the Pandas dtype instance that matches stored
        scalars.
        """
        # Here we assume pd_int_dtype_classes is ordered int
        # then uint and in ascending bitwidth order.

        bitwidth_offset = int(np.log2(self.dtype.bitwidth // 8))
        signed_offset = 0 if self.dtype.signed else 4
        idx = bitwidth_offset + signed_offset
        return pd_int_dtype_classes[idx]()


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(IntegerArrayType)
class IntegerArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(fe_type.dtype, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntegerArrayType, "data", "_data")
make_attribute_wrapper(IntegerArrayType, "null_bitmap", "_null_bitmap")


lower_builtin("getiter", IntegerArrayType)(numba.np.arrayobj.getiter_array)


@typeof_impl.register(pd.arrays.IntegerArray)
def _typeof_pd_int_array(val, c):
    bitwidth = 8 * val.dtype.itemsize
    kind = "" if val.dtype.kind == "i" else "u"
    dtype = getattr(types, f"{kind}int{bitwidth}")
    return IntegerArrayType(dtype)


# dtype object for pd.Int64Dtype() etc.
class IntDtype(types.Number):
    """
    Type class associated with pandas Integer dtypes (e.g. pd.Int64Dtype,
    pd.UInt64Dtype).
    """

    def __init__(self, dtype):
        assert isinstance(dtype, types.Integer)
        self.dtype = dtype
        name = "{}Int{}Dtype()".format("" if dtype.signed else "U", dtype.bitwidth)
        super().__init__(name)


register_model(IntDtype)(models.OpaqueModel)


@box(IntDtype)
def box_intdtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)
    res = c.pyapi.call_method(pd_class_obj, str(typ)[:-2], ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(IntDtype)
def unbox_intdtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


def typeof_pd_int_dtype(val, c):
    bitwidth = 8 * val.itemsize
    kind = "" if val.kind == "i" else "u"
    dtype = getattr(types, f"{kind}int{bitwidth}")
    return IntDtype(dtype)


def _register_int_dtype(t):
    typeof_impl.register(t)(typeof_pd_int_dtype)
    int_dtype = typeof_pd_int_dtype(t(), None)
    type_callable(t)(lambda c: lambda: int_dtype)
    lower_builtin(t)(lambda c, b, s, a: c.get_dummy_value())


pd_int_dtype_classes = (
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
)


for t in pd_int_dtype_classes:
    _register_int_dtype(t)


@numba.extending.register_jitable
def mask_arr_to_bitmap(mask_arr):  # pragma: no cover
    n = len(mask_arr)
    n_bytes = (n + 7) >> 3
    bit_arr = np.empty(n_bytes, np.uint8)
    for i in range(n):
        b_ind = i // 8
        bit_arr[b_ind] ^= (
            np.uint8(-np.uint8(not mask_arr[i]) ^ bit_arr[b_ind]) & kBitmask[i % 8]
        )

    return bit_arr


@unbox(IntegerArrayType)
def unbox_int_array(typ, obj, c):
    """
    Convert a pd.arrays.IntegerArray object to a native IntegerArray structure.
    """
    return bodo.libs.array.unbox_array_using_arrow(typ, obj, c)


@box(IntegerArrayType)
def box_int_arr(typ, val, c):
    """Box int array into pandas ArrowExtensionArray object"""
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@intrinsic
def init_integer_array(typingctx, data, null_bitmap):
    """Create a IntegerArray with provided data and null bitmap values."""
    assert isinstance(data, types.Array)
    assert null_bitmap == types.Array(types.uint8, 1, "C")

    def codegen(context, builder, signature, args):
        data_val, bitmap_val = args
        # create int_arr struct and store values
        int_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        int_arr.data = data_val
        int_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], bitmap_val)

        return int_arr._getvalue()

    ret_typ = IntegerArrayType(data.dtype)
    sig = ret_typ(data, null_bitmap)
    return sig, codegen


@lower_constant(IntegerArrayType)
def lower_constant_int_arr(context, builder, typ, pyval):
    n = len(pyval)
    # Handle IntegerArray, np.array, and ArrowExtensionArray dtype conversion
    dtype = pyval.dtype
    if dtype == np.object_:
        dtype = np.int64
    elif isinstance(dtype, (pd.ArrowDtype, pd.core.dtypes.dtypes.BaseMaskedDtype)):
        dtype = dtype.numpy_dtype
    elif isinstance(dtype, pd.arrays.IntegerArray):
        dtype = pyval.dtype.type
    assert np.issubdtype(dtype, np.integer), f"Invalid dtype {dtype} for IntegerArray"
    data_arr = np.empty(n, dtype)
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

    # create int arr struct
    return lir.Constant.literal_struct([data_const_arr, nulls_const_arr])


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_int_arr_data(A):
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_int_arr_bitmap(A):
    return lambda A: A._null_bitmap


# array analysis extension
def get_int_arr_data_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_int_arr_ext_get_int_arr_data = (
    get_int_arr_data_equiv
)


def init_integer_array_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_int_arr_ext_init_integer_array = (
    init_integer_array_equiv
)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_integer_array(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 2
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_integer_array", "bodo.libs.int_arr_ext")
] = alias_ext_init_integer_array
numba.core.ir_utils.alias_func_extensions[
    ("get_int_arr_data", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_int_arr_bitmap", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func


# high-level allocation function for int arrays
@numba.njit(cache=True, no_cpython_wrapper=True)
def alloc_int_array(n, dtype):  # pragma: no cover
    data_arr = np.empty(n, dtype)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_integer_array(data_arr, nulls)


def alloc_int_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_int_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_int_arr_ext_alloc_int_array = (
    alloc_int_array_equiv
)


# NOTE: also used in regular Python in lower_constant_str_arr
@numba.njit(cache=True)
def set_bit_to_arr(bits, i, bit_is_set):  # pragma: no cover
    bits[i // 8] ^= np.uint8(-np.uint8(bit_is_set) ^ bits[i // 8]) & kBitmask[i % 8]


@numba.extending.register_jitable
def get_bit_bitmap_arr(bits, i):  # pragma: no cover
    return (bits[i >> 3] >> (i & 0x07)) & 1


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
def int_arr_getitem(A, ind):
    if not isinstance(A, IntegerArrayType):
        return

    if isinstance(ind, types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        return lambda A, ind: A._data[ind]

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_bool_index(A, ind)
            return init_integer_array(new_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_int_index(A, ind)
            return init_integer_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_slice_index(A, ind)
            return init_integer_array(new_data, new_mask)

        return impl_slice

    # This should be the only IntegerArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for IntegerArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True, jit_options={"cache": True})
def int_arr_setitem(A, idx, val):
    if not isinstance(A, IntegerArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for IntegerArray with indexing type {idx} received an incorrect 'value' type {val}."

    # Pandas allows booleans and integers but floats are necessary in many practical
    # cases in BodoSQL and otherwise. Numba unifies signed and unsigned integers into
    # Float currently. Numpy allows floats.
    # See test_bitwise.py::test_bitshiftright[vector_scalar_uint64_case]
    # TODO(Nick): Verify inputs can be safely cast to an integer
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

    # TODO(Nick): Verify inputs can be safely case to an integer
    if not (
        (
            is_iterable_type(val)
            and isinstance(val.dtype, (types.Integer, types.Boolean))
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

    # This should be the only IntegerArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for IntegerArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True, jit_options={"cache": True})
def numpy_arr_setitem(A, idx, val):
    """Support setitem of Numpy arrays with nullable int arrays"""
    if not (
        isinstance(A, types.Array)
        and isinstance(A.dtype, types.Integer)
        and isinstance(val, IntegerArrayType)
    ):
        return

    def impl_np_setitem_int_arr(A, idx, val):  # pragma: no cover
        # NOTE: NAs are lost in this operation if present so upstream operations should
        # make sure this is safe. For example, BodoSQL may know output is non-nullable
        # but internal operations may use nullable types by default.
        # See test_literals.py::test_array_literals_case"[integer_literals]"
        A[idx] = val._data

    return impl_np_setitem_int_arr


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_int_arr_len(A):
    if isinstance(A, IntegerArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(IntegerArrayType, "shape", jit_options={"cache": True})
def overload_int_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(IntegerArrayType, "dtype", jit_options={"cache": True})
def overload_int_arr_dtype(A):
    dtype_class = getattr(
        pd, "{}Int{}Dtype".format("" if A.dtype.signed else "U", A.dtype.bitwidth)
    )
    return lambda A: dtype_class()  # pragma: no cover


@overload_attribute(IntegerArrayType, "ndim", jit_options={"cache": True})
def overload_int_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(IntegerArrayType, "nbytes", jit_options={"cache": True})
def int_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


@overload_method(
    IntegerArrayType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_int_arr_copy(A, dtype=None):
    # TODO: Update dtype to do proper parsing with supported types.
    if not is_overload_none(dtype):
        return lambda A, dtype=None: A.astype(dtype, copy=True)  # pragma: no cover
    else:
        return lambda A, dtype=None: bodo.libs.int_arr_ext.init_integer_array(
            bodo.libs.int_arr_ext.get_int_arr_data(A).copy(),
            bodo.libs.int_arr_ext.get_int_arr_bitmap(A).copy(),
        )  # pragma: no cover


@overload_method(
    IntegerArrayType, "astype", no_unliteral=True, jit_options={"cache": True}
)
def overload_int_arr_astype(A, dtype, copy=True):
    # dtype becomes NumberClass if type reference is passed
    # see convert_to_nullable_tup in array_kernels.py
    # see test_series_concat_convert_to_nullable

    # Unwrap the dtype, if typeref
    if isinstance(dtype, types.TypeRef):
        # Unwrap TypeRef
        dtype = dtype.instance_type

    # If dtype is a string, force it to be a literal
    if dtype == types.unicode_type:
        raise_bodo_error(
            "IntegerArray.astype(): 'dtype' when passed as string must be a constant value"
        )

    if isinstance(dtype, types.NumberClass):
        dtype = dtype.dtype

    # same dtype case
    if isinstance(dtype, IntDtype) and A.dtype == dtype.dtype:
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

    # other IntDtype value, needs copy (TODO: copy mask?)
    if isinstance(dtype, IntDtype):
        np_dtype = dtype.dtype
        return lambda A, dtype, copy=True: bodo.libs.int_arr_ext.init_integer_array(
            bodo.libs.int_arr_ext.get_int_arr_data(A).astype(np_dtype),
            bodo.libs.int_arr_ext.get_int_arr_bitmap(A).copy(),
        )

    if isinstance(dtype, bodo.Decimal128Type):
        precision = dtype.precision
        scale = dtype.scale

        def impl_dec(A, dtype, copy=True):  # pragma: no cover
            data = bodo.libs.int_arr_ext.get_int_arr_data(A)
            n = len(data)
            B_data = np.empty(n, dtype=bodo.libs.decimal_arr_ext.int128_type)
            B_nulls = bodo.libs.int_arr_ext.get_int_arr_bitmap(A).copy()
            B = bodo.libs.decimal_arr_ext.init_decimal_array(
                B_data, B_nulls, precision, scale
            )
            for i in numba.parfors.parfor.internal_prange(n):
                if not bodo.libs.array_kernels.isna(A, i):
                    B[i] = data[i]
            return B

        return impl_dec

    # numpy dtypes
    nb_dtype = parse_dtype(dtype, "IntegerArray.astype")
    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):

        def impl_float(A, dtype, copy=True):  # pragma: no cover
            data = bodo.libs.int_arr_ext.get_int_arr_data(A)
            n = len(data)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                B[i] = data[i]
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = np.nan
            return B

        return impl_float

    # TODO: raise error like Pandas when NAs are assigned to integers
    return lambda A, dtype, copy=True: bodo.libs.int_arr_ext.get_int_arr_data(A).astype(
        nb_dtype
    )


@lower_cast(types.Array, IntegerArrayType)
def cast_numpy_to_nullable_int_array(context, builder, fromty, toty, val):
    """cast Numpy int array to nullable int array"""
    f = lambda A: bodo.utils.conversion.coerce_to_array(
        A, use_nullable_array=True
    )  # pragma: no cover
    return context.compile_internal(builder, f, toty(fromty), [val])


@lower_cast(IntegerArrayType, types.Array)
def cast_nullable_int_array_to_numpy(context, builder, fromty, toty, val):
    """cast nullable int array to Numpy int array"""
    dtype = toty.dtype
    f = lambda A: A.astype(dtype)  # pragma: no cover
    return context.compile_internal(builder, f, toty(fromty), [val])


@overload(np.asarray, jit_options={"cache": True})
def overload_asarray(A):
    """Support np.asarray() for nullable int arrays"""
    if not isinstance(A, IntegerArrayType):
        return

    def impl(A):  # pragma: no cover
        return get_int_arr_data(A)

    return impl


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


def create_op_overload(op, n_inputs):
    """creates overloads for operations on Integer arrays"""
    if n_inputs == 1:

        def overload_int_arr_op_nin_1(A):
            if isinstance(A, IntegerArrayType):
                return get_nullable_array_unary_impl(op, A)

        return overload_int_arr_op_nin_1
    elif n_inputs == 2:

        def overload_series_op_nin_2(lhs, rhs):
            if isinstance(lhs, IntegerArrayType) or isinstance(rhs, IntegerArrayType):
                return get_nullable_array_binary_impl(op, lhs, rhs)

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


# inlining in Series pass but avoiding inline="always" since there are Numba-only cases
# that don't need inlining such as repeats.sum() in repeat_kernel()
@overload_method(
    IntegerArrayType, "sum", no_unliteral=True, jit_options={"cache": True}
)
def overload_int_arr_sum(A, skipna=True, min_count=0):
    """A.sum() for nullable integer arrays"""
    unsupported_args = {"skipna": skipna, "min_count": min_count}
    arg_defaults = {"skipna": True, "min_count": 0}
    check_unsupported_args("IntegerArray.sum", unsupported_args, arg_defaults)

    def impl(A, skipna=True, min_count=0):  # pragma: no cover
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
    IntegerArrayType, "unique", no_unliteral=True, jit_options={"cache": True}
)
def overload_unique(A):
    dtype = A.dtype

    def impl_int_arr(A):  # pragma: no cover
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
            set_bit_to_arr(new_mask, j, mask[j])
        return init_integer_array(new_data, new_mask)

    return impl_int_arr


def get_nullable_array_unary_impl(op, A):
    """generate implementation for unary operation on nullable integer, float, or boolean array"""
    # use type inference to get output dtype
    typing_context = numba.core.registry.cpu_target.typing_context
    ret_dtype = typing_context.resolve_function_type(
        op, (types.Array(A.dtype, 1, "C"),), {}
    ).return_type
    ret_dtype = to_nullable_type(ret_dtype)

    def impl(A):  # pragma: no cover
        n = len(A)
        out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, None)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = op(A[i])
        return out_arr

    return impl


def get_nullable_array_binary_impl(op, lhs, rhs):
    """generate implementation for binary operation on nullable integer, float, or boolean array"""
    # TODO: 1 ** np.nan is 1. So we have to unmask those.
    inplace = (
        op in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
    )
    is_lhs_scalar = isinstance(lhs, (types.Number, types.Boolean))
    is_rhs_scalar = isinstance(rhs, (types.Number, types.Boolean))
    # use type inference to get output dtype
    # NOTE: using Numpy array instead of scalar dtypes since output dtype can be
    # different for arrays. For example, int32 + int32 is int64 for scalar but int32 for
    # arrays. see test_series_add.
    dtype1 = types.Array(getattr(lhs, "dtype", lhs), 1, "C")
    dtype2 = types.Array(getattr(rhs, "dtype", rhs), 1, "C")
    typing_context = numba.core.registry.cpu_target.typing_context
    ret_dtype = typing_context.resolve_function_type(
        op, (dtype1, dtype2), {}
    ).return_type
    ret_dtype = to_nullable_type(ret_dtype)

    # make sure there is no ZeroDivisionError (BE-200, test_div_by_zero)
    if op in (operator.truediv, operator.itruediv):
        op = np.true_divide
    elif op in (operator.floordiv, operator.ifloordiv):
        op = np.floor_divide

    # generate implementation function. Example:
    # def impl(lhs, rhs):
    #   n = len(lhs)
    #   out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, None)
    #   for i in numba.parfors.parfor.internal_prange(n):
    #     if (bodo.libs.array_kernels.isna(lhs, i)
    #         or bodo.libs.array_kernels.isna(rhs, i)):
    #       bodo.libs.array_kernels.setna(out_arr, i)
    #       continue
    #     out_arr[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(op(lhs[i], rhs[i]))
    #   return out_arr
    access_str1 = "lhs" if is_lhs_scalar else "lhs[i]"
    access_str2 = "rhs" if is_rhs_scalar else "rhs[i]"
    na_str1 = "False" if is_lhs_scalar else "bodo.libs.array_kernels.isna(lhs, i)"
    na_str2 = "False" if is_rhs_scalar else "bodo.libs.array_kernels.isna(rhs, i)"
    func_text = "def impl(lhs, rhs):\n"
    func_text += "  n = len({})\n".format("lhs" if not is_lhs_scalar else "rhs")
    if inplace:
        func_text += "  out_arr = {}\n".format("lhs" if not is_lhs_scalar else "rhs")
    else:
        func_text += "  out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, None)\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += f"    if ({na_str1}\n"
    func_text += f"        or {na_str2}):\n"
    func_text += "      bodo.libs.array_kernels.setna(out_arr, i)\n"
    func_text += "      continue\n"
    func_text += f"    out_arr[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(op({access_str1}, {access_str2}))\n"
    func_text += "  return out_arr\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "np": np,
            "ret_dtype": ret_dtype,
            "op": op,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


def get_int_array_op_pd_td(op):
    def impl(lhs, rhs):
        """generate implementation for binary operation on nullable integer array op timdelta"""
        # either the lhs or the rhs must be scalar, can't have an array of pd_timedelta types
        is_lhs_scalar = lhs in [pd_timedelta_type]
        is_rhs_scalar = rhs in [pd_timedelta_type]

        if is_lhs_scalar:

            def impl(lhs, rhs):  # pragma: no cover
                n = len(rhs)
                out_arr = np.empty(n, "timedelta64[ns]")
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue
                    out_arr[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                        op(lhs, rhs[i])
                    )
                return out_arr

            return impl
        elif is_rhs_scalar:

            def impl(lhs, rhs):  # pragma: no cover
                n = len(lhs)
                out_arr = np.empty(n, "timedelta64[ns]")
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue
                    out_arr[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                        op(lhs[i], rhs)
                    )
                return out_arr

            return impl

    return impl
