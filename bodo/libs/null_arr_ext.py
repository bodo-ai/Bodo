"""Array implementation for null array type. This is an array that contains
all null values and can be cast to any other array type.
"""

import operator

import numba
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.utils.typing import (
    dtype_to_array_type,
    is_list_like_index_type,
    is_scalar_type,
    to_nullable_type,
    unwrap_typeref,
)


class NullDType(types.Type):
    """
    Type that can be used to represent a null value
    that can be cast to any type.
    """

    def __init__(self):
        super().__init__(name="NullType()")


null_dtype = NullDType()

# The null dtype is just used to represent a null value in typing
register_model(NullDType)(models.OpaqueModel)


class NullArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="NullArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return null_dtype

    def copy(self):
        return NullArrayType()

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    def unify(self, typingctx, other):
        """Allow casting null array to all other arrays (converted to nullable if
        non-nullable).
        """
        if bodo.utils.utils.is_array_typ(other, False):
            return to_nullable_type(other)


null_array_type = NullArrayType()


# store the length of the array as the struct since all values are null
@register_model(NullArrayType)
class NullArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("length", types.int64),
            # Keep an extra field that is always 1 so we can determine
            # if the struct is null or not. We use context.get_constant_null
            # inside ensure_column_unboxed and this will become all 0s.
            # https://github.com/bodo-ai/Bodo/blob/3108eb47a7a79861739b1ae3a4939c1525ef16ae/bodo/hiframes/table.py#L1195
            # https://github.com/numba/numba/blob/135d15047c5237f751d4b81347effe2a3704288b/numba/core/base.py#L522
            # https://github.com/numba/llvmlite/blob/dffe582d6080494ba8e39689d09aacde1952214c/llvmlite/ir/values.py#L457
            # https://github.com/numba/llvmlite/blob/dffe582d6080494ba8e39689d09aacde1952214c/llvmlite/ir/types.py#L545
            ("not_empty", types.boolean),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(NullArrayType, "length", "_length")


@intrinsic
def init_null_array(typingctx, length_t):
    """Create a null array with the provided length."""

    def codegen(context, builder, signature, args):
        (length,) = args
        # create null_arr struct and store values
        null_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        null_arr.length = length
        null_arr.not_empty = lir.Constant(lir.IntType(1), 1)
        return null_arr._getvalue()

    sig = null_array_type(types.int64)
    return sig, codegen


def init_null_array_equiv(self, scope, equiv_set, loc, args, kws):
    """
    Array analysis for init_null_array. The shape is just the first argument.
    """
    assert len(args) == 1 and not kws
    var = args[0]
    return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_null_arr_ext_init_null_array = (
    init_null_array_equiv
)


@typeof_impl.register(pa.NullArray)
def typeof_null_array(val, c):
    return null_array_type


@box(NullArrayType)
def box_null_arr(typ, val, c):
    """Box null array into a pyarrow null array."""
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@unbox(NullArrayType)
def unbox_null_arr(typ, obj, c):
    """Unbox a null array via the length."""
    n_obj = c.pyapi.call_method(obj, "__len__", ())
    n = c.pyapi.long_as_longlong(n_obj)
    c.pyapi.decref(n_obj)
    null_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    null_arr.length = n
    null_arr.not_empty = lir.Constant(lir.IntType(1), 1)
    return NativeValue(null_arr._getvalue())


@overload(len, no_unliteral=True)
def overload_null_arr_len(A):
    if A == null_array_type:
        return lambda A: A._length  # pragma: no cover


@overload_attribute(NullArrayType, "shape")
def overload_null_arr_shape(A):
    return lambda A: (A._length,)  # pragma: no cover


@overload_attribute(NullArrayType, "ndim")
def overload_null_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(NullArrayType, "nbytes")
def overload_null_nbytes(A):
    # A null array always takes exactly 8 bytes
    return lambda A: 8  # pragma: no cover


@overload_method(NullArrayType, "copy")
def overload_null_copy(A):
    # Just return the same array since this array is immutable
    return lambda A: A  # pragma: no cover


@overload_method(NullArrayType, "astype", no_unliteral=True)
def overload_null_astype(A, dtype, copy=True):
    # Note we ignore the copy argument since this array
    # always requires a copy.
    new_dtype = unwrap_typeref(dtype)
    if bodo.utils.utils.is_array_typ(new_dtype, False):
        # Some internal types (e.g. Dictionary encode arrays)
        # must be passed as array types and not dtypes.
        nb_dtype = new_dtype
    else:
        nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)
    if (
        isinstance(
            nb_dtype,
            (bodo.libs.int_arr_ext.IntDtype, bodo.libs.float_arr_ext.FloatDtype),
        )
        or nb_dtype == bodo.libs.bool_arr_ext.boolean_dtype
    ):
        dtype = nb_dtype.dtype
    else:
        dtype = nb_dtype
    if is_scalar_type(dtype):
        dtype = dtype_to_array_type(dtype)
    _arr_typ = to_nullable_type(dtype)
    if _arr_typ == null_array_type:
        return lambda A, dtype, copy=True: A  # pragma: no cover
    else:

        def impl(A, dtype, copy=True):  # pragma: no cover
            return bodo.libs.array_kernels.gen_na_array(A._length, _arr_typ, True)

        return impl


@overload(operator.getitem, no_unliteral=True)
def null_arr_getitem(A, ind):
    if A != null_array_type:
        return

    if isinstance(ind, types.Integer):

        def impl(A, ind):  # pragma: no cover
            return None

        return impl

    # bool arr indexing.
    # array length is the number of true entries.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_array(ind)
            n = ind.sum()
            return init_null_array(n)

        return impl_bool

    # list of ints indexing
    # array length is the length of the list.
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl_list(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_array(ind)
            n = len(ind)
            return init_null_array(n)

        return impl_list

    # slice case
    # array length is number of entries in the range [0, len(arr)).
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            n = len(A)
            slice_idx = numba.cpython.unicode._normalize_slice(ind, n)
            final_len = (slice_idx.stop - slice_idx.start) // slice_idx.step
            return init_null_array(final_len)

        return impl_slice

    raise bodo.utils.typing.BodoError(
        f"getitem for NullArrayType with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def null_arr_setitem(arr, ind, val):
    """Null array setitem may be called in internal code like trim_excess_data
    See
    bodosql/tests/test_kernels/test_variadic_array_kernels.py::test_object_construct_keep_null"[2-int_vector-null]"
    """
    if arr != null_array_type:
        return

    return lambda arr, ind, val: None  # pragma: no cover
