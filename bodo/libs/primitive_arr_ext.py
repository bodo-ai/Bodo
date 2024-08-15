# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Basic 1D array of primitive fixed-sized values to replace character Numpy array in
string array payload.
The goal is to avoid storing data pointer directly to allow meminfo pointer changes in
buffer pool manager. See:
https://bodo.atlassian.net/browse/BSE-528
"""
import operator

import numba
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
)

import bodo
from bodo.utils.cg_helpers import meminfo_to_np_arr
from bodo.utils.typing import BodoArrayIterator

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import array_ext, hstr_ext  # noqa: F401  # isort:skip


class PrimitiveArrayType(types.IterableType, types.ArrayCompatible):
    """1D array of primitive fixed-sized values (int, float, etc.)"""

    def __init__(self, dtype):
        self.dtype = dtype
        super(PrimitiveArrayType, self).__init__(name=f"PrimitiveArrayType({dtype})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return PrimitiveArrayType(self.dtype)

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)


@register_model(PrimitiveArrayType)
class PrimitiveArrayModel(models.StructModel):
    """Store meminfo and data pointer offset (to support getitem views), but not data
    pointer directly to allow meminfo pointer updates by buffer pool manager.
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("length", types.int64),
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("meminfo_offset", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(PrimitiveArrayType, "length", "_length")

lower_builtin("getiter", PrimitiveArrayType)(numba.np.arrayobj.getiter_array)


@overload(len, no_unliteral=True)
def overload_primitive_arr_len(A):
    if isinstance(A, PrimitiveArrayType):
        return lambda A: A._length  # pragma: no cover


@overload_attribute(PrimitiveArrayType, "shape")
def overload_primitive_arr_shape(A):
    return lambda A: (A._length,)  # pragma: no cover


@intrinsic
def alloc_primitive_array(typingctx, n_typ, dtype_typ):
    """Allocate a primitive array with specified length and dtype"""
    assert isinstance(n_typ, types.Integer) and isinstance(
        dtype_typ, types.DType
    ), "alloc_primitive_array: invalid arg types"
    dtype = dtype_typ.dtype

    def codegen(context, builder, signature, args):
        n, _ = args

        arr_type = types.Array(dtype, 1, "C")
        data_arr = bodo.utils.utils._empty_nd_impl(context, builder, arr_type, [n])

        # create array struct and store values
        out_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        out_arr.length = n
        out_arr.meminfo = data_arr.meminfo
        out_arr.meminfo_offset = context.get_constant(types.int64, 0)

        return out_arr._getvalue()

    ret_typ = PrimitiveArrayType(dtype)
    sig = ret_typ(types.int64, dtype_typ)
    return sig, codegen


@intrinsic
def primitive_to_np(typingctx, primitive_arr_t):
    """Convert primitive array to Numpy array (view with same meminfo) to allow reusing
    Numpy operations.
    """
    assert isinstance(
        primitive_arr_t, PrimitiveArrayType
    ), "primitive_to_np: primitive arr expected"
    np_arr_type = types.Array(primitive_arr_t.dtype, 1, "C")

    def codegen(context, builder, signature, args):
        (primitive_arr,) = args
        primitive_arr_struct = context.make_helper(
            builder, primitive_arr_t, primitive_arr
        )
        meminfo = primitive_arr_struct.meminfo
        meminfo_offset = primitive_arr_struct.meminfo_offset
        n = primitive_arr_struct.length
        np_arr = meminfo_to_np_arr(
            context, builder, meminfo, meminfo_offset, n, np_arr_type
        )
        return impl_ret_borrowed(context, builder, np_arr_type, np_arr)

    return np_arr_type(primitive_arr_t), codegen


@intrinsic
def np_to_primitive(typingctx, np_arr_t):
    """Convert Numpy array to primitive array (view with same meminfo)"""
    assert (
        isinstance(np_arr_t, types.Array)
        and np_arr_t.ndim == 1
        and np_arr_t.layout == "C"
    ), "np_to_primitive: 1D numpy arr expected"
    primitive_arr_type = PrimitiveArrayType(np_arr_t.dtype)

    def codegen(context, builder, signature, args):
        (np_arr,) = args
        np_arr_struct = context.make_helper(builder, np_arr_t, np_arr)

        # create array struct and store values
        out_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        out_arr.length = builder.extract_value(np_arr_struct.shape, 0)
        out_arr.meminfo = np_arr_struct.meminfo
        meminfo_data_ptr = context.nrt.meminfo_data(builder, np_arr_struct.meminfo)
        out_arr.meminfo_offset = builder.sub(
            builder.ptrtoint(np_arr_struct.data, lir.IntType(64)),
            builder.ptrtoint(meminfo_data_ptr, lir.IntType(64)),
        )

        return impl_ret_borrowed(
            context, builder, primitive_arr_type, out_arr._getvalue()
        )

    return primitive_arr_type(np_arr_t), codegen


@overload(operator.getitem, no_unliteral=True)
def primitive_arr_getitem(A, ind):
    """Support getitem by reusing Numpy's getitem"""
    if not isinstance(A, PrimitiveArrayType):
        return

    if isinstance(ind, types.Integer):

        def impl_scalar(A, ind):  # pragma: no cover
            np_arr = primitive_to_np(A)
            return np_arr[ind]

        return impl_scalar

    def impl(A, ind):  # pragma: no cover
        np_arr = primitive_to_np(A)
        return np_to_primitive(np_arr[ind])

    return impl


@overload(operator.setitem, no_unliteral=True)
def primitive_arr_setitem(A, idx, val):
    """Support setitem by reusing Numpy's getitem"""
    if not isinstance(A, PrimitiveArrayType):
        return

    if isinstance(val, PrimitiveArrayType):

        def impl_prim(A, idx, val):  # pragma: no cover
            np_arr = primitive_to_np(A)
            np_arr[idx] = primitive_to_np(val)

        return impl_prim

    def impl(A, idx, val):  # pragma: no cover
        np_arr = primitive_to_np(A)
        np_arr[idx] = val

    return impl


@overload_attribute(PrimitiveArrayType, "nbytes")
def prim_arr_nbytes_overload(A):
    return lambda A: primitive_to_np(A).nbytes  # pragma: no cover


@overload_attribute(PrimitiveArrayType, "ndim")
def overload_prim_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_method(PrimitiveArrayType, "copy", no_unliteral=True)
def overload_prim_arr_copy(A):
    return lambda A: np_to_primitive(primitive_to_np(A).copy())  # pragma: no cover


@overload_attribute(PrimitiveArrayType, "dtype")
def overload_prim_arr_dtype(A):
    return lambda A: primitive_to_np(A).dtype  # pragma: no cover
