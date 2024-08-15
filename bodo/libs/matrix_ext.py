# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Matrix data type implementation for np.matrix.
"""


import operator

import numpy as np
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    register_model,
    unbox,
)

from bodo.utils.typing import raise_bodo_error


class MatrixType(types.ArrayCompatible):
    """Data type for np.matrix"""

    ndim = 2

    def __init__(self, dtype, layout):
        self.dtype = dtype
        self.layout = layout
        super(MatrixType, self).__init__(
            name=f"MatrixType({dtype}, {repr(self.layout)})"
        )

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, self.layout)

    def copy(self):
        return MatrixType(self.dtype)


@register_model(MatrixType)
class MatrixModel(models.StructModel):
    """Matrix data model"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(fe_type.dtype, 2, fe_type.layout)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(MatrixType, "data", "data")


@unbox(MatrixType)
def unbox_matrix(typ, val, c):
    """
    Unbox a np.matrix from a Python object.
    """
    nativearycls = c.context.make_array(types.Array(typ.dtype, 2, typ.layout))
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()
    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    if c.context.enable_nrt:
        errcode = c.pyapi.nrt_adapt_ndarray_from_python(val, ptr)
    else:
        errcode = c.pyapi.numba_array_adaptor(val, ptr)
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    matrix.data = c.builder.load(aryptr)
    return NativeValue(
        matrix._getvalue(), is_error=cgutils.is_not_null(c.builder, errcode)
    )


@box(MatrixType)
def box_matrix(typ, val, c):
    """Box np.matrix into a Python object."""
    mod_name = c.context.insert_const_string(c.builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    # box data
    c.context.nrt.incref(c.builder, types.Array(typ.dtype, 2, "C"), matrix.data)
    data_obj = c.pyapi.from_native_value(
        types.Array(typ.dtype, 2, "C"), matrix.data, c.env_manager
    )
    # call np.matrix(data)
    res = c.pyapi.call_method(np_class_obj, "matrix", (data_obj,))
    c.pyapi.decref(data_obj)
    c.pyapi.decref(np_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@intrinsic
def init_np_matrix(typingctx, data_t):
    """Create a np.matrix with the provided underlying 2D array."""
    assert isinstance(data_t, types.Array) and data_t.ndim == 2

    def codegen(context, builder, signature, args):
        (data,) = args
        # create matrix struct and store values
        matrix = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        matrix.data = data
        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data)
        return matrix._getvalue()

    ret_typ = MatrixType(data_t.dtype, data_t.layout)
    sig = ret_typ(data_t)
    return sig, codegen


@overload(len, no_unliteral=True)
def overload_matrix_len(A):  # pragma: no cover
    if isinstance(A, MatrixType):
        return lambda A: len(A.data)


@overload_attribute(MatrixType, "shape")
def overload_matrix_shape(A):  # pragma: no cover
    return lambda A: A.data.shape


@overload_attribute(MatrixType, "ndim")
def overload_matrix_ndim(A):  # pragma: no cover
    return lambda A: 2


@overload_attribute(MatrixType, "T")
def overload_matrix_transpose(A):  # pragma: no cover
    return lambda A: init_np_matrix(A.data.T)


@overload(operator.getitem)
def overload_matrix_getitem(A, key):  # pragma: no cover
    if isinstance(A, MatrixType):
        raise_bodo_error("TODO: support getitem on np.matrix")


@overload(operator.setitem)
def overload_matrix_setitem(A, key, val):  # pragma: no cover
    if isinstance(A, MatrixType):
        raise_bodo_error("TODO: support setitem on np.matrix")


@overload(operator.add)
def overload_matrix_add(A, B):  # pragma: no cover
    if isinstance(A, MatrixType) and isinstance(B, MatrixType):
        return lambda A, B: init_np_matrix(A.data + B.data)


@overload(operator.sub)
def overload_matrix_sub(A, B):  # pragma: no cover
    if isinstance(A, MatrixType) and isinstance(B, MatrixType):
        return lambda A, B: init_np_matrix(A.data - B.data)


@overload(operator.mul)
@overload(operator.matmul)
@overload(np.dot)
@overload(np.matmul)
def overload_matrix_mul(A, B):  # pragma: no cover
    if isinstance(A, MatrixType) and isinstance(B, MatrixType):

        def impl(A, B):
            return init_np_matrix(np.dot(A.data, B.data))

        return impl


@overload(np.asmatrix)
def overload_np_asmatrix(A):
    """Implementation of np.asmatrix for various supported input formats:
        - Matrix -> returns the input unchanged
        - 1D array -> converts into a matrix with 1 row
        - 2D array -> wraps in a matrix
        - List of 1D arrays -> converts into a matrix with 1 row per element in the list
        - List of integer/float/complex -> Converts into a matrix with 1 row
        - Scalar integer/float/complex -> Converts into a 1x1 matrix

    Args:
        A (any scalar/np.ndarray/np.matrix): The input that is to be transformed into a matrix

    Returns:
        (np.matrix) A matrix of the input data.

    """
    if isinstance(A, MatrixType):
        return lambda A: A  # pragma: no cover
    if isinstance(A, types.Array) and A.ndim == 1:
        return lambda A: init_np_matrix(A.reshape((1, len(A))))  # pragma: no cover
    if isinstance(A, types.Array) and A.ndim == 2:
        return lambda A: init_np_matrix(A)  # pragma: no cover
    if (
        isinstance(A, types.List)
        and isinstance(A.dtype, types.Array)
        and A.dtype.ndim == 1
    ):
        dtype = A.dtype.dtype

        def impl(A):  # pragma: no cover
            rows = len(A)
            cols = 0
            for i in range(len(A)):
                if i == 0:
                    cols = len(A[i])
                elif len(A[i]) != cols:
                    print(i, cols, len(A[i]))
                    raise ValueError(
                        "np.asmatrix(List[array]) only valid when all arrays in the list have the same length"
                    )
            res = np.empty((rows, cols), dtype=dtype)
            for i in range(len(A)):
                res[i, :] = A[i]
            return init_np_matrix(res)

        return impl
    if isinstance(A, types.List) and isinstance(
        A.dtype, (types.Integer, types.Float, types.Complex)
    ):
        return lambda A: init_np_matrix(
            np.array([A]).reshape((1, len(A)))
        )  # pragma: no cover
    if isinstance(A, (types.Integer, types.Float, types.Complex)):
        return lambda A: np.asmatrix(np.array([A]))  # pragma: no cover
    raise_bodo_error(
        f"np.asmatrix unsupported on input of type {A}"
    )  # pragma: no cover


@overload(np.asarray)
def overload_np_asarray(A):  # pragma: no cover
    """
    An extra candidate overload for np.asarray to handle cases where
    the input is a Matrix. In this case, we return the underlying array.
    """
    if isinstance(A, MatrixType):
        return lambda A: A.data  # pragma: no cover
