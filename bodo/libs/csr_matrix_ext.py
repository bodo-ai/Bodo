# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""CSR Matrix data type implementation for scipy.sparse.csr_matrix
"""

import operator

import numba
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
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.utils.typing import BodoError


class CSRMatrixType(types.ArrayCompatible):
    """Data type for scipy.sparse.csr_matrix"""

    ndim = 2

    def __init__(self, dtype, idx_dtype):
        self.dtype = dtype
        # idx_dtype is data type of row/column index values, either int32 or int64
        self.idx_dtype = idx_dtype
        super(CSRMatrixType, self).__init__(name=f"CSRMatrixType({dtype}, {idx_dtype})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 2, "C")

    def copy(self):
        return CSRMatrixType(self.dtype, self.idx_dtype)


# TODO(ehsan): make CSRMatrixType inner data mutable using a payload structure
@register_model(CSRMatrixType)
class CSRMatrixModel(models.StructModel):
    """CSR Matrix data model, storing values, row indices and column indices"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(fe_type.dtype, 1, "C")),
            ("indices", types.Array(fe_type.idx_dtype, 1, "C")),
            ("indptr", types.Array(fe_type.idx_dtype, 1, "C")),
            ("shape", types.UniTuple(types.int64, 2)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(CSRMatrixType, "data", "data")
make_attribute_wrapper(CSRMatrixType, "indices", "indices")
make_attribute_wrapper(CSRMatrixType, "indptr", "indptr")
make_attribute_wrapper(CSRMatrixType, "shape", "shape")


@intrinsic
def init_csr_matrix(typingctx, data_t, indices_t, indptr_t, shape_t):
    """Create a CSR matrix with provided data values."""
    assert isinstance(data_t, types.Array)
    assert isinstance(indices_t, types.Array) and isinstance(
        indices_t.dtype, types.Integer
    )
    assert indices_t == indptr_t

    def codegen(context, builder, signature, args):
        data, indices, indptr, shape = args
        # create csr matrix struct and store values
        csr_matrix = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        csr_matrix.data = data
        csr_matrix.indices = indices
        csr_matrix.indptr = indptr
        csr_matrix.shape = shape

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data)
        context.nrt.incref(builder, signature.args[1], indices)
        context.nrt.incref(builder, signature.args[2], indptr)

        return csr_matrix._getvalue()

    ret_typ = CSRMatrixType(data_t.dtype, indices_t.dtype)
    sig = ret_typ(data_t, indices_t, indptr_t, types.UniTuple(types.int64, 2))
    return sig, codegen


if bodo.utils.utils.has_scipy():
    import scipy.sparse

    @typeof_impl.register(scipy.sparse.csr_matrix)
    def _typeof_csr_matrix(val, c):
        """get Numba type for csr_matrix object"""
        dtype = numba.from_dtype(val.dtype)
        idx_dtype = numba.from_dtype(val.indices.dtype)
        return CSRMatrixType(dtype, idx_dtype)


@unbox(CSRMatrixType)
def unbox_csr_matrix(typ, val, c):
    """
    Unbox a scipy.sparse.csv_matrix
    """

    csr_matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data_obj = c.pyapi.object_getattr_string(val, "data")
    indices_obj = c.pyapi.object_getattr_string(val, "indices")
    indptr_obj = c.pyapi.object_getattr_string(val, "indptr")
    shape_obj = c.pyapi.object_getattr_string(val, "shape")
    csr_matrix.data = c.pyapi.to_native_value(
        types.Array(typ.dtype, 1, "C"), data_obj
    ).value
    csr_matrix.indices = c.pyapi.to_native_value(
        types.Array(typ.idx_dtype, 1, "C"), indices_obj
    ).value
    csr_matrix.indptr = c.pyapi.to_native_value(
        types.Array(typ.idx_dtype, 1, "C"), indptr_obj
    ).value
    csr_matrix.shape = c.pyapi.to_native_value(
        types.UniTuple(types.int64, 2), shape_obj
    ).value
    c.pyapi.decref(data_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(indptr_obj)
    c.pyapi.decref(shape_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(csr_matrix._getvalue(), is_error=is_error)


@box(CSRMatrixType)
def box_csr_matrix(typ, val, c):
    """box scipy.sparse.csv_matrix into python objects"""
    mod_name = c.context.insert_const_string(c.builder.module, "scipy.sparse")
    sc_sp_class_obj = c.pyapi.import_module_noblock(mod_name)

    csr_matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # box data, indices, indptr, shape
    c.context.nrt.incref(c.builder, types.Array(typ.dtype, 1, "C"), csr_matrix.data)
    data_obj = c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, "C"), csr_matrix.data, c.env_manager
    )
    c.context.nrt.incref(
        c.builder, types.Array(typ.idx_dtype, 1, "C"), csr_matrix.indices
    )
    indices_obj = c.pyapi.from_native_value(
        types.Array(typ.idx_dtype, 1, "C"), csr_matrix.indices, c.env_manager
    )
    c.context.nrt.incref(
        c.builder, types.Array(typ.idx_dtype, 1, "C"), csr_matrix.indptr
    )
    indptr_obj = c.pyapi.from_native_value(
        types.Array(typ.idx_dtype, 1, "C"), csr_matrix.indptr, c.env_manager
    )
    shape_obj = c.pyapi.from_native_value(
        types.UniTuple(types.int64, 2), csr_matrix.shape, c.env_manager
    )

    # call scipy.sparse.csr_matrix((data, indices, indptr), shape)
    arg1_obj = c.pyapi.tuple_pack([data_obj, indices_obj, indptr_obj])
    res = c.pyapi.call_method(sc_sp_class_obj, "csr_matrix", (arg1_obj, shape_obj))

    c.pyapi.decref(arg1_obj)
    c.pyapi.decref(data_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(indptr_obj)
    c.pyapi.decref(shape_obj)
    c.pyapi.decref(sc_sp_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


# scipy.sparse.csr_matrix doesn't provide len() but we support it for consistency
@overload(len, no_unliteral=True)
def overload_csr_matrix_len(A):
    if isinstance(A, CSRMatrixType):
        return lambda A: A.shape[0]  # pragma: no cover


@overload_attribute(CSRMatrixType, "ndim")
def overload_csr_matrix_ndim(A):
    return lambda A: 2  # pragma: no cover


@overload_method(CSRMatrixType, "copy", no_unliteral=True)
def overload_csr_matrix_copy(A):
    def copy_impl(A):  # pragma: no cover
        return init_csr_matrix(
            A.data.copy(), A.indices.copy(), A.indptr.copy(), A.shape
        )

    return copy_impl


@overload(operator.getitem, no_unliteral=True)
def csr_matrix_getitem(A, idx):
    if not isinstance(A, CSRMatrixType):
        return

    _data_dtype = A.dtype
    _idx_dtype = A.idx_dtype

    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and isinstance(idx[0], types.SliceType)
        and isinstance(idx[1], types.SliceType)
    ):

        def impl(A, idx):  # pragma: no cover
            nrows, ncols = A.shape
            row_slice = numba.cpython.unicode._normalize_slice(idx[0], nrows)
            col_slice = numba.cpython.unicode._normalize_slice(idx[1], ncols)

            if row_slice.step != 1 or col_slice.step != 1:
                raise ValueError(
                    "CSR matrix slice getitem only supports step=1 currently"
                )

            # based on
            # https://github.com/scipy/scipy/blob/e198e0a819a0ae89e9d161076ad5bdc8466a40bc/scipy/sparse/sparsetools/csr.h#L1180
            ir0 = row_slice.start
            ir1 = row_slice.stop
            ic0 = col_slice.start
            ic1 = col_slice.stop
            Ap = A.indptr
            Aj = A.indices
            Ax = A.data
            new_n_row = ir1 - ir0
            new_n_col = ic1 - ic0
            new_nnz = 0
            kk = 0

            # Count nonzeros total/per row.
            for i in range(new_n_row):
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]

                for jj in range(row_start, row_end):
                    if (Aj[jj] >= ic0) and (Aj[jj] < ic1):
                        new_nnz += 1

            # Allocate.
            Bp = np.empty(new_n_row + 1, _idx_dtype)
            Bj = np.empty(new_nnz, _idx_dtype)
            Bx = np.empty(new_nnz, _data_dtype)

            # Assign.
            Bp[0] = 0

            for i in range(new_n_row):
                row_start = Ap[ir0 + i]
                row_end = Ap[ir0 + i + 1]

                for jj in range(row_start, row_end):
                    if (Aj[jj] >= ic0) and (Aj[jj] < ic1):
                        Bj[kk] = Aj[jj] - ic0
                        Bx[kk] = Ax[jj]
                        kk += 1
                Bp[i + 1] = kk

            return init_csr_matrix(Bx, Bj, Bp, (new_n_row, new_n_col))

        return impl

    elif isinstance(idx, types.Array) and idx.ndim == 1 and idx.dtype == _idx_dtype:
        # Indexing into a CSR matrix by a 1D array of rows.
        # Used in conjunction with KFold / train_test_split

        def impl(A, idx):  # pragma: no cover
            nrows, ncols = A.shape

            # based on
            # https://github.com/scipy/scipy/blob/e198e0a819a0ae89e9d161076ad5bdc8466a40bc/scipy/sparse/sparsetools/csr.h#L1249
            Ap = A.indptr
            Aj = A.indices
            Ax = A.data
            new_n_row = len(idx)
            new_nnz = 0
            kk = 0

            # Count nonzeros total/per row.
            for i in range(new_n_row):
                row = idx[i]
                row_start = Ap[row]
                row_end = Ap[row + 1]
                new_nnz += row_end - row_start

            # Allocate.
            Bp = np.empty(new_n_row + 1, _idx_dtype)
            Bj = np.empty(new_nnz, _idx_dtype)
            Bx = np.empty(new_nnz, _data_dtype)

            # Assign.
            Bp[0] = 0

            for i in range(new_n_row):
                row = idx[i]
                row_start = Ap[row]
                row_end = Ap[row + 1]

                # This code implements `Bj = std::copy(Aj + row_start, Aj + row_end, Bj)`.
                # kk denotes the current offset into Bj. We copy `row_end - row_start`
                # bits between `Aj + row_start` and `Aj + row_end` into Bj starting at kk.
                # We also update Bp with kk which denotes where each row ends.
                Bj[kk : kk + row_end - row_start] = Aj[row_start:row_end]
                Bx[kk : kk + row_end - row_start] = Ax[row_start:row_end]
                kk += row_end - row_start
                Bp[i + 1] = kk

            out = init_csr_matrix(Bx, Bj, Bp, (new_n_row, ncols))
            return out

        return impl

    raise BodoError(
        f"getitem for CSR matrix with index type {idx} not supported yet."
    )  # pragma: no cover
