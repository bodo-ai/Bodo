# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""CSR Matrix data type implementation for scipy.sparse.csr_matrix
"""

import numba
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
    unbox,
)

import bodo


class CSRMatrixType(types.ArrayCompatible):
    """Data type for scipy.sparse.csr_matrix"""

    def __init__(self, dtype, idx_dtype):
        self.dtype = dtype
        # idx_dtype is data type of row/column index values, either int32 or int64
        self.idx_dtype = idx_dtype
        super(CSRMatrixType, self).__init__(name=f"CSRMatrixType({dtype}, {idx_dtype})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

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


if bodo.config._has_scipy:
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
