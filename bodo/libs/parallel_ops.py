"""
Implements array operations used by parallel only implementations.
This is used for functions that aren't inlined to avoid compilation
time issues.
"""

import bodo


def get_array_op_describe_dispatcher(arr_typ):
    """
    Helper function to simplify the distributed pass code.
    """
    if arr_typ.dtype == bodo.types.datetime64ns or isinstance(
        arr_typ, bodo.types.DatetimeArrayType
    ):
        return array_op_describe_parallel_dt
    return array_op_describe_parallel


array_op_describe_parallel = bodo.jit(distributed=["arr"], spawn=False)(
    bodo.libs.array_ops.array_op_describe_impl
)
array_op_describe_parallel_dt = bodo.jit(distributed=["arr"], spawn=False)(
    bodo.libs.array_ops.array_op_describe_dt_impl
)

array_op_nbytes_parallel = bodo.jit(distributed=["arr"], spawn=False)(
    bodo.libs.array_ops.array_op_nbytes_impl
)
