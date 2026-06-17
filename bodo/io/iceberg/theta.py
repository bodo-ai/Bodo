"""
Helper functions for interacting with Iceberg Puffin files for Theta Sketches.
"""

from __future__ import annotations

import sys
import typing as pt

import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    models,
    overload,
    register_model,
)

import bodo
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.libs import puffin_file, theta_sketches
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    delete_info,
    info_to_array,
)
from bodo.libs.str_ext import unicode_to_utf8
from bodo.spawn.utils import run_rank0
from bodo.utils.py_objs import install_py_obj_class

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.table.statistics import StatisticsFile

# Create a type for the Iceberg StatisticsFile object
# if we have the connector.
statistics_file_type = None
try:
    from pyiceberg.table.statistics import StatisticsFile

    statistics_file_type = StatisticsFile
except ImportError:
    pass

this_module = sys.modules[__name__]
install_py_obj_class(
    types_name="statistics_file_type",
    python_type=statistics_file_type,
    module=this_module,
    class_name="StatisticsFileType",
    model_name="StatisticsFileModel",
)


# Lazy import for bodo.io.helpers to avoid pulling in numba/hiframes
# when this module is imported from non-JIT code (DataFrame library).
_pyarrow_fs_type = None
_pyarrow_schema_type = None


def _ensure_pyarrow_types():
    global _pyarrow_fs_type, _pyarrow_schema_type
    if _pyarrow_fs_type is None:
        from bodo.io.helpers import pyarrow_fs_type, pyarrow_schema_type

        _pyarrow_fs_type = pyarrow_fs_type
        _pyarrow_schema_type = pyarrow_schema_type


ll.add_symbol("init_theta_sketches", theta_sketches.init_theta_sketches_py_entrypt)
ll.add_symbol("delete_theta_sketches", theta_sketches.delete_theta_sketches_py_entrypt)
ll.add_symbol(
    "fetch_ndv_approximations", theta_sketches.fetch_ndv_approximations_py_entrypt
)
ll.add_symbol("write_puffin_file", puffin_file.write_puffin_file_py_entrypt)
ll.add_symbol("read_puffin_file_ndvs", puffin_file.read_puffin_file_ndvs_py_entrypt)
ll.add_symbol(
    "get_supported_theta_sketch_columns",
    theta_sketches.get_supported_theta_sketch_columns_py_entrypt,
)
ll.add_symbol(
    "get_default_theta_sketch_columns",
    theta_sketches.get_default_theta_sketch_columns_py_entrypt,
)


class ThetaSketchCollectionType(types.Type):
    """Type for C++ pointer to a collection of theta sketches"""

    def __init__(self):  # pragma: no cover
        super().__init__(name="ThetaSketchCollectionType(r)")


register_model(ThetaSketchCollectionType)(models.OpaqueModel)

theta_sketch_collection_type = ThetaSketchCollectionType()


@intrinsic
def _init_theta_sketches(
    typingctx,
    column_bitmask_t,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # table_info*
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_theta_sketches"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        theta_sketch_collection_type(array_info_type),
        codegen,
    )


def get_supported_theta_sketch_columns(iceberg_pyarrow_schema):  # pragma: no cover
    pass


@overload(get_supported_theta_sketch_columns)
def overload_get_supported_theta_sketch_columns(iceberg_pyarrow_schema):
    """
    Returns a boolean array indicating which columns have types that can
    support theta sketches.
    """
    arr_type = bodo.types.boolean_array_type

    def impl(iceberg_pyarrow_schema):  # pragma: no cover
        res_info = _get_supported_theta_sketch_columns(iceberg_pyarrow_schema)
        res = info_to_array(res_info, arr_type)
        delete_info(res_info)
        return res

    return impl


@intrinsic
def _get_supported_theta_sketch_columns(typingctx, iceberg_pyarrow_schema_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # array_info*
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_supported_theta_sketch_columns"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    _ensure_pyarrow_types()
    return (
        array_info_type(_pyarrow_schema_type),
        codegen,
    )


def get_default_theta_sketch_columns(iceberg_pyarrow_schema):  # pragma: no cover
    pass


@overload(get_default_theta_sketch_columns)
def overload_get_default_theta_sketch_columns(iceberg_pyarrow_schema):
    """
    Returns a boolean array indicating which columns have types that output
    theta sketches by default.
    """
    arr_type = bodo.types.boolean_array_type

    def impl(iceberg_pyarrow_schema):  # pragma: no cover
        res_info = _get_default_theta_sketch_columns(iceberg_pyarrow_schema)
        res = info_to_array(res_info, arr_type)
        delete_info(res_info)
        return res

    return impl


@intrinsic
def _get_default_theta_sketch_columns(typingctx, iceberg_pyarrow_schema_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # array_info*
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_default_theta_sketch_columns"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    _ensure_pyarrow_types()
    return (
        array_info_type(_pyarrow_schema_type),
        codegen,
    )


def init_theta_sketches_wrapper(column_bitmask):  # pragma: no cover
    pass


@overload(init_theta_sketches_wrapper)
def overload_init_theta_sketches_wrapper(column_bit_mask):
    """
    Creates a new theta sketch collection when starting to write an Iceberg table.

    Args:
        column_bit_mask (Boolean Array): An array of booleans indicating which columns
            have theta sketches enabled.

    Returns:
        C++ Object: A new theta sketch collection object, which is effectively a pointer
            to an array of theta sketches, with null entries for columns without sketches.
    """

    def impl(column_bit_mask):  # pragma: no cover
        return _init_theta_sketches(array_to_info(column_bit_mask))

    return impl


@intrinsic
def _iceberg_writer_fetch_theta(typingctx, array_info_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # array_info*
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="fetch_ndv_approximations"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        array_info_type(theta_sketch_collection_type),
        codegen,
    )


def iceberg_writer_fetch_theta(writer):
    pass


@overload(iceberg_writer_fetch_theta)
def overload_iceberg_writer_fetch_theta(writer):
    """
    Fetches the current values of the theta sketch approximations
    of NDV for each column in an iceberg writer. For each column
    that does not have a theta sketch, returns null instead. Largely
    used for testing purposes.
    """
    arr_type = bodo.types.FloatingArrayType(types.float64)

    def impl(writer):  # pragma: no cover
        res_info = _iceberg_writer_fetch_theta(writer["theta_sketches"])
        res = info_to_array(res_info, arr_type)
        delete_info(res_info)
        return res

    return impl


@intrinsic
def _read_puffin_file_ndvs(typingctx, puffin_loc_t, bucket_region_t, iceberg_schema_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # array_info*
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="read_puffin_file_ndvs"
        )

        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    _ensure_pyarrow_types()
    return (
        array_info_type(types.voidptr, types.voidptr, _pyarrow_schema_type),
        codegen,
    )


def read_puffin_file_ndvs(puffin_file_loc):  # pragma: no cover
    pass


@overload(read_puffin_file_ndvs)
def overload_read_puffin_file_ndvs(puffin_file_loc, iceberg_schema):
    """
    Reads the NDV values from a puffin file. This is used for testing purposes
    to verify that the theta sketches are being written correctly.
    """
    arr_type = bodo.types.FloatingArrayType(types.float64)

    def impl(puffin_file_loc, iceberg_schema):  # pragma: no cover
        bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(
            puffin_file_loc, parallel=True
        )
        res_info = _read_puffin_file_ndvs(
            unicode_to_utf8(puffin_file_loc),
            unicode_to_utf8(bucket_region),
            iceberg_schema,
        )
        res = info_to_array(res_info, arr_type)
        delete_info(res_info)
        return res

    return impl


@intrinsic
def _write_puffin_file(
    typingctx,
    puffin_file_loc_t,
    bucket_region_t,
    snapshot_id_t,
    sequence_number_t,
    theta_sketches_t,
    output_pyarrow_schema_t,
    arrow_fs_t,
    exist_puffin_loc_t,
):
    def codegen(context, builder, sig, args):
        (
            puffin_file_loc,
            bucket_region,
            snapshot_id,
            sequence_number,
            theta_sketches,
            output_pyarrow_schema,
            arrow_fs,
            exist_puffin_loc,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # puffin_file_loc
                lir.IntType(8).as_pointer(),  # bucket_region
                lir.IntType(64),  # snapshot_id
                lir.IntType(64),  # sequence_number
                lir.IntType(8).as_pointer(),  # theta_sketches
                lir.IntType(8).as_pointer(),  # output_pyarrow_schema
                lir.IntType(8).as_pointer(),  # arrow_fs
                lir.IntType(8).as_pointer(),  # exist_puffin_loc
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="write_puffin_file"
        )
        ret = builder.call(
            fn_tp,
            [
                puffin_file_loc,
                bucket_region,
                snapshot_id,
                sequence_number,
                theta_sketches,
                output_pyarrow_schema,
                arrow_fs,
                exist_puffin_loc,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        # Wrap the PyObject* in a version that can track reference counts.
        return bodo.utils.py_objs.create_struct_from_pyobject(
            sig.return_type, ret, context, builder, context.get_python_api(builder)
        )

    _ensure_pyarrow_types()
    return (
        types.statistics_file_type(
            types.voidptr,  # Pass UTF-8 string as void*
            types.voidptr,  # const Pass UTF-8 string as void*
            types.int64,
            types.int64,
            theta_sketch_collection_type,
            output_pyarrow_schema_t,
            _pyarrow_fs_type,
            types.voidptr,  # Pass UTF-8 string as void*
        ),
        codegen,
    )


@run_rank0
def commit_statistics_file(
    conn_str: str,
    table_id: str,
    statistics_file: StatisticsFile,
) -> None:
    table = conn_str_to_catalog(conn_str).load_table(table_id).refresh()
    with table.update_statistics() as update:
        update.set_statistics(statistics_file)


def delete_theta_sketches(theta_sketches):  # pragma: no cover
    pass


@overload(delete_theta_sketches)
def overload_delete_theta_sketches(theta_sketches):
    def impl(theta_sketches):  # pragma: no cover
        _delete_theta_sketches(theta_sketches)

    return impl


@intrinsic
def _delete_theta_sketches(typingctx, theta_sketches_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_theta_sketches"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.void(theta_sketch_collection_type),
        codegen,
    )
