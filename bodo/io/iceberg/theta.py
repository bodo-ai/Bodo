"""
Helper functions for interacting with Iceberg Puffin files for Theta Sketches.
"""

from __future__ import annotations

import sys
import typing as pt
from uuid import uuid4

import llvmlite.binding as ll
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    models,
    overload,
    register_model,
)

import bodo
from bodo.io.fs_io import pyarrow_fs_type
from bodo.io.helpers import pyarrow_schema_type
from bodo.io.iceberg.common import _format_data_loc, _fs_from_file_path
from bodo.libs import puffin_file, theta_sketches
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    delete_info,
    info_to_array,
)
from bodo.libs.str_ext import unicode_to_utf8
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.utils import run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.table import Transaction
    from pyiceberg.table.metadata import TableMetadataV2


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

# Create a type for the Iceberg StatisticsFile object
# if we have the connector.
statistics_file_type = None
try:
    import bodo_iceberg_connector

    statistics_file_type = bodo_iceberg_connector.StatisticsFile
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
    arr_type = bodo.boolean_array_type

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

    return (
        array_info_type(pyarrow_schema_type),
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
    arr_type = bodo.boolean_array_type

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

    return (
        array_info_type(pyarrow_schema_type),
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
    arr_type = bodo.FloatingArrayType(types.float64)

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
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="read_puffin_file_ndvs"
        )

        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        array_info_type(types.voidptr, types.voidptr, pyarrow_schema_type),
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
    arr_type = bodo.FloatingArrayType(types.float64)

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

    return (
        types.statistics_file_type(
            types.voidptr,  # Pass UTF-8 string as void*
            types.voidptr,  # const Pass UTF-8 string as void*
            types.int64,
            types.int64,
            theta_sketch_collection_type,
            output_pyarrow_schema_t,
            pyarrow_fs_type,
            types.voidptr,  # Pass UTF-8 string as void*
        ),
        codegen,
    )


@run_rank0
def fetch_puffin_metadata(txn: Transaction) -> tuple[int, int, str]:
    """
    Fetch the puffin file metadata that we need from the committed
    transaction to write the puffin file. These are the:
        1. Snapshot ID for the committed data
        2. Sequence Number for the committed data
        3. The Location at which to write the puffin file.

    Args:
        txn (Transaction): Transaction to get metadata from.

    Returns:
        tuple[int, int, str]: Tuple of the snapshot ID, sequence number, and
        location at which to write the puffin file.
    """
    metadata: TableMetadataV2 = txn.table_metadata

    snapshot_id = metadata.current_snapshot_id
    assert snapshot_id is not None

    snapshot = metadata.current_snapshot()
    assert snapshot is not None
    sequence_number = snapshot.sequence_number
    assert sequence_number is not None

    # Statistics file location
    location = _format_data_loc(
        f"{metadata.location}/metadata/{snapshot_id}-{uuid4()}.stats",
        _fs_from_file_path(metadata.location, txn._table.io),
    )

    return snapshot_id, sequence_number, location


@run_rank0
def commit_statistics_file(
    conn_str: str,
    table_id: str,
    snapshot_id: int,
    statistic_file_info,
) -> None:
    """
    Commit the statistics file to the iceberg table. This occurs after
    the puffin file has already been written and records the statistic_file_info
    in the metadata.

    Args:
        conn_str (str): The Iceberg connector string.
        table_id (str): The iceberg table identifier.
        snapshot_id (int): The snapshot ID of the committed data.
        statistic_file_info (bodo_iceberg_connector.StatisticsFile):
            The Python object containing the statistics file information.
    """
    import bodo_iceberg_connector

    assert bodo.get_rank() == 0, (
        "bodo/io/iceberg.py::get_old_statistics_file_path must be called from rank 0 only"
    )
    return bodo_iceberg_connector.commit_statistics_file(
        conn_str, table_id, snapshot_id, statistic_file_info
    )


@run_rank0
def table_columns_have_theta_sketches(
    conn_str: str, table_id: str
) -> pd.arrays.BooleanArray:
    import bodo_iceberg_connector

    assert bodo.get_rank() == 0, (
        "bodo/io/iceberg.py::table_columns_have_theta_sketches must be called from rank 0 only"
    )
    return bodo_iceberg_connector.table_columns_have_theta_sketches(conn_str, table_id)


@run_rank0
def table_columns_enabled_theta_sketches(txn: Transaction) -> pd.arrays.BooleanArray:
    """
    Get an array of booleans indicating whether each column in the table
    has theta sketches enabled, as per the table property of
    'bodo.write.theta_sketch_enabled.<column_name>'.

    Args:
        conn_str (str): The Iceberg connector string.
        db_name (str): The iceberg database name.
        table_name (str): The iceberg table.
    """
    cols = txn.table_metadata.schema().columns
    props: dict[str, str] = txn.table_metadata.properties

    enabled = [
        props.get(f"bodo.write.theta_sketch_enabled.{col.name}", "true") == "true"
        for col in cols
    ]
    return pd.array(enabled, dtype="boolean")  # type: ignore[return]


@run_rank0
def get_old_statistics_file_path(conn_str: str, table_id: str) -> str:
    """
    Get the old puffin file path from the connector. We know that the puffin file
    must exist because of previous checks.
    """
    import bodo_iceberg_connector

    assert bodo.get_rank() == 0, (
        "bodo/io/iceberg.py::get_old_statistics_file_path must be called from rank 0 only"
    )
    return bodo_iceberg_connector.get_old_statistics_file_path(conn_str, table_id)


def delete_theta_sketches(theta_sketches):  # pragma: no cover
    pass


@overload(delete_theta_sketches)
def overload_delete_theta_sketches(theta_sketches):
    """Delete the theta sketches"""

    def impl(theta_sketches):  # pragma: no cover
        _delete_theta_sketches(theta_sketches)

    return impl


@intrinsic
def _delete_theta_sketches(typingctx, theta_sketches_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
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
