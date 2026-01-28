"""
Operations for performing writes to Iceberg tables.
This file contains both code for
- The transaction handling (setup and teardown)
- Writing the Parquet files in the expected format
"""

from __future__ import annotations

import sys
import typing as pt

import llvmlite.binding as ll
import numba
import pyarrow as pa
from llvmlite import ir as lir
from mpi4py import MPI
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
import bodo.utils.tracing as tracing
from bodo.io import arrow_cpp
from bodo.io.helpers import pyarrow_fs_type, pyarrow_schema_type
from bodo.io.iceberg.theta import theta_sketch_collection_type
from bodo.io.iceberg.write_utils import (
    CreateTableMeta,
    generate_data_file_info_seq,
    register_table_write_seq,
    start_write_rank_0,
)
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.libs.str_ext import unicode_to_utf8
from bodo.spawn.utils import run_rank0
from bodo.utils.py_objs import install_opaque_class, install_py_obj_class
from bodo.utils.typing import CreateTableMetaType
from bodo.utils.utils import BodoError

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Transaction


# ----------------------- Compiler Utils ----------------------- #
ll.add_symbol("iceberg_pq_write_py_entry", arrow_cpp.iceberg_pq_write_py_entry)

try:
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Transaction
except ImportError:
    # PyIceberg is not installed
    PartitionSpec = None
    Transaction = None

this_module = sys.modules[__name__]

_, transaction_type = install_py_obj_class(
    types_name="transaction_type",
    module=this_module,
    python_type=Transaction,
    class_name="TransactionType",
    model_name="TransactionModel",
)
_, partition_spec_type = install_py_obj_class(
    types_name="partition_spec_type",
    module=this_module,
    python_type=PartitionSpec,
    class_name="PartitionSpecType",
    model_name="PartitionSpecModel",
)
_, python_list_of_heterogeneous_tuples_type = install_opaque_class(
    types_name="python_list_of_heterogeneous_tuples_type",
    module=this_module,
    class_name="PythonListOfHeterogeneousTuples",
)
_, dict_type = install_py_obj_class(
    types_name="dict_type",
    module=this_module,
    class_name="DictType",
    model_name="DictModel",
)


@intrinsic
def iceberg_pq_write_table_cpp(
    typingctx,
    table_data_loc_t,
    table_t,
    col_names_t,
    partition_spec_t,
    sort_order_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    iceberg_metadata_t,
    iceberg_schema_t,
    arrow_fs,
    sketch_collection_t,
):
    """
    Call C++ iceberg parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            # Iceberg Files Info (list of tuples)
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                # Partition Spec
                lir.IntType(8).as_pointer(),
                # Sort Order
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="iceberg_pq_write_py_entry"
        )

        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.python_list_of_heterogeneous_tuples_type(  # type: ignore
            types.voidptr,
            table_t,
            col_names_t,
            python_list_of_heterogeneous_tuples_type,
            python_list_of_heterogeneous_tuples_type,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
            pyarrow_schema_type,
            pyarrow_fs_type,
            theta_sketch_collection_type,
        ),
        codegen,
    )


# ----------------------- Helper Functions ----------------------- #


def generate_data_file_info(
    iceberg_files_info: list[tuple[pt.Any, pt.Any, pt.Any]],
) -> tuple[
    list[str] | None,
    list[dict[str, pt.Any]] | None,
    list[tuple] | None,
]:
    """
    Collect C++ Iceberg File Info to a single rank
    and process before handing off to the connector / committing functions
    """
    comm = MPI.COMM_WORLD
    # Information we need:
    # 1. File names
    # 2. file_size_in_bytes
    # Metrics we provide to Iceberg:
    # 1. recordCount -- Number of rows in this file
    # 2. valueCounts -- Number of records per field id. This is most useful for
    #    nested data types where each row may have multiple records.
    # 3. nullValueCounts - Null count per field id.
    # 4. lowerBounds - Lower bounds per field id.
    # 5. upperBounds - Upper bounds per field id.

    combined_data: list[list[tuple]] | None = comm.gather(iceberg_files_info)
    # Flatten the list of lists
    file_infos = (
        [item for sub in combined_data for item in sub] if combined_data else None
    )
    return generate_data_file_info_seq(file_infos)


register_table_write = run_rank0(register_table_write_seq)


@numba.njit
def iceberg_pq_write(
    table_loc,
    bodo_table,
    col_names,
    partition_spec,
    sort_order,
    iceberg_schema_str,
    is_parallel,
    expected_schema,
    arrow_fs,
    sketch_collection,
    bucket_region,
    properties,
):  # pragma: no cover
    """
    Writes a table to Parquet files in an Iceberg table's data warehouse
    following Iceberg rules and semantics.
    Args:
        table_loc (str): Location of the data/ folder in the warehouse
        bodo_table: Table object to pass to C++
        col_names: Array object containing column names (passed to C++)
        partition_spec: Array of Tuples containing Partition Spec for Iceberg Table (passed to C++)
        sort_order: Array of Tuples containing Sort Order for Iceberg Table (passed to C++)
        iceberg_schema_str (str): JSON Encoding of Iceberg Schema to include in Parquet metadata
        is_parallel (bool): Whether the write is occurring on a distributed DataFrame
        expected_schema (pyarrow.Schema): Expected schema of output PyArrow table written
            to Parquet files in the Iceberg table. None if not necessary
        arrow_fs (Arrow.fs.FileSystem): Optional Arrow FileSystem object to use for writing, will fallback to parsing
            the table_loc if not provided
        sketch_collection: collection of theta sketches being used to build NDV values during write

    Returns:
        Distributed list of written file info needed by Iceberg for committing
        1) file_path (after the table_loc prefix)
        2) record_count / Number of rows
        3) File size in bytes
        4) *partition-values
    """
    rg_size = -1
    with bodo.ir.object_mode.no_warning_objmode(compression="unicode_type"):
        compression = properties.get("write.parquet.compression-codec", "snappy")

    # Call the C++ function to write the parquet files.
    # Information about them will be returned as a list of tuples
    # See docstring for format
    iceberg_files_info = iceberg_pq_write_table_cpp(
        unicode_to_utf8(table_loc),
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        unicode_to_utf8(compression),
        is_parallel,
        unicode_to_utf8(bucket_region),
        rg_size,
        unicode_to_utf8(iceberg_schema_str),
        expected_schema,
        arrow_fs,
        sketch_collection,
    )

    return iceberg_files_info


def wrap_start_write(
    conn: str,
    table_id: str,
    df_schema: pa.Schema,
    if_exists: pt.Literal["fail", "append", "replace"],
    allow_downcasting: bool,
    create_table_info_arg: CreateTableMetaType | None = None,
):
    comm = MPI.COMM_WORLD

    txn = None
    result = ()
    err: Exception | None = None
    if comm.Get_rank() == 0:
        try:
            # Convert type to regular object to avoid compiler import in write_utils.py
            create_table_info = (
                CreateTableMeta(
                    create_table_info_arg.table_comment,
                    create_table_info_arg.column_comments,
                    create_table_info_arg.table_properties,
                )
                if create_table_info_arg
                else None
            )
            txn, *result = start_write_rank_0(
                conn,
                table_id,
                df_schema,
                if_exists,
                allow_downcasting,
                create_table_info,
            )
        except Exception as e:
            err = e

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    result = comm.bcast(result)
    return txn, *result


@numba.njit
def iceberg_write(
    conn,
    table_id,
    bodo_table,
    col_names,
    # Same semantics as pandas to_sql for now
    if_exists,
    is_parallel,
    df_pyarrow_schema,  # Additional Param to Compare Compile-Time and Iceberg Schema
    allow_downcasting,
):  # pragma: no cover
    """
    Iceberg Basic Write Implementation for parquet based tables.
    Args:
        conn (str): connection string
        table_id (str): Table Identifier of the iceberg database
        bodo_table : table object to pass to c++
        col_names : array object containing column names (passed to c++)
        if_exists (str): behavior when table exists. must be one of ['fail', 'append', 'replace']
        is_parallel (bool): whether the write is occurring on a distributed DataFrame
        df_pyarrow_schema (pyarrow.Schema): PyArrow schema of the DataFrame being written
        allow_downcasting (bool): Perform write downcasting on table columns to fit Iceberg schema
            This includes both type and nullability downcasting

    Raises:
        ValueError, Exception, BodoError
    """

    ev = tracing.Event("iceberg_write_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed DataFrames"
    with bodo.ir.object_mode.no_warning_objmode(
        txn="transaction_type",
        fs="pyarrow_fs_type",
        data_loc="unicode_type",
        output_schema="pyarrow_schema_type",
        iceberg_schema_str="unicode_type",
        partition_spec="partition_spec_type",
        partition_tuples="python_list_of_heterogeneous_tuples_type",
        sort_order_id="i8",
        sort_tuples="python_list_of_heterogeneous_tuples_type",
        num_cols="i8",
        properties=dict_type,
    ):
        (
            txn,
            fs,
            data_loc,
            output_schema,
            iceberg_schema_str,
            partition_spec,
            partition_tuples,
            sort_order_id,
            sort_tuples,
            properties,
        ) = wrap_start_write(
            conn,
            table_id,
            df_pyarrow_schema,
            if_exists,
            allow_downcasting,
        )
        num_cols = len(df_pyarrow_schema)

    dummy_theta_sketch = bodo.io.iceberg.theta.init_theta_sketches_wrapper(
        alloc_false_bool_array(num_cols)
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(data_loc, is_parallel)
    iceberg_files_info = iceberg_pq_write(
        data_loc,
        bodo_table,
        col_names,
        partition_tuples,
        sort_tuples,
        iceberg_schema_str,
        is_parallel,
        output_schema,
        fs,
        dummy_theta_sketch,
        bucket_region,
        properties,
    )

    with bodo.ir.object_mode.no_warning_objmode(success="bool_"):
        fnames, file_records, partition_infos = generate_data_file_info(
            iceberg_files_info
        )
        success = True
        # Send file names, metrics and schema to Iceberg connector
        success = register_table_write(
            txn,
            fnames,
            file_records,
            partition_infos,
            partition_spec,
            None if sort_order_id == 0 else sort_order_id,
        )

    if not success:
        # TODO [BE-3249] If it fails due to schema changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg write failed.")

    bodo.io.iceberg.theta.delete_theta_sketches(dummy_theta_sketch)
    ev.finalize()
