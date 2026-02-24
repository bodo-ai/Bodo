"""
Helper for performing MERGE INTO on Iceberg tables. Note that
MERGE INTO is only supported for BodoSQL and is currently disabled
due to various issues.
"""

import typing as pt

import numba
import pandas as pd

import bodo

import bodo.decorators  # isort:skip # noqa
from mpi4py import MPI

import bodo.utils.tracing as tracing
from bodo.io.iceberg.common import get_rest_catalog_config
from bodo.io.iceberg.write import (
    generate_data_file_info,
    iceberg_pq_write,
)
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    py_table_to_cpp_table,
)
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.utils.utils import BodoError


@numba.njit()
def iceberg_merge_cow(
    table_name,
    conn,
    database_schema,
    bodo_table,
    snapshot_id,
    old_fnames,
    col_names,
    df_pyarrow_schema,
    num_cols,
    is_parallel,
):  # pragma: no cover
    ev = tracing.Event("iceberg_merge_cow_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed DataFrames"

    with bodo.ir.object_mode.no_warning_objmode(
        already_exists="bool_",
        table_loc="unicode_type",
        partition_spec="python_list_of_heterogeneous_tuples_type",
        sort_order="python_list_of_heterogeneous_tuples_type",
        iceberg_schema_str="unicode_type",
        output_pyarrow_schema="pyarrow_schema_type",
        catalog_uri="unicode_type",
        bearer_token="unicode_type",
        warehouse="unicode_type",
    ):
        (
            table_loc,
            already_exists,
            _,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            output_pyarrow_schema,
            _,
        ) = get_table_details_before_write(
            table_name,
            conn,
            database_schema,
            df_pyarrow_schema,
            "append",
            allow_downcasting=True,
        )
        catalog_uri, bearer_token, warehouse = "", "", ""
        conf = get_rest_catalog_config(conn)
        if conf is not None:
            catalog_uri, bearer_token, warehouse = conf

    if not already_exists:
        raise ValueError("Iceberg MERGE INTO: Table does not exist at write")

    arrow_fs = None
    if catalog_uri and bearer_token and warehouse:
        # TODO: Update MERGE INTO to use PyIceberg
        pass

    dummy_theta_sketch = bodo.io.iceberg.theta.init_theta_sketches_wrapper(
        alloc_false_bool_array(num_cols)
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(table_loc, is_parallel)
    iceberg_files_info = iceberg_pq_write(
        table_loc,
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        is_parallel,
        output_pyarrow_schema,
        arrow_fs,
        dummy_theta_sketch,
        bucket_region,
    )

    with bodo.ir.object_mode.no_warning_objmode(success="bool_"):
        fnames, file_size_bytes, metrics = generate_data_file_info(iceberg_files_info)

        # Send file names, metrics and schema to Iceberg connector
        success = register_table_merge_cow(
            conn,
            database_schema,
            table_name,
            table_loc,
            old_fnames,
            fnames,
            file_size_bytes,
            metrics,
            snapshot_id,
        )

    if not success:
        # TODO [BE-3249] If it fails due to snapshot changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg MERGE INTO: write failed")

    bodo.io.iceberg.theta.delete_theta_sketches(dummy_theta_sketch)

    ev.finalize()


def register_table_merge_cow(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    old_fnames: list[str],
    new_fnames: list[str],
    file_size_bytes: list[int],
    all_metrics: dict[str, list[pt.Any]],  # TODO: Explain?
    snapshot_id: int,
):  # pragma: no cover
    """
    Wrapper around bodo_iceberg_connector.commit_merge_cow to run on
    a single rank and broadcast the result
    """
    ev = tracing.Event("iceberg_register_table_merge_cow")

    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD

    success = False
    if comm.Get_rank() == 0:
        success = bodo_iceberg_connector.commit_merge_cow(
            conn_str,
            db_name,
            table_name,
            table_loc,
            old_fnames,
            new_fnames,
            file_size_bytes,
            all_metrics,
            snapshot_id,
        )

    success: bool = comm.bcast(success)
    ev.finalize()
    return success


@numba.generated_jit(nopython=True)
def iceberg_merge_cow_py(
    table_name,
    conn,
    database_schema,
    bodo_df,
    snapshot_id,
    old_fnames,
    is_parallel=True,
):
    df_pyarrow_schema = bodo.io.helpers.numba_to_pyarrow_schema(
        bodo_df, is_iceberg=True
    )
    col_names_py = pd.array(bodo_df.columns)
    num_cols = len(col_names_py)

    if bodo_df.is_table_format:
        bodo_table_type = bodo_df.table_type

        def impl(
            table_name,
            conn,
            database_schema,
            bodo_df,
            snapshot_id,
            old_fnames,
            is_parallel=True,
        ):  # pragma: no cover
            iceberg_merge_cow(
                table_name,
                format_iceberg_conn_njit(conn),
                database_schema,
                py_table_to_cpp_table(
                    bodo.hiframes.pd_dataframe_ext.get_dataframe_table(bodo_df),
                    bodo_table_type,
                ),
                snapshot_id,
                old_fnames,
                array_to_info(col_names_py),
                df_pyarrow_schema,
                num_cols,
                is_parallel,
            )

    else:
        data_args = ", ".join(
            f"array_to_info(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(bodo_df, {i}))"
            for i in range(len(bodo_df.columns))
        )

        func_text = (
            "def impl(\n"
            "    table_name,\n"
            "    conn,\n"
            "    database_schema,\n"
            "    bodo_df,\n"
            "    snapshot_id,\n"
            "    old_fnames,\n"
            "    is_parallel=True,\n"
            "):\n"
            f"    info_list = [{data_args}]\n"
            "    table = arr_info_list_to_table(info_list)\n"
            "    iceberg_merge_cow(\n"
            "        table_name,\n"
            "        format_iceberg_conn_njit(conn),\n"
            "        database_schema,\n"
            "        table,\n"
            "        snapshot_id,\n"
            "        old_fnames,\n"
            "        array_to_info(col_names_py),\n"
            "        df_pyarrow_schema,\n"
            f"        {num_cols},\n"
            "        is_parallel,\n"
            "    )\n"
        )

        locals = {}
        globals = {
            "bodo": bodo,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "iceberg_merge_cow": iceberg_merge_cow,
            "format_iceberg_conn_njit": format_iceberg_conn_njit,
            "col_names_py": col_names_py,
            "df_pyarrow_schema": df_pyarrow_schema,
        }
        exec(func_text, globals, locals)
        impl = locals["impl"]

    return impl


def format_iceberg_conn_njit():
    pass


def get_table_details_before_write():
    pass
