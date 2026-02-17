from __future__ import annotations

import json
import typing as pt

import llvmlite.binding as ll
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_index_ext import SingleIndexType, array_type_to_index
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.time_ext import TimeArrayType
from bodo.io import arrow_cpp
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.utils.typing import (
    BodoError,
    assert_bodo_error,
    get_overload_const_bool,
    get_overload_const_list,
    is_overload_constant_bool,
    is_overload_none,
    is_str_arr_type,
)

if pt.TYPE_CHECKING:
    from bodo.hiframes.pd_dataframe_ext import DataFrameType

ll.add_symbol("pq_write_py_entry", arrow_cpp.pq_write_py_entry)
ll.add_symbol("pq_write_create_dir_py_entry", arrow_cpp.pq_write_create_dir_py_entry)
ll.add_symbol("pq_write_partitioned_py_entry", arrow_cpp.pq_write_partitioned_py_entry)


@intrinsic
def parquet_write_table_cpp(
    typingctx,
    filename_t,
    table_t,
    col_names_t,
    metadata_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    file_prefix,
    convert_timedelta_to_int64,
    timestamp_tz,
    downcast_time_ns_to_us,
    create_dir,
    force_hdfs=False,
):
    """
    Call C++ parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),  # path_name
                lir.IntType(8).as_pointer(),  # table
                lir.IntType(8).as_pointer(),  # col_names_arr
                lir.IntType(8).as_pointer(),  # metadata
                lir.IntType(8).as_pointer(),  # compression
                lir.IntType(1),  # is_parallel
                lir.IntType(8).as_pointer(),  # bucket_region
                lir.IntType(64),  # row_group_size
                lir.IntType(8).as_pointer(),  # file_prefix
                lir.IntType(1),  # convert_timedelta_to_int64
                lir.IntType(8).as_pointer(),  # tz
                lir.IntType(1),  # downcast_time_ns_to_us
                lir.IntType(1),  # create_dir
                lir.IntType(1),  # force_hdfs
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_write_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.int64(
            types.voidptr,  # path_name
            table_t,  # table
            col_names_t,  # col_names_arr
            types.voidptr,  # metadata
            types.voidptr,  # compression
            types.boolean,  # is_parallel
            types.voidptr,  # bucket_region
            types.int64,  # row_group_size
            types.voidptr,  # file_prefix
            types.boolean,  # convert_timedelta_to_int64
            types.voidptr,  # tz
            types.boolean,  # downcast_time_ns_to_us
            types.boolean,  # create dir
            types.boolean,  # force_hdfs
        ),
        codegen,
    )


@intrinsic
def pq_write_create_dir(
    typingctx,
    filename_t,
):
    """
    Call C++ parquet write directory creation function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_write_create_dir_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.none(
            types.voidptr,
        ),
        codegen,
    )


@intrinsic
def parquet_write_table_partitioned_cpp(
    typingctx,
    filename_t,
    data_table_t,
    col_names_t,
    col_names_no_partitions_t,
    cat_table_t,
    part_col_idxs_t,
    num_part_col_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    file_prefix,
    timestamp_tz,
):
    """
    Call C++ parquet write partitioned function

    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),  # tz
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_write_partitioned_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.void(
            types.voidptr,
            data_table_t,
            col_names_t,
            col_names_no_partitions_t,
            types.voidptr,
            types.voidptr,
            types.int32,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.voidptr,  # tz
        ),
        codegen,
    )


def gen_pandas_parquet_metadata_template(
    column_names,
    data_types,
    index_type: SingleIndexType | MultiIndexType,
    write_non_range_index_to_metadata: bool,
    write_rangeindex_to_metadata: bool,
    partition_cols: pt.Sequence[str] | None = None,
    is_runtime_columns: bool = False,
):
    """
    Returns dict with pandas column metadata for parquet storage.
    For more information, see:
    https://pandas.pydata.org/pandas-docs/stable/development/developer.html#storing-pandas-dataframe-objects-in-apache-parquet-format
    """

    if partition_cols is None:
        partition_cols = []

    pandas_metadata = {
        "index_columns": [],
        "column_indexes": [],
        "columns": [],
        "creator": {
            "library": "pyarrow",
            "version": pa.__version__,
        },
        "pandas_version": pd.__version__,
    }

    # Table column metadata
    for col_name, col_type in zip(column_names, data_types):
        if col_name in partition_cols:
            # partition columns are not written to parquet files, and don't appear
            # in pandas metadata
            continue
        # Currently only timezone types contain metadata
        metadata = None
        if isinstance(col_type, bodo.types.DatetimeArrayType):
            numpy_type = "datetime64[ns]"
            if col_type.tz is None:
                pandas_type = "datetime"
            else:
                pandas_type = "datetimetz"
                # Reuse pyarrow to construct the metadata.
                if isinstance(col_type.tz, int):
                    tz = bodo.libs.pd_datetime_arr_ext.nanoseconds_to_offset(
                        col_type.tz
                    )
                else:
                    tz = pd.DatetimeTZDtype(tz=col_type.tz).tz
                metadata = {"timezone": pa.lib.tzinfo_to_string(tz)}
        elif isinstance(col_type, types.Array) or col_type == boolean_array_type:
            pandas_type = numpy_type = col_type.dtype.name
            if numpy_type.startswith("datetime"):
                pandas_type = "datetime"
        elif is_str_arr_type(col_type):
            pandas_type = "unicode"
            numpy_type = "object"
        elif col_type == binary_array_type:
            pandas_type = "bytes"
            numpy_type = "object"
        elif isinstance(col_type, DecimalArrayType):
            pandas_type = numpy_type = "object"
        elif isinstance(col_type, IntegerArrayType):
            dtype_name = col_type.dtype.name
            # Pandas dtype is int8/uint8/int16/...
            # numpy dtype is Int8/UInt8/Int16/... (capitalize to specify nullable array)
            if dtype_name.startswith("int"):
                numpy_type = "Int" + dtype_name[3:]
            elif dtype_name.startswith("uint"):
                numpy_type = "UInt" + dtype_name[4:]
            else:  # pragma: no cover
                if is_runtime_columns:
                    # If columns are determined at runtime we don't have names
                    col_name = "Runtime determined column of type"
                raise BodoError(
                    f"to_parquet(): unknown dtype in nullable Integer column {col_name} {col_type}"
                )
            pandas_type = col_type.dtype.name
        elif isinstance(col_type, bodo.types.FloatingArrayType):
            dtype_name = col_type.dtype.name
            # Pandas dtype is float32/float64
            # numpy dtype is Float32/Float64 (capitalize to specify nullable array)
            pandas_type = dtype_name
            numpy_type = dtype_name.capitalize()
        elif col_type == datetime_date_array_type:
            pandas_type = "datetime"
            numpy_type = "object"
        elif isinstance(col_type, TimeArrayType):
            pandas_type = "datetime"
            numpy_type = "object"
        elif isinstance(
            col_type,
            (
                bodo.types.ArrayItemArrayType,
                bodo.types.StructArrayType,
                bodo.types.MapArrayType,
            ),
        ):
            # TODO: provide meaningful pandas_type when possible.
            # For example "pandas_type": "list[list[int64]]", "numpy_type": "object"
            pandas_type = "object"
            numpy_type = "object"
        # TODO: metadata for categorical arrays
        else:  # pragma: no cover
            if is_runtime_columns:
                # If columns are determined at runtime we don't have names
                col_name = "Runtime determined column of type"
            raise BodoError(
                f"to_parquet(): unsupported column type for metadata generation : {col_name} {col_type}"
            )

        col_metadata = {
            "name": col_name,
            "field_name": col_name,
            "pandas_type": pandas_type,
            "numpy_type": numpy_type,
            "metadata": metadata,
        }
        pandas_metadata["columns"].append(col_metadata)

    # Index column metadata
    if write_non_range_index_to_metadata and not write_rangeindex_to_metadata:
        if isinstance(index_type, MultiIndexType):
            pandas_metadata["column_indexes"] = [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ]
            for idx, (name_type, arr_type) in enumerate(
                zip(index_type.names_typ, index_type.array_types)
            ):
                name_placeholder = None if name_type == types.none else f"{{i{idx}}}"
                # Parquet doesn't support non-string index names
                field_name_placeholder = (
                    f"__index_level_{idx}__"
                    if name_type == types.none
                    else f"{{if{idx}}}"
                )

                index_type = array_type_to_index(arr_type, name_type)
                pandas_metadata["index_columns"].append(field_name_placeholder)
                pandas_metadata["columns"].append(
                    {
                        "name": name_placeholder,
                        "field_name": field_name_placeholder,
                        "pandas_type": index_type.pandas_type_name,
                        "numpy_type": index_type.numpy_type_name,
                        "metadata": None,
                    }
                )
        else:
            name_placeholder = None if index_type.name_typ == types.none else "{i0}"
            field_name_placeholder = (
                "__index_level_0__" if index_type.name_typ == types.none else "{if0}"
            )

            pandas_metadata["index_columns"].append(field_name_placeholder)
            pandas_metadata["columns"].append(
                {
                    "name": name_placeholder,
                    "field_name": field_name_placeholder,
                    "pandas_type": index_type.pandas_type_name,
                    "numpy_type": index_type.numpy_type_name,
                    "metadata": None,
                }
            )

    return pandas_metadata


def _apply_template(
    metadata_temp: dict[str, pt.Any],
    range_info: tuple[int, int, int] | None,
    col_names_arr,
    index_field_names,
    index_names,
):
    if range_info is not None:
        metadata_temp["index_columns"] = [
            {
                "kind": "range",
                "name": index_names[0],
                "start": range_info[0],
                "stop": range_info[1],
                "step": range_info[2],
            }
        ]

    for i in range(len(metadata_temp["index_columns"])):
        if metadata_temp["index_columns"][i] == f"{{if{i}}}":
            metadata_temp["index_columns"][i] = index_field_names[i]

    for column in metadata_temp["columns"]:
        name: str = column["name"]
        if name is None:
            continue
        elif name.startswith("{c"):
            idx = int(name[2:-1])
            column["name"] = col_names_arr[idx]
            column["field_name"] = col_names_arr[idx]
        elif name.startswith("{i"):
            idx = int(name[2:-1])
            column["name"] = index_names[idx]
            column["field_name"] = index_field_names[idx]

    return json.dumps(metadata_temp)


def gen_pandas_parquet_metadata(
    df,
    col_names_arr,
    partition_cols,
    write_non_range_index_to_metadata,
    write_rangeindex_to_metadata,
) -> tuple[str, list[str]]:  # type: ignore
    pass


@overload(gen_pandas_parquet_metadata, no_unliteral=True, jit_options={"cache": True})
def overload_gen_pandas_parquet_metadata(
    df,
    col_names_arr,
    partition_cols,
    write_non_range_index_to_metadata,
    write_rangeindex_to_metadata,
):
    df = pt.cast("DataFrameType", df)
    assert_bodo_error(is_overload_constant_bool(write_non_range_index_to_metadata))
    assert_bodo_error(is_overload_constant_bool(write_rangeindex_to_metadata))
    write_non_range_index = get_overload_const_bool(write_non_range_index_to_metadata)
    write_rangeindex = get_overload_const_bool(write_rangeindex_to_metadata)

    if not is_overload_none(partition_cols):
        partition_cols = get_overload_const_list(partition_cols)
    else:
        partition_cols = None

    # Construct the metadata template to fill at runtime
    if df.has_runtime_cols:
        # Parquet can't support multi-index in general. Support Multi-Index
        # columns once MultiIndex is supported.
        if isinstance(df.runtime_colname_typ, MultiIndexType):
            raise BodoError(
                "DataFrame.to_parquet(): Not supported with MultiIndex runtime column names. Please return the DataFrame to regular Python to update typing information."
            )
        if not isinstance(
            df.runtime_colname_typ, bodo.hiframes.pd_index_ext.StringIndexType
        ):
            # This is the Pandas error message.
            raise BodoError(
                "DataFrame.to_parquet(): parquet must have string column names. Please return the DataFrame with runtime column names to regular Python to modify column names."
            )
        # If our DataFrame has runtime columns. Then we can't generate
        # metadata str at compile time. Instead, we generate format strings
        # for each column type and fill them at runtime. We still generate
        # surrounding metadata and information for the index at compile time.
        data_columns = df.runtime_data_types
        num_col_types = len(data_columns)
        metadata_temp = gen_pandas_parquet_metadata_template(
            [f"{{c{i}}}" for i in range(num_col_types)],
            data_columns,
            df.index,
            write_non_range_index,
            write_rangeindex,
            partition_cols=partition_cols,
            is_runtime_columns=True,
        )
    else:
        metadata_temp = gen_pandas_parquet_metadata_template(
            df.columns,
            df.data,
            df.index,
            write_non_range_index,
            write_rangeindex,
            partition_cols=partition_cols,
            is_runtime_columns=False,
        )

    @bodo.wrap_python(types.Tuple((types.unicode_type, bodo.types.string_array_type)))
    def _gen_pandas_parquet_metadata_helper(
        range_info, index_names, col_names_arr, write_non_range_index_to_metadata
    ):
        # In the underlying Parquet file, for non-range index columns with no name,
        # we use the name __index_level_{idx}__ to identify the column.
        index_field_names = [
            f"__index_level_{idx}__" if name is None else name
            for idx, name in enumerate(index_names)
        ]
        metadata_str = _apply_template(
            metadata_temp, range_info, col_names_arr, index_field_names, index_names
        )
        if write_non_range_index_to_metadata:
            out_names_arr = pd.array(col_names_arr.tolist() + index_field_names)
        else:
            out_names_arr = col_names_arr

        return metadata_str, out_names_arr

    if write_rangeindex:

        def impl(
            df,
            col_names_arr,
            partition_cols,
            write_non_range_index_to_metadata,
            write_rangeindex_to_metadata,
        ):
            index = df.index
            index_names = index.names

            # Add range_info to metadata,
            # This is the only line that is different between the two impls.
            range_info = (index.start, index.stop, index.step)

            return _gen_pandas_parquet_metadata_helper(
                range_info,
                index_names,
                col_names_arr,
                write_non_range_index_to_metadata,
            )
    else:

        def impl(
            df,
            col_names_arr,
            partition_cols,
            write_non_range_index_to_metadata,
            write_rangeindex_to_metadata,
        ):
            index = df.index
            index_names = index.names

            range_info = None

            return _gen_pandas_parquet_metadata_helper(
                range_info,
                index_names,
                col_names_arr,
                write_non_range_index_to_metadata,
            )

    return impl
