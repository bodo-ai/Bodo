from __future__ import annotations

import datetime
import os
import re
import time
import traceback
import warnings
from enum import Enum
from typing import Any

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import ir, types

import bodo
from bodo.ir.sql_ext import parse_dbtype
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError, dtype_to_array_type
from bodosql.bodosql_types.database_catalog import DatabaseCatalog
from bodosql.bodosql_types.table_path import TablePath, TablePathType
from bodosql.imported_java_classes import (
    JavaEntryPoint,
    build_java_array_list,
    build_java_hash_map,
)
from bodosql.utils import BodoSQLWarning, error_to_string

# Prefix to add to table argument names when passed to JIT to avoid variable name conflicts
TABLE_ARG_PREFIX = "_ARG_"
# Prefix to add to bind variable argument names when passed to JIT to avoid variable name conflicts
DYNAMIC_PARAM_ARG_PREFIX = "_DYNAMIC_PARAM_"
# Prefix to add to named parameter argument names when passed to JIT to avoid variable name conflicts
NAMED_PARAM_ARG_PREFIX = "_NAMED_PARAM_"


# NOTE: These are defined in BodoSQLColumnDataType and must match here
class SqlTypeEnum(Enum):
    Null = 0
    Int8 = 1
    Int16 = 2
    Int32 = 3
    Int64 = 4
    UInt8 = 5
    UInt16 = 6
    UInt32 = 7
    UInt64 = 8
    Float32 = 9
    Float64 = 10
    Decimal = 11
    Bool = 12
    Date = 13
    Time = 14
    Timestamp_Ntz = 15
    Timestamp_Ltz = 16
    Timestamp_Tz = 17
    Timedelta = 18
    DateOffset = 19
    String = 20
    Binary = 21
    Categorical = 22
    # Note Array, Object, Struct, and Variant are currently unused
    # on the Python side but this enum is updated to be consistent.
    Array = 23
    Json_Object = 24
    Struct = 25
    Variant = 26
    # Fixed Size columns are for columns with a compile time known size.
    # These are only used for special Iceberg types but are added here for
    # consistency.
    Fixed_Size_String = 27
    Fixed_Size_Binary = 28
    Unsupported = 29


# Scalar dtypes for supported Bodo Arrays
_numba_to_sql_column_type_map = {
    bodo.null_dtype: SqlTypeEnum.Null.value,
    types.int8: SqlTypeEnum.Int8.value,
    types.uint8: SqlTypeEnum.UInt8.value,
    types.int16: SqlTypeEnum.Int16.value,
    types.uint16: SqlTypeEnum.UInt16.value,
    types.int32: SqlTypeEnum.Int32.value,
    types.uint32: SqlTypeEnum.UInt32.value,
    types.int64: SqlTypeEnum.Int64.value,
    types.uint64: SqlTypeEnum.UInt64.value,
    types.float32: SqlTypeEnum.Float32.value,
    types.float64: SqlTypeEnum.Float64.value,
    types.NPDatetime("ns"): SqlTypeEnum.Timestamp_Ntz.value,
    types.NPTimedelta("ns"): SqlTypeEnum.Timedelta.value,
    types.bool_: SqlTypeEnum.Bool.value,
    bodo.string_type: SqlTypeEnum.String.value,
    bodo.bytes_type: SqlTypeEnum.Binary.value,
    # Note date doesn't have native support yet, but the code to
    # cast to datetime64 is handled in the Java code.
    bodo.datetime_date_type: SqlTypeEnum.Date.value,
    bodo.timestamptz_type: SqlTypeEnum.Timestamp_Tz.value,
}

# Scalar dtypes for supported parameters
_numba_to_sql_param_type_map = {
    types.none: SqlTypeEnum.Null.value,
    types.int8: SqlTypeEnum.Int8.value,
    types.uint8: SqlTypeEnum.UInt8.value,
    types.int16: SqlTypeEnum.Int16.value,
    types.uint16: SqlTypeEnum.UInt16.value,
    types.int32: SqlTypeEnum.Int32.value,
    types.uint32: SqlTypeEnum.UInt32.value,
    types.int64: SqlTypeEnum.Int64.value,
    types.uint64: SqlTypeEnum.UInt64.value,
    types.float32: SqlTypeEnum.Float32.value,
    types.float64: SqlTypeEnum.Float64.value,
    types.bool_: SqlTypeEnum.Bool.value,
    bodo.string_type: SqlTypeEnum.String.value,
    # Scalar datetime and timedelta are assumed
    # to be scalar Pandas Timestamp/Timedelta
    bodo.pd_timestamp_tz_naive_type: SqlTypeEnum.Timestamp_Ntz.value,
    bodo.timestamptz_type: SqlTypeEnum.Timestamp_Tz.value,
    # TODO: Support Date and Binary parameters [https://bodo.atlassian.net/browse/BE-3542]
}


def construct_tz_aware_array_type(typ, nullable):
    """Construct a BodoSQL data type for a tz-aware timestamp array

    Args:
        typ (types.Type): A tz-aware Bodo type
        nullable (bool): Is the column Nullable

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    # Timestamps only support precision 9 right now.
    precision = 9
    if typ.tz is None:
        # TZ = None is a timezone naive timestamp
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Ntz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, precision)
    else:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Ltz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, precision)


def construct_time_array_type(typ: bodo.TimeArrayType | bodo.TimeType, nullable: bool):
    """Construct a BodoSQL data type for a time array.

    Args:
        typ (Union[bodo.TimeArrayType, bodo.TimeType]): A time Bodo type
        nullable (bool): Is the column Nullable

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
        SqlTypeEnum.Time.value
    )
    return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, typ.precision)


def construct_array_item_array_type(arr_type):
    """Construct a BodoSQL data type for an array item array
    value.

    Args:
        typ (bodo.ArrayItemArrayType): A ArrayItemArray type
        col_name (str): Column name

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    child = get_sql_data_type(arr_type.dtype)
    type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
        SqlTypeEnum.Array.value
    )
    return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, child)


def construct_json_array_type(arr_type):
    """Construct a BodoSQL data type for a JSON array
    value.

    Args:
        typ (bodo.StructArrayType or bodo.MapArrayType): A StructArray or MapArray type
        col_name (str): Column name

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    if isinstance(arr_type, bodo.StructArrayType):
        # TODO: FIXME. We don't support full structs of types yet.
        # As a placeholder we will just match Snowflake.
        key_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.String.value
        )
        key = JavaEntryPoint.buildColumnDataTypeInfo(key_enum, True)
        value_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Variant.value
        )
        value = JavaEntryPoint.buildColumnDataTypeInfo(value_enum, True)
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Json_Object.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, key, value)
    else:
        # TODO: Add map scalar support
        key = get_sql_data_type(arr_type.key_arr_type)
        value = get_sql_data_type(arr_type.value_arr_type)
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Json_Object.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, key, value)


def get_sql_column_type(arr_type, col_name):
    data_type = get_sql_data_type(arr_type)
    return JavaEntryPoint.buildBodoSQLColumnImpl(col_name, data_type)


def get_sql_data_type(arr_type):
    """get SQL type for a given array type."""
    warning_msg = f"Encountered type {arr_type} which is not supported in BodoSQL. BodoSQL will attempt to optimize the query to remove this column, but this can lead to errors in compilation. Please refer to the supported types: https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    # We currently treat NaT as nullable in BodoSQL, so for any array that has timestamp elements
    # type, we treat it as nullable.
    dtype_has_nullable = arr_type.dtype in (
        bodo.datetime64ns,
        bodo.timedelta64ns,
    )
    nullable = dtype_has_nullable or bodo.utils.typing.is_nullable_type(arr_type)
    if isinstance(arr_type, bodo.DatetimeArrayType):
        # Timezone-aware Timestamp columns have their own special handling.
        return construct_tz_aware_array_type(arr_type, nullable)
    elif arr_type == bodo.timestamptz_array_type:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Tz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    elif isinstance(arr_type, bodo.TimeArrayType):
        # Time array types have their own special handling for precision
        return construct_time_array_type(arr_type, nullable)
    elif isinstance(arr_type, bodo.DecimalArrayType):
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Decimal.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(
            type_enum, nullable, arr_type.precision, arr_type.scale
        )
    elif isinstance(arr_type, bodo.ArrayItemArrayType):
        return construct_array_item_array_type(arr_type)
    elif isinstance(arr_type, (bodo.StructArrayType, bodo.MapArrayType)):
        return construct_json_array_type(arr_type)
    elif arr_type.dtype in _numba_to_sql_column_type_map:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            _numba_to_sql_column_type_map[arr_type.dtype]
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    elif isinstance(arr_type.dtype, bodo.PDCategoricalDtype):
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Categorical.value
        )
        child = get_sql_data_type(dtype_to_array_type(arr_type.dtype.elem_type, True))
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, child)
    else:
        # The type is unsupported we raise a warning indicating this is a possible
        # error but we generate a dummy type because we may be able to support it
        # if its optimized out.
        warnings.warn(BodoSQLWarning(warning_msg))
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Unsupported.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)


def create_java_dynamic_parameter_type_list(dynamic_params_list: list[Any]):
    """Convert a list of dynamic parameters or dynamic parameter types
    into a Java List of ColumnDataType values.

    Args:
        dynamic_params_list (List[Any]): The input list, either a Bodo type
        or a Python value to convert to a java type.

    Returns:
        JavaObject: A java array to pass to code generation.
    """
    types_list = []
    for val in dynamic_params_list:
        typ = val if isinstance(val, types.Type) else bodo.typeof(val)
        types_list.append(get_sql_param_column_type_info(typ))
    return build_java_array_list(types_list)


def create_java_named_parameter_type_map(named_params: dict[str, Any]):
    """Convert a list of keys and list of values into a Java
    Map from key to ColumnDataType values.

    Args:
        dynamic_params_list (List[Any]): The input list, either a Bodo type
        or a Python value to convert to a java type.

    Returns:
        JavaObject: A java map to pass to code generation.
    """
    d = {
        key: get_sql_param_column_type_info(
            val if isinstance(val, types.Type) else bodo.typeof(val)
        )
        for key, val in named_params.items()
    }
    return build_java_hash_map(d)


def get_sql_param_column_type_info(param_type: types.Type):
    """Get the SQL type information for a given Dynamic
    parameter type.

    Args:
        param_type (types.Type): The bodo type to lower as a parameter.
    Return:
        JavaObject: The ColumnDataTypeInfo for the parameter type.
    """
    unliteral_type = types.unliteral(param_type)
    # The named parameters are always scalars. We don't support
    # Optional types or None types yet. As a result this is always
    # non-null.
    nullable = False
    if (
        isinstance(unliteral_type, bodo.PandasTimestampType)
        and unliteral_type.tz != None
    ):
        return construct_tz_aware_array_type(param_type, nullable)
    elif isinstance(unliteral_type, bodo.TimeType):
        # Time array types have their own special handling for precision
        return construct_time_array_type(param_type, nullable)
    elif isinstance(unliteral_type, bodo.Decimal128Type):
        # Decimal types need handling for precision and scale.
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Decimal.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(
            type_enum, nullable, unliteral_type.precision, unliteral_type.scale
        )
    elif unliteral_type in _numba_to_sql_param_type_map:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            _numba_to_sql_param_type_map[unliteral_type]
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    raise TypeError(
        f"Dynamic Parameter with type {param_type} not supported in BodoSQL. Please cast your data to a supported type. https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    )


def compute_df_types(df_list, is_bodo_type):
    """Given a list of Bodo types or Python objects,
    determines the DataFrame type for each object. This
    is used by both Python and JIT, where Python converts to
    Bodo types via the is_bodo_type argument. This function
    converts any TablePathType to the actual DataFrame type,
    which must be done in parallel.

    Args:
        df_list (List[types.Type | pd.DataFrame | bodosql.TablePath]):
            List of table either from Python or JIT.
        is_bodo_type (bool): Is this being called from JIT? If so we
            don't need to get the type of each member of df_list

    Raises:
        BodoError: If a TablePathType is passed with invalid
            values we raise an exception.

    Returns:
        Tuple(orig_bodo_types, df_types): Returns the Bodo types and
            the bodo.DataFrameType for each table. The original bodo
            types are kept to determine when code needs to be generated
            for TablePathType
    """

    orig_bodo_types = []
    df_types = []
    for df_val in df_list:
        if is_bodo_type:
            typ = df_val
        else:
            typ = bodo.typeof(df_val)
        orig_bodo_types.append(typ)

        if isinstance(typ, TablePathType):
            table_info = typ
            file_type = table_info._file_type
            file_path = table_info._file_path
            if file_type == "pq":
                # Extract the parquet information using Bodo
                type_info = bodo.io.parquet_pio.parquet_file_schema(file_path, None)
                # Future proof against additional return values that are unused
                # by BodoSQL by returning a tuple.
                col_names = type_info[0]
                col_types = type_info[1]
                index_col = type_info[2]
                # If index_col is not a column name, we use a range type
                if index_col is None or isinstance(index_col, dict):
                    if isinstance(index_col, dict) and index_col["name"] is not None:
                        index_col_name = types.StringLiteral(index_col["name"])
                    else:
                        index_col_name = None
                    index_typ = bodo.RangeIndexType(index_col_name)

                # Otherwise the index is a specific column
                else:
                    # if the index_col is __index_level_0_, it means it has no name.
                    # Thus we do not write the name instead of writing '__index_level_0_' as the name
                    if "__index_level_" in index_col:
                        index_name = None
                    else:
                        index_name = index_col
                    # Convert the column type to an index type
                    index_loc = col_names.index(index_col)
                    index_elem_dtype = col_types[index_loc].dtype

                    index_typ = bodo.utils.typing.index_typ_from_dtype_name_arr(
                        index_elem_dtype, index_name, col_types[index_loc]
                    )

                    # Remove the index from the DataFrame.
                    col_names.pop(index_loc)
                    col_types.pop(index_loc)
            elif file_type == "sql":
                const_conn_str = table_info._conn_str
                db_type, _ = parse_dbtype(const_conn_str)
                if db_type == "iceberg":
                    db_schema = table_info._db_schema
                    iceberg_table_name = table_info._file_path
                    # table_name = table_info.
                    type_info = bodo.transforms.untyped_pass
                    # schema = table_info._schema
                    (
                        col_names,
                        col_types,
                        _pyarrow_table_schema,
                    ) = bodo.io.iceberg.get_iceberg_orig_schema(
                        const_conn_str,
                        f"{db_schema}.{iceberg_table_name}",
                    )
                else:
                    type_info = (
                        bodo.transforms.untyped_pass._get_sql_types_arr_colnames(
                            f"{file_path}",
                            const_conn_str,
                            # _bodo_read_as_dict
                            None,
                            ir.Var(None, "dummy_var", ir.Loc("dummy_loc", -1)),
                            ir.Loc("dummy_loc", -1),
                            # is_table_input
                            True,
                            False,
                            # downcast_decimal_to_double
                            False,
                            convert_snowflake_column_names=False,
                        )
                    )
                    # Future proof against additional return values that are unused
                    # by BodoSQL by returning a tuple.
                    col_names = type_info[1]
                    col_types = type_info[3]

                # Generate the index type. We don't support an index column,
                # so this is always a RangeIndex.
                index_typ = bodo.RangeIndexType(None)
            else:
                raise BodoError(
                    "Internal error, 'compute_df_types' found a TablePath with an invalid file type"
                )

            # Generate the DataFrame type
            df_type = bodo.DataFrameType(
                tuple(col_types),
                index_typ,
                tuple(col_names),
            )
        else:
            df_type = typ
        df_types.append(df_type)
    return orig_bodo_types, df_types


def add_table_type(
    table_name: str,
    schema,
    df_type: bodo.DataFrameType,
    estimated_row_count: int | None,
    estimated_ndvs: dict[str, int] | None,
    bodo_type: types.Type,
    table_num: int,
    from_jit: bool,
    write_type: str,
):
    """Registers a new table into the schema. This is used to pass tables via DataFrames or the
    TablePath API.

    Args:
        table_name (str): The name of the table.
        schema (Java LocalSchema): The schema to update.
        df_type (bodo.DataFrameType): The Bodo DataFrame type.
        estimated_row_count (Optional[int]): The expected number of rows in the table for the
            Volcano Planner. None if no estimate is provided.
        estimated_ndvs (Optional[dict[str, int]]): Estimated NDV values for the columns. This
            maps the column names to the NDV estimate. Providing some and not all column NDVs
            is supported. None if no estimate is provided.
        bodo_type (types.Type): Bodo type for the table. This stores the original type so a TablePath
            isn't converted to its DataFrameType, which the df_type always is.
        table_num (int): ID for the table being processed.
        from_jit (bool): Is this typing coming from JIT?
        write_type (str): String describing the type of write used for generating the write code.
            Will be "MERGE" for MERGE INTO queries, and defaults to "INSERT" for all other
            queries.
    """
    assert bodo.get_rank() == 0, "add_table_type should only be called on rank 0."
    sql_types = [
        get_sql_column_type(df_type.data[i], cname)
        for i, cname in enumerate(df_type.columns)
    ]
    col_arr = build_java_array_list(sql_types)

    # To support writing to SQL Databases we register is_writeable
    # for SQL databases.
    is_writeable = (
        isinstance(bodo_type, TablePathType) and bodo_type._file_type == "sql"
    )

    if is_writeable:
        schema_code_to_sql = (
            f"schema='{bodo_type._db_schema}'"
            if bodo_type._db_schema is not None
            else ""
        )
        if write_type == "MERGE":
            # Note. We only support MERGE for Iceberg. We check this in the
            # Java code to ensure we also handle catalogs. Note the
            # last argument is for passing additional arguments as key=value pairs.
            write_format_code = f"bodo.io.iceberg.merge_into.iceberg_merge_cow_py('{bodo_type._file_path}', '{bodo_type._conn_str}', '{bodo_type._db_schema}', %s, %s)"
        else:
            write_format_code = f"%s.to_sql('{bodo_type._file_path}', '{bodo_type._conn_str}', if_exists='append', index=False, {schema_code_to_sql}, %s)"
    else:
        write_format_code = ""

    # Determine the DB Type for generating java code.
    if isinstance(bodo_type, TablePathType):
        if bodo_type._file_type == "pq":
            db_type = "PARQUET"
        else:
            assert bodo_type._file_type == "sql", (
                "TablePathType is only implement for parquet and SQL APIs"
            )
            const_conn_str = bodo_type._conn_str
            db_type, _ = parse_dbtype(const_conn_str)
    else:
        db_type = "MEMORY"

    read_code = _generate_table_read(table_name, bodo_type, table_num, from_jit)

    # Convert the Python dict to a Java HashMap:
    estimated_ndvs = {} if estimated_ndvs is None else estimated_ndvs
    estimated_ndvs_java_map = build_java_hash_map(estimated_ndvs)

    table = JavaEntryPoint.buildLocalTable(
        table_name,
        schema,
        col_arr,
        is_writeable,
        read_code,
        write_format_code,
        # TablePath is a wrapper for a file so it results in an IO read.
        # The only other option is an in memory Pandas DataFrame.
        isinstance(bodo_type, TablePathType),
        db_type,
        estimated_row_count,
        estimated_ndvs_java_map,
    )
    JavaEntryPoint.addTableToSchema(schema, table)


def _get_estimated_row_count(table: pd.DataFrame | TablePath) -> int | None:
    if isinstance(table, pd.DataFrame):
        # TODO: Handle distributed inputs.
        # Generate lengths if known.
        return len(table)
    elif isinstance(table, TablePath):
        return table.estimated_row_count
    else:
        # Pass None for unknown lengths.
        # TODO: Support other inputs types
        return None


def _get_estimated_ndv(table: pd.DataFrame | TablePath) -> dict[str, int]:
    if isinstance(table, TablePath):
        return table._statistics.get("ndv", {})
    else:
        return {}


def _generate_table_read(
    table_name: str,
    bodo_type: types.Type,
    table_num: int,
    from_jit: bool,
) -> str:
    """Generates the read code for a table to pass to Java.

    Args:
        table_name (str): Name of the table
        bodo_type (types.Type): Bodo Type of the table. If this is
            a TablePath different code is generated.
        table_num (int): What number table is being processed.
        from_jit (bool): Is the code being generated from JIT?

    Raises:
        BodoError: If code generation is not supported for the given type.

    Returns:
        str: A string that is the generated code for a read expression.
    """
    if isinstance(bodo_type, TablePathType):
        file_type = bodo_type._file_type
        file_path = bodo_type._file_path
        # Escape "\" in Windows paths
        file_path = file_path.replace("\\", "\\\\")

        read_dict_list = (
            ""
            if bodo_type._bodo_read_as_dict is None
            else f"_bodo_read_as_dict={bodo_type._bodo_read_as_dict}"
        )
        if file_type == "pq":
            # TODO: Replace with runtime variable once we support specifying
            # the schema
            if read_dict_list:
                read_line = f"pd.read_parquet('{file_path}', {read_dict_list}, _bodo_use_index=False, _bodo_read_as_table=True, %s)"
            else:
                read_line = f"pd.read_parquet('{file_path}', _bodo_use_index=False, _bodo_read_as_table=True, %s)"
        elif file_type == "sql":
            # TODO: Replace with runtime variable once we support specifying
            # the schema
            conn_str = bodo_type._conn_str
            db_type, _ = parse_dbtype(conn_str)
            if db_type == "iceberg":
                # Avoid errors for Windows path backslashes in generated code later
                conn_str = conn_str.replace("\\", "/")
                if read_dict_list:
                    read_line = f"pd.read_sql_table('{file_path}', '{conn_str}', '{bodo_type._db_schema}', {read_dict_list}, _bodo_read_as_table=True, %s)"
                else:
                    read_line = f"pd.read_sql_table('{file_path}', '{conn_str}', '{bodo_type._db_schema}', _bodo_read_as_table=True, %s)"
            else:
                read_line = f"pd.read_sql('select * from {file_path}', '{conn_str}', _bodo_read_as_table=True, %s)"
        else:
            raise BodoError(
                f"Internal Error: Unsupported TablePathType for type: '{file_type}'"
            )
    elif from_jit:
        read_line = f"bodo_sql_context.dataframes[{table_num}]"
    else:
        read_line = TABLE_ARG_PREFIX + table_name
    return read_line


class BodoSQLContext:
    def __init__(self, tables=None, catalog=None, default_tz=None):
        # We only need to initialize the tables values on all ranks, since that is needed for
        # creating the JIT function on all ranks for bc.sql calls. We also initialize df_types on all ranks,
        # for consistency. All the other attributes
        # are only used for generating the func text, which is only done on rank 0.
        if tables is None:
            tables = {}

        self.tables = tables
        self.default_tz = default_tz
        # Check types
        if any(not isinstance(key, str) for key in self.tables.keys()):
            raise BodoError("BodoSQLContext(): 'table' keys must be strings")
        if any(
            not isinstance(value, (pd.DataFrame, TablePath))
            for value in self.tables.values()
        ):
            raise BodoError(
                "BodoSQLContext(): 'table' values must be DataFrames or TablePaths"
            )

        if not (catalog is None or isinstance(catalog, DatabaseCatalog)):
            raise BodoError(
                "BodoSQLContext(): 'catalog' must be a bodosql.DatabaseCatalog if provided"
            )
        self.catalog = catalog

        # This except block can run in the case that our iceberg connector raises an error
        failed = False
        msg = ""
        try:
            # Convert to a dictionary mapping name -> type. For consistency
            # we first unpack the dictionary.
            names = []
            dfs = []
            estimated_row_counts = []
            estimated_ndvs = []
            for k, v in tables.items():
                names.append(k)
                dfs.append(v)
                estimated_row_counts.append(_get_estimated_row_count(v))
                estimated_ndvs.append(_get_estimated_ndv(v))
            orig_bodo_types, df_types = compute_df_types(dfs, False)
            schema = initialize_schema()
            self.schema = schema
            self.names = names
            self.df_types = df_types
            self.orig_bodo_types = orig_bodo_types
            self.estimated_row_counts = estimated_row_counts
            self.estimated_ndvs = estimated_ndvs
        except Exception as e:
            failed = True
            msg = error_to_string(e)

        failed = bcast_scalar(failed)
        msg = bcast_scalar(msg)
        if failed:
            raise BodoError(msg)

    def __getstate__(self) -> object:
        """
        Returns a state object used during pickling.
        """
        # 'schema' is a Java Object which cannot be pickled, so we
        # remove it from the state. We will re-initialize it during
        # unpickling (see __setstate__).
        dict_cp = self.__dict__.copy()
        dict_cp.pop("schema")
        return dict_cp

    def __setstate__(self, state):
        """
        Inverse of __getstate__ where we modify this
        object using the provided state.
        """
        # Set the state and initialize the Java objects from scratch.
        self.__dict__ = state
        self.schema = initialize_schema()

    def validate_query_compiles(self, sql, params_dict=None, dynamic_params_list=None):
        """
        Verifies BodoSQL can fully compile the query in Bodo.
        """
        try:
            t1 = time.time()
            self._compile(sql, params_dict, dynamic_params_list)
            compile_time = time.time() - t1
            compiles_flag = True
            error_message = "No error"
        except Exception as e:
            stack_trace = traceback.format_exc()
            compile_time = time.time() - t1
            compiles_flag = False
            error_message = repr(e)
            if os.environ.get("NUMBA_DEVELOPER_MODE", False):
                error_message = error_message + "\n" + stack_trace

        return compiles_flag, compile_time, error_message

    def _compile(self, sql, params_dict=None, dynamic_params_list=None):
        """compiles the query in Bodo."""
        import bodosql

        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            False,  # We need to execute the code so don't hide credentials.
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        is_ddl = bcast_scalar(is_ddl)
        if is_ddl:
            warning_msg = "Encountered a DDL query. These queries are executed directly by bc.sql() so this wont't properly test compilation."
            warnings.warn(BodoSQLWarning(warning_msg))
        func_text, lowered_globals = self._convert_to_pandas(
            sql,
            dynamic_params_list,
            params_dict,
            generator,
            is_ddl,
        )

        glbls = {
            "np": np,
            "pd": pd,
            "bodosql": bodosql,
            "re": re,
            "bodo": bodo,
            "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
            "MetaType": bodo.utils.typing.MetaType,
            "numba": numba,
            "time": time,
            "datetime": datetime,
            "bif": bodo.ir.filter,
        }

        glbls.update(lowered_globals)
        return self._functext_compile(
            func_text, dynamic_params_list, params_dict, glbls
        )

    def _functext_compile(self, func_text, dynamic_params_list, params_dict, glbls):
        """
        Helper function for _compile, that compiles the function text.
        This is mostly separated out for testing purposes.
        """

        arg_types = []
        for table_arg in self.tables.values():
            arg_types.append(bodo.typeof(table_arg))
        for dynamic_param_arg in dynamic_params_list:
            arg_types.append(bodo.typeof(dynamic_param_arg))
        for param_arg in params_dict.values():
            arg_types.append(bodo.typeof(param_arg))

        sig = tuple(arg_types)

        loc_vars = {}
        exec(
            func_text,
            glbls,
            loc_vars,
        )
        impl = loc_vars["impl"]

        dispatcher = bodo.jit(sig)(impl)
        return dispatcher

    def validate_query(self, sql):
        """
        Verifies BodoSQL can compute query,
        but does not actually compile the query in Bodo.
        """
        try:
            self.convert_to_pandas(sql)
            executable_flag = True
        except Exception:
            executable_flag = False

        return executable_flag

    def convert_to_pandas(
        self, sql, params_dict=None, dynamic_params_list=None, hide_credentials=True
    ):
        """converts SQL code to Pandas"""
        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            hide_credentials,
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        is_ddl = bcast_scalar(is_ddl)
        if is_ddl:
            warning_msg = "Encountered a DDL query. These queries are executed directly by bc.sql() so this wont't properly represent generated code."
            warnings.warn(BodoSQLWarning(warning_msg))
        pd_code, lowered_globals = self._convert_to_pandas(
            sql,
            dynamic_params_list,
            params_dict,
            generator,
            is_ddl,
        )
        # add the imports so someone can directly run the code.
        imports = [
            "import numpy as np",
            "import pandas as pd",
            "import time",
            "import datetime",
            "import numba",
            "import bodo",
            "import bodosql",
            "from bodo.utils.typing import ColNamesMetaType",
            "from bodo.utils.typing import MetaType",
            "import bodo.ir.filter as bif",
        ]
        added_globals = []
        # Add a decorator so someone can directly run the code.
        decorator = "@bodo.jit\n"
        # Add the global variable definitions at the beginning of the fn,
        # for better readability
        for varname, glbl in lowered_globals.items():
            added_globals.append(varname + " = " + repr(glbl))

        return (
            "\n".join(imports)
            + "\n"
            + "\n".join(added_globals)
            + "\n"
            + decorator
            + pd_code
        )

    def _create_planner_and_parse_query(self, sql: str, hide_credentials: bool):
        from bodo.mpi4py import MPI

        comm = MPI.COMM_WORLD

        plan_generator = None
        error_message = None
        if bodo.get_rank() == 0:
            plan_generator = self._create_generator(hide_credentials)
            try:
                if sql.strip() == "":
                    bodo.utils.typing.raise_bodo_error(
                        "BodoSQLContext passed empty query string"
                    )
                JavaEntryPoint.parseQuery(plan_generator, sql)
                # Write type is used for the current Merge Into code path decisions.
                # This should be removed when we revisit Merge Into
                write_type = JavaEntryPoint.getWriteType(plan_generator, sql)
                update_schema(
                    self.schema,
                    self.names,
                    self.df_types,
                    self.estimated_row_counts,
                    self.estimated_ndvs,
                    self.orig_bodo_types,
                    False,
                    write_type,
                )
            except Exception as e:
                error_message = error_to_string(e)

        error_message = comm.bcast(error_message)
        if error_message is not None:
            raise BodoError(
                f"Unable to parse SQL Query. Error message:\n{error_message}"
            )
        return plan_generator

    def _convert_to_pandas(
        self,
        sql: str,
        dynamic_params_list: list[Any],
        named_params_dict: dict[str, Any],
        generator,
        is_ddl: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Generate the func_text for the Python code generated for the given SQL query.
        This is always computed entirely on rank 0 to avoid parallelism errors.

        Args:
            sql (str): The SQL query to process.
            dynamic_params_list (List[Any]): The list of dynamic parameters to lower.
            named_params_dict (Dict[str, Any]): The named parameters to lower.
            generator (RelationalAlgebraGenerator Java Object): The relational algebra generator
                used to generate the code.
            is_ddl (bool): Is this a DDL query?
        Raises:
            BodoError: If the SQL query cannot be processed.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        from bodo.mpi4py import MPI

        comm = MPI.COMM_WORLD
        func_text_or_err_msg = ""
        failed = False
        globalsToLower = ()
        if bodo.get_rank() == 0:
            # This try block should never run under normal circumstances,
            # but it's nice to have for debugging purposes so things don't hang
            # if we make any changes that could lead to a runtime error.
            try:
                # Generate the code
                pd_code, globalsToLower = self._get_pandas_code(
                    sql, generator, dynamic_params_list, named_params_dict
                )
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [(str(k), str(v)) for k, v in globalsToLower.items()]
                )
                # Hard code the context name for DDL execution. This is used
                # for compilation testing and JIT code generation.
                context_names = ["bodo_sql_context"] if is_ddl else []
                table_names = [TABLE_ARG_PREFIX + x for x in self.tables.keys()]
                dynamic_param_names = [
                    DYNAMIC_PARAM_ARG_PREFIX + str(i)
                    for i in range(len(dynamic_params_list))
                ]
                named_param_names = [
                    NAMED_PARAM_ARG_PREFIX + x for x in named_params_dict.keys()
                ]
                args = ", ".join(
                    context_names
                    + table_names
                    + dynamic_param_names
                    + named_param_names
                )
                func_text_or_err_msg += f"def impl({args}):\n"
                func_text_or_err_msg += f"{pd_code}\n"
            except Exception as e:
                failed = True
                func_text_or_err_msg = error_to_string(e)

        failed = bcast_scalar(failed)
        func_text_or_err_msg = bcast_scalar(func_text_or_err_msg)
        if failed:
            raise BodoError(func_text_or_err_msg)

        globalsToLower = comm.bcast(globalsToLower)
        globalsDict = {}
        # convert the global map list of tuples of string varname and string value, to a map of string varname -> python value.
        for varname, str_value in globalsToLower:
            locs = {}
            exec(
                f"value = {str_value}",
                {
                    "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
                    "MetaType": bodo.utils.typing.MetaType,
                    "bodo": bodo,
                    "numba": numba,
                    "time": time,
                    "pd": pd,
                    "datetime": datetime,
                    "bif": bodo.ir.filter,
                    "np": np,
                },
                locs,
            )
            globalsDict[varname] = locs["value"]
        return func_text_or_err_msg, globalsDict

    def sql(self, sql, params_dict=None, dynamic_params_list=None, **jit_options):
        import bodosql
        from bodo.spawn.spawner import SpawnDispatcher

        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            False,  # We need to execute the code so don't hide credentials.
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        is_ddl = bcast_scalar(is_ddl)
        if is_ddl:
            # Just execute DDL operations directly and return the DataFrame.
            return self.execute_ddl(sql, generator)
        else:
            func_text, lowered_globals = self._convert_to_pandas(
                sql,
                dynamic_params_list,
                params_dict,
                generator,
                False,  # This path is never DDL.s
            )
            glbls = {
                "np": np,
                "pd": pd,
                "bodosql": bodosql,
                "re": re,
                "bodo": bodo,
                "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
                "MetaType": bodo.utils.typing.MetaType,
                "numba": numba,
                "time": time,
                "datetime": datetime,
                "bif": bodo.ir.filter,
            }

            glbls.update(lowered_globals)
            loc_vars = {}
            exec(
                func_text,
                glbls,
                loc_vars,
            )
            impl = loc_vars["impl"]

            # Add table argument name prefix to user provided distributed flags to match
            # stored names
            if "distributed" in jit_options and isinstance(
                jit_options["distributed"], (list, set)
            ):
                jit_options["distributed"] = [
                    TABLE_ARG_PREFIX + x for x in jit_options["distributed"]
                ]
            if "replicated" in jit_options and isinstance(
                jit_options["replicated"], (list, set)
            ):
                jit_options["replicated"] = [
                    TABLE_ARG_PREFIX + x for x in jit_options["replicated"]
                ]

            dispatcher = bodo.jit(impl, **jit_options)

            # Save BodoSQL globals in SpawnDispatcher to be handled in pickling
            # properly. Internal CASE implementation strings may use some globals that
            # are not visible to cloudpickle as used by the function. See:
            # test_json_fns.py::test_object_construct_keep_null[no_nested-no_null-with_case]
            if isinstance(dispatcher, SpawnDispatcher):
                # __builtins__ which is added to glbls by exec causes issues in
                # Jupyter/IPython pickling.
                glbls.pop("__builtins__", None)
                dispatcher.add_extra_globals(glbls)

            return dispatcher(
                *(
                    list(self.tables.values())
                    + dynamic_params_list
                    + list(params_dict.values())
                )
            )

    def generate_plan(
        self, sql, params_dict=None, dynamic_params_list=None, show_cost=False
    ):
        """
        Return the optimized plan for the SQL code as
        as a Python string.
        """
        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(sql, True)
        failed = False
        plan_or_err_msg = ""
        if bodo.get_rank() == 0:
            try:
                java_params_array = create_java_dynamic_parameter_type_list(
                    dynamic_params_list
                )
                java_named_params_map = create_java_named_parameter_type_map(
                    params_dict
                )
                plan_or_err_msg = str(
                    JavaEntryPoint.getOptimizedPlanString(
                        generator,
                        sql,
                        show_cost,
                        java_params_array,
                        java_named_params_map,
                    )
                )
            except Exception as e:
                failed = True
                plan_or_err_msg = error_to_string(e)

        failed = bcast_scalar(failed)
        plan_or_err_msg = bcast_scalar(plan_or_err_msg)
        if failed:
            raise BodoError(plan_or_err_msg)
        return plan_or_err_msg

    def _get_pandas_code(
        self,
        sql: str,
        generator,
        dynamic_params_list: list[Any],
        named_params_dict: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Generate the Pandas code for the given SQL string.

        Args:
            sql (str): The SQL query text.
            generator (RelationalAlgebraGenerator Java Object): The relational algebra generator
                used to generate the code.

        Raises:
            bodo.utils.typing.BodoError: The SQL text is not supported.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        # Construct the relational algebra generator
        try:
            java_params_array = create_java_dynamic_parameter_type_list(
                dynamic_params_list
            )
            java_named_params_map = create_java_named_parameter_type_map(
                named_params_dict
            )
            pd_code = str(
                JavaEntryPoint.getPandasString(
                    generator, sql, java_params_array, java_named_params_map
                )
            )
            failed = False
        except Exception as e:
            message = error_to_string(e)
            failed = True
        if failed:
            # Raise BodoError outside except to avoid stack trace
            raise bodo.utils.typing.BodoError(
                f"Unable to compile SQL Query. Error message:\n{message}"
            )
        return pd_code, JavaEntryPoint.getLoweredGlobals(generator)

    def _create_generator(self, hide_credentials: bool):
        """Creates a RelationalAlgebraGenerator from the schema.

        Args:
            hide_credentials (bool): Should credentials be hidden for
                any generated code.

        Returns:
            RelationalAlgebraGenerator Java Object: The java object holding
                the relational algebra generator.
        """
        verbose_level = bodo.user_logging.get_verbose_level()
        tracing_level = bodo.tracing_level
        if self.catalog is not None:
            catalog_obj = self.catalog.get_java_object()
        else:
            catalog_obj = None
        return JavaEntryPoint.buildRelationalAlgebraGenerator(
            catalog_obj,
            self.schema,
            bodo.bodosql_use_streaming_plan,
            verbose_level,
            tracing_level,
            bodo.bodosql_streaming_batch_size,
            hide_credentials,
            bodo.enable_snowflake_iceberg,
            bodo.enable_timestamp_tz,
            bodo.enable_streaming_sort,
            bodo.enable_streaming_sort_limit_offset,
            bodo.bodo_sql_style,
            bodo.bodosql_full_caching,
            bodo.prefetch_sf_iceberg,
            self.default_tz,
        )

    def add_or_replace_view(
        self, name: str, table: pd.DataFrame | TablePath
    ) -> BodoSQLContext:
        """Create a new BodoSQLContext that contains all of the old DataFrames and the
        new table being provided. If there is a DataFrame in the old BodoSQLContext with
        the same name, it is replaced by the new table in the new BodoSQLContext. Otherwise
        the new table is just added under the new name.

        Args:
            name (str): Name of the new table
            table (Union[pd.DataFrame,  TablePath]): New tables

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(name, str):
            raise BodoError(
                "BodoSQLContext.add_or_replace_view(): 'name' must be a string"
            )
        if not isinstance(table, (pd.DataFrame, TablePath)):
            raise BodoError(
                "BodoSQLContext.add_or_replace_view(): 'table' must be a Pandas DataFrame or BodoSQL TablePath"
            )
        new_tables = self.tables.copy()
        new_tables[name] = table
        return BodoSQLContext(new_tables, self.catalog, self.default_tz)

    def remove_view(self, name: str):
        """Create a new BodoSQLContext by removing the table with the
        given name.

        Args:
            name (str): Name of the table to remove.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(name, str):
            raise BodoError(
                "BodoSQLContext.remove_view(): 'name' must be a constant string"
            )
        new_tables = self.tables.copy()
        if name not in new_tables:
            raise BodoError(
                "BodoSQLContext.remove_view(): 'name' must refer to a registered view"
            )
        del new_tables[name]
        return BodoSQLContext(new_tables, self.catalog, self.default_tz)

    def add_or_replace_catalog(self, catalog: DatabaseCatalog):
        """
        Creates a new BodoSQL context by replacing the previous catalog,
        if it exists, with the provided catalog.

        Args:
            catalog (DatabaseCatalog): DatabaseCatalog to add to the context.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(catalog, DatabaseCatalog):
            raise BodoError(
                "BodoSQLContext.add_or_replace_catalog(): 'catalog' must be a bodosql.DatabaseCatalog"
            )
        return BodoSQLContext(self.tables, catalog, self.default_tz)

    def remove_catalog(self):
        """
        Creates a new BodoSQL context by remove the previous catalog.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if self.catalog is None:
            raise BodoError(
                "BodoSQLContext.remove_catalog(): BodoSQLContext must have an existing catalog registered."
            )
        return BodoSQLContext(self.tables, self.default_tz, self.default_tz)

    def __eq__(self, bc: object) -> bool:
        if isinstance(bc, BodoSQLContext):
            # Since the dictionary can contain either
            # DataFrames or table paths, we must add separate
            # checks for both.
            curr_keys = set(self.tables.keys())
            bc_keys = set(bc.tables.keys())
            if curr_keys == bc_keys:
                for key in curr_keys:
                    if isinstance(self.tables[key], TablePath) and isinstance(
                        bc.tables[key], TablePath
                    ):
                        if not self.tables[key].equals(
                            bc.tables[key]
                        ):  # pragma: no cover
                            return False
                    elif isinstance(self.tables[key], pd.DataFrame) and isinstance(
                        bc.tables[key], pd.DataFrame
                    ):  # pragma: no cover
                        # DataFrames may not have exactly the same dtypes becasue of flags inside boxing (e.g. object -> string)
                        # As a result we determine equality using assert_frame_equals
                        try:
                            pd.testing.assert_frame_equal(
                                self.tables[key],
                                bc.tables[key],
                                check_dtype=False,
                                check_index_type=False,
                            )
                        except AssertionError:
                            return False
                    else:
                        return False
                return self.catalog == bc.catalog
        return False  # pragma: no cover

    def execute_ddl(self, sql: str, generator=None) -> pd.DataFrame:
        """API to directly execute DDL queries. This is used by the JIT
        path to execute DDL queries and can be used as a fast path when you
        statically know the query you want to execute is a DDL query to avoid the
        control flow/cleanup code.

        This will execute any DDL query on rank 0 and then broadcast the result
        to all ranks.

        Args:
            sql (str): The DDL query to execute.
            generator (Optional[RelationalAlgebraGenerator Java object]): The prepared planner
                information used for executing the query. If None we need to create
                the planner.

        Returns:
            pd.DataFrame: The result of the DDL query as a Pandas DataFrame.
        """
        from bodo.mpi4py import MPI

        comm = MPI.COMM_WORLD
        result = None
        error = None
        create_generator = comm.bcast(generator is None)
        if create_generator:
            # Prepare the relational algebra generator on rank 0.
            # The assumption is this code is called directly as the
            # external API so we need to parse the query.
            generator = self._create_planner_and_parse_query(
                sql,
                False,  # We need to execute the code so don't hide credentials.
            )

        if bodo.get_rank() == 0:
            try:
                ddl_result = JavaEntryPoint.executeDDL(generator, sql)
                # Convert the output to a DataFrame.
                column_names = list(
                    JavaEntryPoint.getDDLExecutionColumnNames(ddl_result)
                )
                column_types = [
                    _generate_ddl_column_type(t)
                    for t in JavaEntryPoint.getDDLExecutionColumnTypes(ddl_result)
                ]
                data = [
                    # Use astype to avoid issues with Java conversion.
                    pd.array(column, dtype=object).astype(column_types[i])
                    for i, column in enumerate(
                        JavaEntryPoint.getDDLColumnValues(ddl_result)
                    )
                ]
                df_dict = {column_names[i]: data[i] for i in range(len(column_names))}
                result = pd.DataFrame(
                    df_dict,
                )
            except Exception as e:
                error = error_to_string(e)
        result = comm.bcast(result)
        error = comm.bcast(error)
        # Throw the error on all ranks.
        if error is not None:
            raise BodoError(error)
        return result


def _generate_ddl_column_type(type_string: str) -> Any:
    """Convert a string representation of a Pandas column type
    passed from Java to a Python type.

    Args:
        type_string (str): A string for the expression you would
            execution in Python to get the type.

    Returns:
        Any: The actual type object.
    """
    glbls = {"pd": pd, "pa": pa}
    locs = {}
    exec(f"ddl_type = {type_string}", glbls, locs)
    return locs["ddl_type"]


def initialize_schema():
    """Create the BodoSQL Schema used to store all local DataFrames.

    Returns:
        Java LocalSchema: Java type for the BodoSQL schema.
    """
    # TODO(ehsan): create and store generator during bodo_sql_context initialization
    if bodo.get_rank() == 0:
        schema = JavaEntryPoint.buildLocalSchema("__BODOLOCAL__")
    else:
        schema = None
    return schema


def update_schema(
    schema,
    table_names: list[str],
    df_types: list[bodo.DataFrameType],
    estimated_row_counts: list[int | None],
    estimated_ndvs: list[dict[str, int] | None],
    bodo_types: list[types.Type],
    from_jit: bool,
    write_type: str,
):
    """Update a local schema with local tables.

    Args:
        schema (Java LocalSchema): The schema to update.
        table_names (List[str]): List of tables to add to the schema.
        df_types (List[bodo.DataFrameType]): List of Bodo DataFrame types for each table.
        estimated_row_counts (List[Optional[int]]): The expected number of rows in each input
            table for the volcano planner. None if no estimate is provided.
        estimated_ndvs (List[Optional[dict[str, int]]]): The NDV estimates for each input table.
        bodo_types (List[types.Type]): List of Bodo types for each table. This stores
            the original type, so a TablePath isn't converted to its
            DataFrameType, which it is for df_types.
        from_jit (bool): Is this typing coming from JIT?
        write_type (str): String describing the type of write used for generating the write code.
            Will be "MERGE" for MERGE INTO queries, and defaults to "INSERT" for all other
            queries.
    """
    if bodo.get_rank() == 0:
        for i in range(len(table_names)):
            add_table_type(
                table_names[i],
                schema,
                df_types[i],
                estimated_row_counts[i],
                estimated_ndvs[i],
                bodo_types[i],
                i,
                from_jit,
                write_type,
            )


def _ensure_dynamic_params_list(dynamic_params_list: Any) -> list:
    """Verify the supplied Dynamic params list is a supported type
    and converts the result to a list.

    Args:
        dynamic_params_list (Any): A representation of the dynamic params list.

    Returns:
        List: The dynamic params list converted to a list equivalent.
    """
    if dynamic_params_list is None:
        return []
    elif isinstance(dynamic_params_list, tuple):
        return list(dynamic_params_list)
    elif isinstance(dynamic_params_list, list):
        return dynamic_params_list
    else:
        # Only specify tuple in the error message because we may not be able
        # to support lists in JIT.
        raise BodoError(
            "dynamic_params_list must be a tuple of Python variables if provided"
        )
