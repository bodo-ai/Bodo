import datetime
import os
import re
import time
import traceback
import warnings
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import pandas as pd
from numba.core import ir, types

import bodo
from bodo.ir.sql_ext import parse_dbtype, remove_iceberg_prefix
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError, dtype_to_array_type
from bodosql.bodosql_types.database_catalog import DatabaseCatalog
from bodosql.bodosql_types.table_path import TablePath, TablePathType
from bodosql.imported_java_classes import (
    ArrayListClass,
    BodoTZInfoClass,
    ColumnClass,
    ColumnDataEnum,
    ColumnDataTypeClass,
    LocalSchemaClass,
    LocalTableClass,
    RelationalAlgebraGeneratorClass,
)
from bodosql.utils import BodoSQLWarning, error_to_string

# Name for parameter table
NAMED_PARAM_TABLE_NAME = "__$BODO_NAMED_PARAM_TABLE__"


# NOTE: These are defined in BodoSQLColumnDataType and must match here
class SqlTypeEnum(Enum):
    Empty = 0
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
    # Note: This is only used for General Iceberg and isn't used for
    # decimal arrays yet.
    Decimal = 11
    Bool = 12
    Date = 13
    Time = 14
    Datetime = 15
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
    Unsupported = 27


# Scalar dtypes for supported Bodo Arrays
_numba_to_sql_column_type_map = {
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
    types.NPDatetime("ns"): SqlTypeEnum.Datetime.value,
    types.NPTimedelta("ns"): SqlTypeEnum.Timedelta.value,
    types.bool_: SqlTypeEnum.Bool.value,
    bodo.string_type: SqlTypeEnum.String.value,
    bodo.bytes_type: SqlTypeEnum.Binary.value,
    # Note date doesn't have native support yet, but the code to
    # cast to datetime64 is handled in the Java code.
    bodo.datetime_date_type: SqlTypeEnum.Date.value,
}

# Scalar dtypes for supported parameters
_numba_to_sql_param_type_map = {
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
    bodo.pd_timestamp_tz_naive_type: SqlTypeEnum.Datetime.value,
    bodo.pd_timedelta_type: SqlTypeEnum.Timedelta.value,
    bodo.timestamptz_array_type: SqlTypeEnum.Timestamp_Tz.value,
    # date_offset_type represents Timedelta year/month
    # and is support only for scalars
    bodo.date_offset_type: SqlTypeEnum.DateOffset.value,
    # TODO: Support Date and Binary parameters [https://bodo.atlassian.net/browse/BE-3542]
}


# Hacky way to get the planner type option to Java.
# I don't want to access the Java enum class or the constants
# defined in Java that are used for this decision from Python
# so we're going to redefine the enum here.
#
# Not intended as a public API.
class _PlannerType(IntEnum):
    Volcano = 0
    Streaming = 1


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
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Datetime.value)
        return ColumnDataTypeClass(type_enum, nullable, precision)
    else:
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Timestamp_Ltz.value)
        # Create the BodoTzInfo Java object.
        tz_info = BodoTZInfoClass(
            str(typ.tz), "int" if isinstance(typ.tz, int) else "str"
        )
        return ColumnDataTypeClass(type_enum, nullable, precision, tz_info)


def construct_time_array_type(
    typ: Union[bodo.TimeArrayType, bodo.TimeType], nullable: bool
):
    """Construct a BodoSQL data type for a time array.

    Args:
        typ (Union[bodo.TimeArrayType, bodo.TimeType]): A time Bodo type
        nullable (bool): Is the column Nullable

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Time.value)
    return ColumnDataTypeClass(type_enum, nullable, typ.precision)


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
    type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Array.value)
    return ColumnDataTypeClass(type_enum, True, child)


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
        key_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.String.value)
        key = ColumnDataTypeClass(key_enum, True)
        value_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Variant.value)
        value = ColumnDataTypeClass(value_enum, True)
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Json_Object.value)
        return ColumnDataTypeClass(type_enum, True, key, value)
    else:
        # TODO: Add map scalar support
        key = get_sql_data_type(arr_type.key_arr_type)
        value = get_sql_data_type(arr_type.value_arr_type)
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Json_Object.value)
        return ColumnDataTypeClass(type_enum, True, key, value)


def get_sql_column_type(arr_type, col_name):
    data_type = get_sql_data_type(arr_type)
    return ColumnClass(col_name, data_type)


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
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Timestamp_Tz.value)
        return ColumnDataTypeClass(type_enum, nullable)
    elif isinstance(arr_type, bodo.TimeArrayType):
        # Time array types have their own special handling for precision
        return construct_time_array_type(arr_type, nullable)
    elif isinstance(arr_type, bodo.DecimalArrayType):
        # For now, treat Decimal like float64 in BodoSQL planning
        warnings.warn(
            f"Type {arr_type} is not properly supported from a Python + Pandas DataFrame. BodoSQL will implicitly treat this column as a float64, which may lead to unexpected conversion errors. Please refer to the supported types: https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
        )
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Float64.value)
        return ColumnDataTypeClass(type_enum, nullable)
    elif isinstance(arr_type, bodo.ArrayItemArrayType):
        return construct_array_item_array_type(arr_type)
    elif isinstance(arr_type, (bodo.StructArrayType, bodo.MapArrayType)):
        return construct_json_array_type(arr_type)
    elif arr_type.dtype in _numba_to_sql_column_type_map:
        type_enum = ColumnDataEnum.fromTypeId(
            _numba_to_sql_column_type_map[arr_type.dtype]
        )
        return ColumnDataTypeClass(type_enum, nullable)
    elif isinstance(arr_type.dtype, bodo.PDCategoricalDtype):
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Categorical.value)
        child = get_sql_data_type(dtype_to_array_type(arr_type.dtype.elem_type, True))
        return ColumnDataTypeClass(type_enum, nullable, child)
    else:
        # The type is unsupported we raise a warning indicating this is a possible
        # error but we generate a dummy type because we may be able to support it
        # if its optimized out.
        warnings.warn(BodoSQLWarning(warning_msg))
        type_enum = ColumnDataEnum.fromTypeId(SqlTypeEnum.Unsupported.value)
        return ColumnDataTypeClass(type_enum, nullable)


def get_sql_param_type(param_type, param_name):
    """get SQL type from a Bodo scalar type. Also returns
    if there was a literal type used for outputting a warning."""
    unliteral_type = types.unliteral(param_type)
    # The named parameters are always scalars. We don't support
    # Optional types or None types yet. As a result this is always
    # non-null.
    nullable = False
    is_literal: bool = unliteral_type != param_type
    if (
        isinstance(unliteral_type, bodo.PandasTimestampType)
        and unliteral_type.tz != None
    ):
        # Timezone-aware Timestamps have their own special handling.
        return (
            ColumnClass(
                param_name, construct_tz_aware_array_type(param_type, nullable)
            ),
            is_literal,
        )
    elif isinstance(unliteral_type, bodo.TimeType):
        # Time array types have their own special handling for precision
        return (
            ColumnClass(param_name, construct_time_array_type(param_type, nullable)),
            is_literal,
        )
    elif unliteral_type in _numba_to_sql_param_type_map:
        type_enum = ColumnDataEnum.fromTypeId(
            _numba_to_sql_param_type_map[unliteral_type]
        )
        data_type = ColumnDataTypeClass(type_enum, nullable)
        return ColumnClass(param_name, data_type), is_literal
    raise TypeError(
        f"Scalar value: '{param_name}' with type {param_type} not supported in BodoSQL. Please cast your data to a supported type. https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
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
                    pruned_conn_str = remove_iceberg_prefix(const_conn_str)
                    db_schema = table_info._db_schema
                    iceberg_table_name = table_info._file_path
                    # table_name = table_info.
                    type_info = bodo.transforms.untyped_pass
                    # schema = table_info._schema
                    (
                        col_names,
                        col_types,
                        _pyarrow_table_schema,
                    ) = bodo.io.iceberg.get_iceberg_type_info(
                        iceberg_table_name, pruned_conn_str, db_schema
                    )
                else:
                    type_info = bodo.transforms.untyped_pass._get_sql_types_arr_colnames(
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
    schema: LocalSchemaClass,
    df_type: bodo.DataFrameType,
    estimated_row_count: Optional[int],
    bodo_type: types.Type,
    table_num: int,
    from_jit: bool,
    write_type: str,
):
    """Registers a new table into the schema. This is used to pass tables via DataFrames or the
    TablePath API.

    Args:
        table_name (str): The name of the table.
        schema (LocalSchemaClass): The schema to update.
        df_type (bodo.DataFrameType): The Bodo DataFrame type.
        estimated_row_count (Optional[int]): The expected number of rows in the table for the
            Volcano Planner. None if no estimate is provided.
        bodo_type (types.Type): Bodo type for the table. This stores the original type so a TablePath
            isn't converted to its DataFrameType, which the df_type always is.
        table_num (int): ID for the table being processed.
        from_jit (bool): Is this typing coming from JIT?
        write_type (str): String describing the type of write used for generating the write code.
            Will be "MERGE" for MERGE INTO queries, and defaults to "INSERT" for all other
            queries.
    """
    assert bodo.get_rank() == 0, "add_table_type should only be called on rank 0."
    col_arr = ArrayListClass()
    for i, cname in enumerate(df_type.columns):
        column = get_sql_column_type(df_type.data[i], cname)
        col_arr.add(column)

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
            write_format_code = f"bodo.io.iceberg.iceberg_merge_cow_py('{bodo_type._file_path}', '{bodo_type._conn_str}', '{bodo_type._db_schema}', %s, %s)"
        else:
            write_format_code = f"%s.to_sql('{bodo_type._file_path}', '{bodo_type._conn_str}', if_exists='append', index=False, {schema_code_to_sql}, %s)"
    else:
        write_format_code = ""

    # Determine the DB Type for generating java code.
    if isinstance(bodo_type, TablePathType):
        if bodo_type._file_type == "pq":
            db_type = "PARQUET"
        else:
            assert (
                bodo_type._file_type == "sql"
            ), "TablePathType is only implement for parquet and SQL APIs"
            const_conn_str = bodo_type._conn_str
            db_type, _ = parse_dbtype(const_conn_str)
    else:
        db_type = "MEMORY"

    read_code = _generate_table_read(table_name, bodo_type, table_num, from_jit)
    table = LocalTableClass(
        table_name,
        schema.getFullPath(),
        col_arr,
        is_writeable,
        read_code,
        write_format_code,
        # TablePath is a wrapper for a file so it results in an IO read.
        # The only other option is an in memory Pandas DataFrame.
        isinstance(bodo_type, TablePathType),
        db_type,
        estimated_row_count,
    )
    schema.addTable(table)


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
        read_line = "_ARG_" + table_name
    return read_line


def add_param_table(table_name, schema, param_keys, param_values):
    """get SQL Table type in Java for Numba DataFrame type"""
    assert bodo.get_rank() == 0, "add_param_table should only be called on rank 0."
    param_arr = ArrayListClass()
    literal_params = []
    for i in range(len(param_keys)):
        param_name = param_keys[i]
        param_type = param_values[i]
        param_java_type, is_literal = get_sql_param_type(param_type, param_name)
        if is_literal:
            literal_params.append(param_name)
        param_arr.add(param_java_type)

    if literal_params:
        warning_msg = (
            f"\nThe following named parameters: {literal_params} were typed as literals.\n"
            + "If these values are changed BodoSQL will be forced to recompile the code.\n"
            + "If you are passing JITs literals, you should consider passing these values"
            + " as arguments to your Python function.\n"
            + "For more information please refer to:\n"
            + "https://docs.bodo.ai/latest/api_docs/BodoSQL/#bodosql_named_params"
        )
        warnings.warn(BodoSQLWarning(warning_msg))

    # The readCode is unused for named Parameters as they will never reach
    # a table scan. Instead the original Python variable names will always
    # be used.
    schema.addTable(
        LocalTableClass(
            table_name,
            schema.getFullPath(),
            param_arr,
            False,
            "",
            "",
            False,
            "MEMORY",
            -1,
        )
    )


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
        if any([not isinstance(key, str) for key in self.tables.keys()]):
            raise BodoError("BodoSQLContext(): 'table' keys must be strings")
        if any(
            [
                not isinstance(value, (pd.DataFrame, TablePath))
                for value in self.tables.values()
            ]
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
            for k, v in tables.items():
                names.append(k)
                dfs.append(v)
                estimated_row_counts.append(_get_estimated_row_count(v))
            orig_bodo_types, df_types = compute_df_types(dfs, False)
            schema = initialize_schema(None)
            self.schema = schema
            self.names = names
            self.df_types = df_types
            self.orig_bodo_types = orig_bodo_types
            self.estimated_row_counts = estimated_row_counts
        except Exception as e:
            failed = True
            msg = error_to_string(e)

        failed = bcast_scalar(failed)
        msg = bcast_scalar(msg)
        if failed:
            raise BodoError(msg)

    def validate_query_compiles(self, sql, params_dict=None):
        """
        Verifies BodoSQL can fully compile the query in Bodo.
        """
        try:
            t1 = time.time()
            compiled_cpu_dispatcher = self._compile(sql, params_dict)
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

    def _compile(self, sql, params_dict=None):
        """compiles the query in Bodo."""
        optimizePlan = True
        import bodosql

        if params_dict is None:
            params_dict = dict()

        func_text, lowered_globals = self._convert_to_pandas(
            sql,
            optimizePlan,
            params_dict,
            False,  # We need to execute the code so don't hide credentials.
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
            "pd": pd,
        }

        glbls.update(lowered_globals)
        return self._functext_compile(func_text, params_dict, glbls)

    def _functext_compile(self, func_text, params_dict, glbls):
        """
        Helper function for _compile, that compiles the function text.
        This is mostly separated out for testing purposes.
        """

        arg_types = []
        for table_arg in self.tables.values():
            arg_types.append(bodo.typeof(table_arg))

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

        dispatcher = bodo.jit(
            sig,
            args_maybe_distributed=False,
            returns_maybe_distributed=False,
        )(impl)

        return dispatcher

    def validate_query(self, sql):
        """
        Verifies BodoSQL can compute query,
        but does not actually compile the query in Bodo.
        """
        try:
            code = self.convert_to_pandas(sql)
            executable_flag = True
        except:
            executable_flag = False

        return executable_flag

    def convert_to_pandas(self, sql, params_dict=None, hide_credentials=True):
        """converts SQL code to Pandas"""
        pd_code, lowered_globals = self._convert_to_pandas(
            sql,
            True,
            params_dict,
            hide_credentials,
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

    def _convert_to_pandas_unoptimized(
        self,
        sql,
        params_dict=None,
    ):
        """convert SQL code to Pandas"""
        pd_code, lowered_globals = self._convert_to_pandas(
            sql,
            False,
            params_dict,
            False,  # We need to execute the code so don't hide credentials.
        )
        # Add the global variable definitions at the begining of the fn,
        # for better readability
        added_defs = ""
        for varname, glbl in lowered_globals.items():
            added_defs += varname + " = " + repr(glbl) + "\n"
        return added_defs + pd_code

    def _setup_named_params(self, params_dict):
        assert (
            bodo.get_rank() == 0
        ), "_setup_named_params should only be called on rank 0."
        if params_dict is None:
            params_dict = dict()

        # Create the named params table
        param_values = [bodo.typeof(x) for x in params_dict.values()]
        add_param_table(
            NAMED_PARAM_TABLE_NAME, self.schema, tuple(params_dict.keys()), param_values
        )

    def _remove_named_params(self):
        self.schema.removeTable(NAMED_PARAM_TABLE_NAME)

    def _convert_to_pandas(
        self,
        sql: str,
        optimize_plan: bool,
        params_dict: Dict[str, Any],
        hide_credentials: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate the func_text for the Python code generated for the given SQL query.
        This is always computed entirely on rank 0 to avoid parallelism errors.

        Args:
            sql (str): The SQL query to process.
            optimize_plan (bool): Should the generated plan be optimized?
            params_dict (Dict[str, Any]): A python dictionary mapping Python variables
                to usable SQL names that can be referenced in the query.
            hide_credentials (bool): Should credentials be hidden in the generated code. This
                is set to true when we want to inspect the code but not run the code.

        Raises:
            BodoError: If the SQL query cannot be processed.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        func_text_or_err_msg = ""
        failed = False
        globalsToLower = ()
        if bodo.get_rank() == 0:
            # This try block should never run under normal circumstances,
            # but it's nice to have for debugging purposes so things don't hang
            # if we make any changes that could lead to a runtime error.
            try:
                if params_dict is None:
                    params_dict = dict()

                # Add named params to the schema
                self._setup_named_params(params_dict)

                # Generate the code
                pd_code, globalsToLower = self._get_pandas_code(
                    sql, optimize_plan, hide_credentials
                )
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [(str(k), str(v)) for k, v in globalsToLower.items()]
                )

                # Remove the named Params table
                self._remove_named_params()
                table_names = ["_ARG_" + x for x in self.tables.keys()]
                params_names = ["_PARAM_" + x for x in params_dict.keys()]
                args = ", ".join(table_names + params_names)
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
                },
                locs,
            )
            globalsDict[varname] = locs["value"]
        return func_text_or_err_msg, globalsDict

    def sql(self, sql, params_dict=None):
        return self._sql(sql, True, params_dict)

    def _test_sql_unoptimized(self, sql, params_dict=None):
        return self._sql(sql, False, params_dict)

    def _sql(self, sql, optimizePlan, params_dict):
        import bodosql

        if params_dict is None:
            params_dict = dict()

        func_text, lowered_globals = self._convert_to_pandas(
            sql,
            optimizePlan,
            params_dict,
            False,  # We need to execute the code so don't hide credentials.
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
            "pd": pd,
        }

        glbls.update(lowered_globals)
        loc_vars = {}
        exec(
            func_text,
            glbls,
            loc_vars,
        )
        impl = loc_vars["impl"]
        # TODO [BS-514]: Determine how to support parallel flags from Python
        return bodo.jit(
            impl, args_maybe_distributed=False, returns_maybe_distributed=False
        )(*(list(self.tables.values()) + list(params_dict.values())))

    def generate_plan(self, sql, params_dict=None, show_cost=False):
        """
        Return the optimized plan for the SQL code as
        as a Python string.
        """
        failed = False
        plan_or_err_msg = ""
        if bodo.get_rank() == 0:
            try:
                self._setup_named_params(params_dict)
                generator = self._create_generator(False)
                # Handle the parsing step.
                generator.parseQuery(sql)
                # Determine the write type
                write_type = generator.getWriteType(sql)
                # Update the schema with types.
                update_schema(
                    self.schema,
                    self.names,
                    self.df_types,
                    self.estimated_row_counts,
                    self.orig_bodo_types,
                    False,
                    write_type,
                )
                plan_or_err_msg = str(generator.getOptimizedPlanString(sql, show_cost))
                # Remove the named Params table
                self._remove_named_params()
            except Exception as e:
                failed = True
                plan_or_err_msg = error_to_string(e)
        failed = bcast_scalar(failed)
        plan_or_err_msg = bcast_scalar(plan_or_err_msg)
        if failed:
            raise BodoError(plan_or_err_msg)
        return plan_or_err_msg

    def generate_unoptimized_plan(self, sql, params_dict=None):
        """
        Return the unoptimized plan for the SQL code as
        as a Python string.
        """

        failed = False
        plan_or_err_msg = ""
        if bodo.get_rank() == 0:
            try:
                self._setup_named_params(params_dict)
                generator = self._create_generator(False)
                # Handle the parsing step.
                generator.parseQuery(sql)
                # Determine the write type
                write_type = generator.getWriteType(sql)
                # Update the schema with types.
                update_schema(
                    self.schema,
                    self.names,
                    self.df_types,
                    self.estimated_row_counts,
                    self.orig_bodo_types,
                    False,
                    write_type,
                )
                plan_or_err_msg = str(generator.getUnoptimizedPlanString(sql))
                # Remove the named Params table
                self._remove_named_params()
            except Exception as e:
                failed = True
                plan_or_err_msg = error_to_string(e)
        failed = bcast_scalar(failed)
        plan_or_err_msg = bcast_scalar(plan_or_err_msg)
        if failed:
            raise BodoError(plan_or_err_msg)
        return plan_or_err_msg

    def _get_pandas_code(
        self, sql: str, optimized: bool, hide_credentials: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate the Pandas code for the given SQL string.

        Args:
            sql (str): The SQL query text.
            optimized (bool): Should the plan be optimized for the generated code.
            hide_credentials (bool): Should credentials be hidden in the
                generated code. This is true when we want to just inspect code,
                not run it.

        Raises:
            bodo.utils.typing.BodoError: The SQL text is not supported.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        # Construct the relational algebra generator

        debugDeltaTable = bool(
            os.environ.get("__BODO_TESTING_DEBUG_DELTA_TABLE", False)
        )
        if sql.strip() == "":
            bodo.utils.typing.raise_bodo_error(
                "BodoSQLContext passed empty query string"
            )

        generator = self._create_generator(hide_credentials)
        # Handle the parsing step.
        generator.parseQuery(sql)
        # Determine the write type
        write_type = generator.getWriteType(sql)
        # Update the schema with types.
        update_schema(
            self.schema,
            self.names,
            self.df_types,
            self.estimated_row_counts,
            self.orig_bodo_types,
            False,
            write_type,
        )

        if optimized:
            try:
                pd_code = str(generator.getPandasString(sql, debugDeltaTable))
                failed = False
            except Exception as e:
                message = error_to_string(e)
                failed = True
            if failed:
                # Raise BodoError outside except to avoid stack trace
                raise bodo.utils.typing.BodoError(
                    f"Unable to parse SQL Query. Error message:\n{message}"
                )
        else:
            try:
                pd_code = str(
                    generator.getPandasStringUnoptimized(sql, debugDeltaTable)
                )
                failed = False
            except Exception as e:
                message = error_to_string(e)
                failed = True
            if failed:
                # Raise BodoError outside except to avoid stack trace
                raise bodo.utils.typing.BodoError(
                    f"Unable to parse SQL Query. Error message:\n{message}"
                )
        if failed:
            # Raise BodoError outside except to avoid stack trace
            raise bodo.utils.typing.BodoError(
                f"Unable to parse SQL Query. Error message:\n{message}"
            )
        return pd_code, generator.getLoweredGlobalVariables()

    def _create_generator(self, hide_credentials: bool):
        """Creates a RelationalAlgebraGenerator from the schema.

        Args:
            hide_credentials (bool): Should credentials be hidden for
                any generated code.

        Returns:
            RelationalAlgebraGeneratorClass: The java object holding
                the relational algebra generator.
        """
        verbose_level = bodo.user_logging.get_verbose_level()
        if bodo.bodosql_use_streaming_plan:
            planner_type = _PlannerType.Streaming.value
        else:
            planner_type = _PlannerType.Volcano.value
        if self.catalog is not None:
            return RelationalAlgebraGeneratorClass(
                self.catalog.get_java_object(),
                self.schema,
                NAMED_PARAM_TABLE_NAME,
                planner_type,
                verbose_level,
                bodo.bodosql_streaming_batch_size,
                hide_credentials,
                bodo.enable_snowflake_iceberg,
                bodo.enable_timestamp_tz,
                bodo.enable_runtime_join_filters,
            )
        extra_args = () if self.default_tz is None else (self.default_tz,)
        generator = RelationalAlgebraGeneratorClass(
            self.schema,
            NAMED_PARAM_TABLE_NAME,
            planner_type,
            verbose_level,
            bodo.bodosql_streaming_batch_size,
            hide_credentials,
            bodo.enable_snowflake_iceberg,
            bodo.enable_timestamp_tz,
            bodo.enable_runtime_join_filters,
            *extra_args,
        )
        return generator

    def add_or_replace_view(self, name: str, table: Union[pd.DataFrame, TablePath]):
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


def initialize_schema(
    param_key_values: Optional[Tuple[List[str], List[str]]] = None,
):
    """Create the BodoSQL Schema used to store all local DataFrames
    and update the named parameters.

    Args:
        param_key_values (Optional[Tuple[List[str], List[str]]]): Tuple of
            lists of named_parameter key value pairs. Defaults to None.

    Returns:
        LocalSchemaClass: Java type for the BodoSQL schema.
    """

    assert param_key_values is None or isinstance(param_key_values, tuple)

    # TODO(ehsan): create and store generator during bodo_sql_context initialization
    if bodo.get_rank() == 0:
        schema = LocalSchemaClass("__BODOLOCAL__")
        if param_key_values is not None:
            (param_keys, param_values) = param_key_values
            add_param_table(NAMED_PARAM_TABLE_NAME, schema, param_keys, param_values)
    else:
        schema = None
    return schema


def update_schema(
    schema: LocalSchemaClass,
    table_names: List[str],
    df_types: List[bodo.DataFrameType],
    estimated_row_counts: List[Optional[int]],
    bodo_types: List[types.Type],
    from_jit: bool,
    write_type: str,
):
    """Update a local schema with local tables.

    Args:
        schema (LocalSchemaClass): The schema to update.
        table_names (List[str]): List of tables to add to the schema.
        df_types (List[bodo.DataFrameType]): List of Bodo DataFrame types for each table.
        estimated_row_counts (List[Optional[int]]): The expected number of rows in each input
            table for the volcano planner. None if no estimate is provided.
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
                bodo_types[i],
                i,
                from_jit,
                write_type,
            )
