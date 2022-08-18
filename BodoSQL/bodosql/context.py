import re
import warnings
from enum import Enum
from typing import Union

import numba
import numpy as np
import pandas as pd
from bodosql.bodosql_types.database_catalog import DatabaseCatalog
from bodosql.bodosql_types.table_path import TablePath, TablePathType
from bodosql.py4j_gateway import get_gateway
from bodosql.utils import BodoSQLWarning, java_error_to_msg
from numba.core import ir, types

import bodo
from bodo.ir.sql_ext import parse_dbtype, remove_iceberg_prefix
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError

# Name for paramter table
NAMED_PARAM_TABLE_NAME = "__$bodo_named_param_table__"


error = None
# Based on my understanding of the Py4J Memory model, it should be safe to just
# Create/use java objects in much the same way as we did with jpype.
# https://www.py4j.org/advanced_topics.html#py4j-memory-model
saw_error = False
msg = ""
gateway = get_gateway()
if bodo.get_rank() == 0:
    try:
        ArrayClass = gateway.jvm.java.util.ArrayList
        ColumnTypeClass = (
            gateway.jvm.com.bodosql.calcite.catalog.domain.CatalogColumnDataType
        )
        ColumnClass = gateway.jvm.com.bodosql.calcite.catalog.domain.CatalogColumnImpl
        TableClass = gateway.jvm.com.bodosql.calcite.catalog.domain.CatalogTableImpl
        DatabaseClass = (
            gateway.jvm.com.bodosql.calcite.catalog.domain.CatalogDatabaseImpl
        )
        BodoSqlSchemaClass = gateway.jvm.com.bodosql.calcite.schema.BodoSqlSchema
        RelationalAlgebraGeneratorClass = (
            gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator
        )
        PropertiesClass = gateway.jvm.java.util.Properties
        SnowflakeCatalogImplClass = (
            gateway.jvm.com.bodosql.calcite.catalog.connection.SnowflakeCatalogImpl
        )
    except Exception as e:
        saw_error = True
        msg = str(e)
else:
    ArrayClass = None
    ColumnTypeClass = None
    ColumnClass = None
    TableClass = None
    DatabaseClass = None
    BodoSqlSchemaClass = None
    RelationalAlgebraGeneratorClass = None
    PropertiesClass = None
    SnowflakeCatalogImplClass = None

saw_error = bcast_scalar(saw_error)
msg = bcast_scalar(msg)
if saw_error:
    raise BodoError(msg)


# NOTE: These are defined in CatalogColumnDataType and must match here
class SqlTypeEnum(Enum):
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
    Bool = 11
    Date = 12
    Datetime = 13
    Timedelta = 14
    DateOffset = 15
    String = 16
    Binary = 17


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
    # TODO: Enable these types when more coverage exists
    # bodo.datetime_date_type: SqlTypeEnum.Date.value,
    # Binary should be disabled, but it is needed for an
    # Apple use case.
    bodo.bytes_type: SqlTypeEnum.Binary.value,
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
    bodo.pd_timestamp_type: SqlTypeEnum.Datetime.value,
    bodo.pd_timedelta_type: SqlTypeEnum.Timedelta.value,
    # date_offset_type represents Timedelta year/month
    # and is support only for scalars
    bodo.date_offset_type: SqlTypeEnum.DateOffset.value,
    # TODO: Support Date and Binary
}


def get_sql_column_type_id(arr_type, col_name, columns_needing_conversion):
    """get SQL type id from Numba array type
    columns_needing_conversion is a dictionary that should map column_names to
    valid strings used to convert the types.
    """
    if arr_type.dtype in _numba_to_sql_column_type_map:
        return _numba_to_sql_column_type_map[arr_type.dtype]
    elif arr_type.dtype == bodo.datetime_date_type:
        columns_needing_conversion[col_name] = "datetime64[ns]"
        return _numba_to_sql_column_type_map[bodo.datetime64ns]
    elif isinstance(arr_type.dtype, bodo.PDCategoricalDtype):
        elem_type = arr_type.dtype.elem_type
        if elem_type == bodo.datetime64ns:
            conv_str = "datetime64[ns]"
        elif elem_type == bodo.timedelta64ns:
            conv_str = "timedelta64[ns]"
        elif elem_type == types.unicode_type:
            conv_str = "str"
        elif isinstance(elem_type, types.Integer):
            if elem_type.signed:
                conv_str = "Int" + str(elem_type.bitwidth)
            else:
                conv_str = "UInt" + str(elem_type.bitwidth)
        elif isinstance(elem_type, types.Float):
            conv_str = elem_type.name
        elif elem_type == types.bool_:
            conv_str = "boolean"
        else:
            conv_str = None
        if conv_str is not None:
            columns_needing_conversion[col_name] = conv_str
            return _numba_to_sql_column_type_map[elem_type]
    elif isinstance(arr_type, bodo.DatetimeArrayType):
        # TODO [BS-641]: Treat TZ-Aware as its own internal type.
        return SqlTypeEnum.Timedelta.value
    raise BodoError(
        f"Pandas column '{col_name}' with type {arr_type} not supported in BodoSQL. Please cast your data to a supported type. https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    )


def get_sql_param_type_id(param_type, param_name):
    """get SQL type id from a Bodo scalar type. Also returns
    if there was a literal type used for outputting a warning.x"""
    unliteral_type = types.unliteral(param_type)
    if unliteral_type in _numba_to_sql_param_type_map:
        return (
            _numba_to_sql_param_type_map[unliteral_type],
            unliteral_type != param_type,
        )
    raise TypeError(
        f"Scalar value: '{param_name}' with type {param_type} not supported in BodoSQL. Please cast your data to a supported type. https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    )


def compute_df_types(df_list, is_bodo_type):
    """Given a list of Bodo types or Python objects,
    determines the dataframe type for each object. This
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
                    type_info = (
                        bodo.transforms.untyped_pass._get_sql_types_arr_colnames(
                            f"select * from {file_path}",
                            const_conn_str,
                            ir.Var(None, "dummy_var", ir.Loc("dummy_loc", -1)),
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


def get_table_type(
    table_name, db, df_type, columns_needing_conversion, table_types, bodo_type
):
    """get SQL Table type in Java for Numba dataframe type
    columns_needing_conversion is a dictionary that should be populated
    with any column names and a string that should be used for an astype conversion.

    Only returns the TableClass on rank 0. On all other ranks, only serves to update table_types.
    """
    # Store the typing information for conversion
    table_types[table_name] = df_type

    if bodo.get_rank() == 0:
        col_arr = ArrayClass()
        for i, cname in enumerate(df_type.columns):
            type_id = get_sql_column_type_id(
                df_type.data[i], cname, columns_needing_conversion
            )
            dataType = ColumnTypeClass.fromTypeId(type_id)
            column = ColumnClass(cname, dataType, i)
            col_arr.add(column)

        # To support writing to SQL Databases we register a different output
        # for TablePath APIs.
        # TODO: Revamp BodoSQL table representation with a more consistent
        # format.
        if isinstance(bodo_type, TablePathType) and bodo_type._file_type == "sql":
            return TableClass(
                table_name,
                db,
                col_arr,
                bodo_type._file_path,
                bodo_type._db_schema,
                bodo_type._conn_str,
            )
        else:
            return TableClass(table_name, db, col_arr)


def get_param_table_type(table_name, db, param_keys, param_values):
    """get SQL Table type in Java for Numba dataframe type"""
    assert bodo.get_rank() == 0, "get_table_type should only be called on rank 0."
    param_arr = ArrayClass()
    literal_params = []
    for i in range(len(param_keys)):
        param_name = param_keys[i]
        param_type = param_values[i]
        type_id, is_literal = get_sql_param_type_id(param_type, param_name)
        if is_literal:
            literal_params.append(param_name)
        dataType = ColumnTypeClass.fromTypeId(type_id)
        paramType = ColumnClass(param_name, dataType, i)
        param_arr.add(paramType)

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

    return TableClass(table_name, db, param_arr)


class BodoSQLContext:
    def __init__(self, tables=None, catalog=None):
        # We only need to initialize the tables values on all ranks, since that is needed for
        # creating the JIT function on all ranks for bc.sql calls. We also intialize df_types on all ranks,
        # for consistency. All the other attributes
        # are only used for generating the functext, which is only done on rank 0.
        if tables is None:
            tables = {}

        self.tables = tables
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
            for k, v in tables.items():
                names.append(k)
                dfs.append(v)
            orig_bodo_types, df_types = compute_df_types(dfs, False)
            table_map = {names[i]: df_types[i] for i in range(len(dfs))}
            orig_bodo_types_map = {
                names[i]: orig_bodo_types[i] for i in range(len(dfs))
            }
            db, table_conversions = intialize_database(
                table_map, orig_bodo_types_map, param_key_values=None
            )
            self.db = db
            self.table_conversions = table_conversions
            self.orig_bodo_types = orig_bodo_types
            self.df_types = df_types
        except Exception as e:
            failed = True
            msg = str(e)

        failed = bcast_scalar(failed)
        msg = bcast_scalar(msg)
        if failed:
            raise BodoError(msg)

    def convert_to_pandas(self, sql, params_dict=None):
        """converts SQL code to Pandas"""
        pd_code, lowered_globals = self._convert_to_pandas(sql, True, params_dict)
        # Replace the global variable with the actual constant value, for better readability
        for varname, glbl in lowered_globals.items():
            pd_code = pd_code.replace(varname, str(glbl))
        return pd_code

    def _convert_to_pandas_unoptimized(self, sql, params_dict=None):
        """convert SQL code to Pandas"""
        pd_code, lowered_globals = self._convert_to_pandas(sql, False, params_dict)
        # Replace the global variable with the actual constant value, for better readability
        for varname, glbl in lowered_globals.items():
            pd_code = pd_code.replace(varname, str(glbl))
        return pd_code

    def _setup_named_params(self, params_dict):

        assert (
            bodo.get_rank() == 0
        ), "_setup_named_params should only be called on rank 0."
        if params_dict is None:
            params_dict = dict()

        # Create the named params table
        param_values = [bodo.typeof(x) for x in params_dict.values()]
        paramsJava = get_param_table_type(
            NAMED_PARAM_TABLE_NAME, self.db, tuple(params_dict.keys()), param_values
        )

        # Add named params to the schema
        self.db.addTable(paramsJava)

    def _remove_named_params(self):
        self.db.removeTable(NAMED_PARAM_TABLE_NAME)

    def _convert_to_pandas(self, sql, optimizePlan, params_dict):
        """convert SQL code to Pandas functext. Generates the func_text on rank 0, the
        errors/results are broadcast to all ranks."""
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
                pd_code, used_tables, globalsToLower = self._get_pandas_code(
                    sql, optimizePlan
                )
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [(str(k), str(v)) for k, v in globalsToLower.items()]
                )

                # Remove the named Params table
                self._remove_named_params()

                args = ", ".join(list(self.tables.keys()) + list(params_dict.keys()))
                func_text_or_err_msg += f"def impl({args}):\n"

                # Determine which tables require an astype conversion.
                table_conversions = self.table_conversions

                # Convert to the same form as JIT to reuse code.
                table_names, table_types = self._get_jit_table_name_type_format()
                pd_func_text = f"{pd_code}\n"
                func_text_or_err_msg += generate_used_table_func_text(
                    table_names,
                    used_tables,
                    table_types,
                    table_conversions,
                    pd_func_text,
                    False,
                )
            except Exception as e:
                failed = True
                func_text_or_err_msg = str(e)

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
            sql, optimizePlan, params_dict
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

    def generate_plan(self, sql, params_dict=None):
        """
        Return the optimized plan for the SQL code as
        as a Python string.
        """
        failed = False
        plan_or_err_msg = ""
        if bodo.get_rank() == 0:
            try:
                self._setup_named_params(params_dict)
                generator = self._create_generator()
                plan_or_err_msg = str(generator.getOptimizedPlanString(sql))
                # Remove the named Params table
                self._remove_named_params()
            except Exception as e:
                failed = True
                plan_or_err_msg = str(e)
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
                generator = self._create_generator()
                plan_or_err_msg = str(generator.getUnoptimizedPlanString(sql))
                # Remove the named Params table
                self._remove_named_params()
            except Exception as e:
                failed = True
                plan_or_err_msg = str(e)
        failed = bcast_scalar(failed)
        plan_or_err_msg = bcast_scalar(plan_or_err_msg)
        if failed:
            raise BodoError(plan_or_err_msg)
        return plan_or_err_msg

    def _get_pandas_code(self, sql, optimized):
        # Construct the relational algebra generator
        if sql.strip() == "":
            bodo.utils.typing.raise_bodo_error(
                "BodoSQLContext passed empty query string"
            )
        generator = self._create_generator()

        if optimized:
            try:
                pd_code = str(generator.getPandasString(sql))
                failed = False
            except Exception as e:
                message = java_error_to_msg(e)
                failed = True
            if failed:
                # Raise BodoError outside except to avoid stack trace
                raise bodo.utils.typing.BodoError(
                    f"Unable to parse SQL Query. Error message:\n{message}"
                )
        else:
            try:
                pd_code = str(generator.getPandasStringUnoptimized(sql))
                failed = False
            except Exception as e:
                message = java_error_to_msg(e)
                failed = True
            if failed:
                # Raise BodoError outside except to avoid stack trace
                raise bodo.utils.typing.BodoError(
                    f"Unable to parse SQL Query. Error message:\n{message}"
                )
        try:
            used_tables = get_used_tables_from_generator(generator)
        except Exception as e:
            message = java_error_to_msg(e)
            failed = True
        if failed:
            # Raise BodoError outside except to avoid stack trace
            raise bodo.utils.typing.BodoError(
                f"Unable to parse SQL Query. Error message:\n{message}"
            )
        return pd_code, used_tables, generator.getLoweredGlobalVariables()

    def _create_generator(self):
        """
        Creates a generator from the given schema
        """
        schema = BodoSqlSchemaClass(self.db)
        generator = RelationalAlgebraGeneratorClass(schema, NAMED_PARAM_TABLE_NAME)
        return generator

    def _get_jit_table_name_type_format(self):
        """
        Outputs 3 lists table_names, table_types, and df_types
        that conform to how JIT organizes tables. This is done
        to enable reusing code.
        """
        table_names = []
        for key in self.tables.keys():
            table_names.append(key)
        return table_names, self.orig_bodo_types

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
        return BodoSQLContext(new_tables, self.catalog)

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
        return BodoSQLContext(new_tables, self.catalog)

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
        return BodoSQLContext(self.tables, catalog)

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
        return BodoSQLContext(self.tables)

    def __eq__(self, bc: object) -> bool:
        if isinstance(bc, BodoSQLContext):
            # Since the dictionary can contain either
            # DataFrames or table paths, we must add separate
            # checks for both.
            curr_keys = set(self.tables.keys())
            bc_keys = set(bc.tables.keys())
            if curr_keys == bc_keys:
                for key in curr_keys:
                    if not self.tables[key].equals(bc.tables[key]):  # pragma: no cover
                        return False
                return self.catalog == bc.catalog
        return False  # pragma: no cover


def intialize_database(name_to_df_type, name_to_bodo_type, param_key_values=None):
    """Helper function used by both BodoSQLContext, and BodoSQLContextType.
    Initializes the java database, and returns two dictionaries containing needed conversions,
    and the dataframe types
    Args:
        name_to_df_dict: dict of table names to dataframe types,
        param_key_values: An optional tuple of (parameter keys, parameter values).
        database: the database object
        table_conversions: a dict of df names -> tuple of columns that need to be converted.
            None on all ranks but 0.
    """

    assert param_key_values is None or isinstance(param_key_values, tuple)

    # TODO(ehsan): create and store generator during bodo_sql_context initialization
    if bodo.get_rank() == 0:
        db = DatabaseClass("__bodolocal__")
        table_conversions = {}
    else:
        db = None
        table_conversions = None
    df_types = {}
    for table_name, df_type in name_to_df_type.items():
        columns_needing_conversion = {}
        bodo_type = name_to_bodo_type[table_name]
        tableJava = get_table_type(
            table_name, db, df_type, columns_needing_conversion, df_types, bodo_type
        )
        if bodo.get_rank() == 0:
            table_conversions[table_name] = columns_needing_conversion
            db.addTable(tableJava)

    if bodo.get_rank() == 0 and param_key_values is not None:
        (param_keys, param_values) = param_key_values
        paramsJava = get_param_table_type(
            NAMED_PARAM_TABLE_NAME, db, param_keys, param_values
        )
        db.addTable(paramsJava)
    return db, table_conversions


def get_used_tables_from_generator(generator):
    """
    Given a generator, extract the used tables and convert to a
    Python data structure. This assumes a sql query has already been
    processed.
    """
    used_tables = generator.getUsedTables()
    # Convert the Java structures to Python
    used_tables = {
        tuple([x for x in val]): used_tables[val] for val in used_tables.keySet()
    }
    return used_tables


def generate_used_table_func_text(
    table_names,
    used_tables,
    table_types,
    table_conversions,
    func_text,
    from_jit,
):
    """
    Given an iterable of table names, a set of used tables, an iterable of table_types,
    where each member is either a DataFrameType or TablePathType, an iterable of df_types
    which are the DataFrameType for each table and table_conversions which is a dictionary mapping
    each table name to a dictionary of columns to new types, generates the func text to load the
    DataFrame.

    func_text contains all the code generated from Java and excludes any extra code
    inserted to make a proper Python function. As a result, the line numbers for
    where to insert tables from used_tables are accurate for this func_text.

    from_jit handles code differences between jit and python
    """
    # List of (line_number, code) for code to insert into the generated
    # code. This is done to delay IO loads until first use.
    total_code = []
    inserted_code = []
    for i, df_name in enumerate(table_names):
        # used_tables contains a set of tuples with (schema, table_name)
        # By default, tables are part of the __bodolocal__ schema
        key = ("__bodolocal__", df_name)
        column_conversions = table_conversions[df_name]
        if key in used_tables:
            table_typ = table_types[i]
            generated_line = ""
            if isinstance(table_typ, TablePathType):
                file_type = table_typ._file_type
                file_path = table_typ._file_path
                if file_type == "pq":
                    # TODO: Replace with runtime variable once we support specifying
                    # the schema
                    generated_line += f"  {df_name} = pd.read_parquet('{file_path}')\n"
                elif file_type == "sql":
                    # TODO: Replace with runtime variable once we support specifying
                    # the schema
                    conn_str = table_typ._conn_str
                    db_type, _ = parse_dbtype(conn_str)
                    if db_type == "iceberg":
                        generated_line += f"  {df_name} = pd.read_sql_table('{file_path}', '{conn_str}', '{table_typ._db_schema}')\n"
                    else:
                        generated_line += f"  {df_name} = pd.read_sql('select * from {file_path}', '{conn_str}')\n"
                else:
                    raise BodoError(
                        f"Internal Error: Unsupported TablePathType for type: '{file_type}'"
                    )
            # In the case we have a non-table path, we need to do a copy, so that the dataframe setitems
            # Don't have unintended side effects due to aliasing
            elif from_jit:
                if column_conversions:
                    generated_line += f"  {df_name} = bodo_sql_context.dataframes[{i}].copy(deep=False)\n"
                else:
                    generated_line += (
                        f"  {df_name} = bodo_sql_context.dataframes[{i}]\n"
                    )
            elif column_conversions:
                generated_line += f"  {df_name} = {df_name}.copy(deep=False)\n"

            # Do the conversions if needed, using bodosql_replace_columns
            # using bodosql_replace_columns is needed, because it will allow us to know that these assignments are safe
            # for filter pushdown in typing pass.
            if column_conversions:
                col_names = ", ".join(map(repr, column_conversions.keys()))
                astypes = ", ".join(
                    map(
                        lambda col_name_typ_tuple: f"{df_name}[{col_name_typ_tuple[0]!r}].astype('{col_name_typ_tuple[1]}', _bodo_nan_to_str=False)",
                        column_conversions.items(),
                    )
                )
                generated_line += f"  {df_name} = bodo.hiframes.dataframe_impl.__bodosql_replace_columns_dummy({df_name}, ({col_names},), ({astypes},))\n"

            if generated_line:
                if isinstance(table_typ, TablePathType) and table_typ._reorder_io:
                    # If we reorder io then we place it just before the first use
                    inserted_code.append((used_tables[key], generated_line))
                else:
                    # Otherwise we load all data at the start.
                    total_code.append(generated_line)
    if total_code or inserted_code:
        # If we are inserting any code, we want to insert each line at the correct
        # spot. As a result, we split the generated code by newlines and insert our code.
        inserted_code = sorted(inserted_code, key=lambda x: x[0])
        slice_start = 0
        generated_code = func_text.split("\n")
        for line_num, new_line in inserted_code:
            total_code.extend(generated_code[slice_start:line_num])
            total_code.append(new_line)
            slice_start = line_num
        # Insert any remaining code at the end
        total_code.extend(generated_code[slice_start:])
        # Recompute the func_text by combining with newlines.
        func_text = "\n".join(total_code)
    return func_text
