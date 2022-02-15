import hashlib
import inspect
import warnings

import snowflake.sqlalchemy
import sqlalchemy.types as sqltypes
from sqlalchemy import exc as sa_exc
from sqlalchemy import util as sa_util
from sqlalchemy.sql import text

# flag for checking whether the functions we are replacing have changed in a later Snowflake
# sqlalchemy release. Needs to be checked for every new release so we update our changes.
_check_snowflake_sqlalchemy_change = True


# Bodo Change: Replace unknown column types with null
# TODO: Remove when this the fix is merged
# https://github.com/snowflakedb/snowflake-sqlalchemy/pull/215


def _get_schema_columns(self, connection, schema, **kw):
    """Get all columns in the schema, if we hit 'Information schema query returned too much data' problem return
    None, as it is cacheable and is an unexpected return type for this function"""
    ans = {}
    current_database, _ = self._current_database_schema(connection, **kw)
    full_schema_name = self._denormalize_quote_join(current_database, schema)
    try:
        schema_primary_keys = self._get_schema_primary_keys(
            connection, full_schema_name, **kw
        )
        result = connection.execute(
            text(
                """
        SELECT /* sqlalchemy:_get_schema_columns */
                ic.table_name,
                ic.column_name,
                ic.data_type,
                ic.character_maximum_length,
                ic.numeric_precision,
                ic.numeric_scale,
                ic.is_nullable,
                ic.column_default,
                ic.is_identity,
                ic.comment
            FROM information_schema.columns ic
            WHERE ic.table_schema=:table_schema
            ORDER BY ic.ordinal_position"""
            ),
            {"table_schema": self.denormalize_name(schema)},
        )
    except sa_exc.ProgrammingError as pe:
        if pe.orig.errno == 90030:
            # This means that there are too many tables in the schema, we need to go more granular
            return None  # None triggers _get_table_columns while staying cacheable
        raise
    for (
        table_name,
        column_name,
        coltype,
        character_maximum_length,
        numeric_precision,
        numeric_scale,
        is_nullable,
        column_default,
        is_identity,
        comment,
    ) in result:
        table_name = self.normalize_name(table_name)
        column_name = self.normalize_name(column_name)
        if table_name not in ans:
            ans[table_name] = list()
        if column_name.startswith("sys_clustering_column"):
            continue  # ignoring clustering column
        col_type = self.ischema_names.get(coltype, None)
        col_type_kw = {}
        if col_type is None:
            sa_util.warn(
                "Did not recognize type '{}' of column '{}'".format(
                    coltype, column_name
                )
            )
            col_type = sqltypes.NULLTYPE
        else:
            if issubclass(col_type, sqltypes.FLOAT):
                col_type_kw["precision"] = numeric_precision
                col_type_kw["decimal_return_scale"] = numeric_scale
            elif issubclass(col_type, sqltypes.Numeric):
                col_type_kw["precision"] = numeric_precision
                col_type_kw["scale"] = numeric_scale
            elif issubclass(col_type, (sqltypes.String, sqltypes.BINARY)):
                col_type_kw["length"] = character_maximum_length

        # BODO CHANGE: Replace with default NULL
        type_instance = (
            col_type
            if isinstance(col_type, sqltypes.NullType)
            else col_type(**col_type_kw)
        )

        current_table_pks = schema_primary_keys.get(table_name)

        ans[table_name].append(
            {
                "name": column_name,
                "type": type_instance,
                "nullable": is_nullable == "YES",
                "default": column_default,
                "autoincrement": is_identity == "YES",
                "comment": comment,
                "primary_key": (
                    column_name
                    in schema_primary_keys[table_name]["constrained_columns"]
                )
                if current_table_pks
                else False,
            }
        )
    return ans


if _check_snowflake_sqlalchemy_change:  # pragma: no cover
    # TODO: Determine if @reflection.cache can cause issues
    lines = inspect.getsource(
        snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_schema_columns
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "fdf39af1ac165319d3b6074e8cf9296a090a21f0e2c05b644ff8ec0e56e2d769"
    ):
        warnings.warn(
            "snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_schema_columns has changed"
        )

snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_schema_columns = (
    _get_schema_columns
)


# Bodo Change: Replace unknown column types with null.
# TODO: Remove when this the fix is merged
# https://github.com/snowflakedb/snowflake-sqlalchemy/pull/215
def _get_table_columns(self, connection, table_name, schema=None, **kw):
    """Get all columns in a table in a schema"""
    ans = []
    current_database, _ = self._current_database_schema(connection, **kw)
    full_schema_name = self._denormalize_quote_join(current_database, schema)
    schema_primary_keys = self._get_schema_primary_keys(
        connection, full_schema_name, **kw
    )
    result = connection.execute(
        text(
            """
    SELECT /* sqlalchemy:get_table_columns */
            ic.table_name,
            ic.column_name,
            ic.data_type,
            ic.character_maximum_length,
            ic.numeric_precision,
            ic.numeric_scale,
            ic.is_nullable,
            ic.column_default,
            ic.is_identity,
            ic.comment
        FROM information_schema.columns ic
        WHERE ic.table_schema=:table_schema
        AND ic.table_name=:table_name
        ORDER BY ic.ordinal_position"""
        ),
        {
            "table_schema": self.denormalize_name(schema),
            "table_name": self.denormalize_name(table_name),
        },
    )
    for (
        table_name,
        column_name,
        coltype,
        character_maximum_length,
        numeric_precision,
        numeric_scale,
        is_nullable,
        column_default,
        is_identity,
        comment,
    ) in result:
        table_name = self.normalize_name(table_name)
        column_name = self.normalize_name(column_name)
        if column_name.startswith("sys_clustering_column"):
            continue  # ignoring clustering column
        col_type = self.ischema_names.get(coltype, None)
        col_type_kw = {}
        if col_type is None:
            sa_util.warn(
                "Did not recognize type '{}' of column '{}'".format(
                    coltype, column_name
                )
            )
            col_type = sqltypes.NULLTYPE
        else:
            if issubclass(col_type, sqltypes.FLOAT):
                col_type_kw["precision"] = numeric_precision
                col_type_kw["decimal_return_scale"] = numeric_scale
            elif issubclass(col_type, sqltypes.Numeric):
                col_type_kw["precision"] = numeric_precision
                col_type_kw["scale"] = numeric_scale
            elif issubclass(col_type, (sqltypes.String, sqltypes.BINARY)):
                col_type_kw["length"] = character_maximum_length

        # BODO CHANGE: Replace with default NULL
        type_instance = (
            col_type
            if isinstance(col_type, sqltypes.NullType)
            else col_type(**col_type_kw)
        )

        current_table_pks = schema_primary_keys.get(table_name)

        ans.append(
            {
                "name": column_name,
                "type": type_instance,
                "nullable": is_nullable == "YES",
                "default": column_default,
                "autoincrement": is_identity == "YES",
                "comment": comment if comment != "" else None,
                "primary_key": (
                    column_name
                    in schema_primary_keys[table_name]["constrained_columns"]
                )
                if current_table_pks
                else False,
            }
        )
    return ans


if _check_snowflake_sqlalchemy_change:  # pragma: no cover
    # TODO: Determine if @reflection.cache can cause issues
    lines = inspect.getsource(
        snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_table_columns
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "9ecc8a2425c655836ade4008b1b98a8fd1819f3be43ba77b0fbbfc1f8740e2be"
    ):
        warnings.warn(
            "snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_table_columns has changed"
        )


snowflake.sqlalchemy.snowdialect.SnowflakeDialect._get_table_columns = (
    _get_table_columns
)
