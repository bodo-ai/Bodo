package com.bodosql.calcite.ddl

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.type.SqlTypeName

/**
 * Generate the "schema" for DDL operations.
 */
class GenerateDDLTypes(private val typeFactory: RelDataTypeFactory) {
    fun generateType(ddlNode: SqlNode): RelDataType {
        // All DDL types likely use a string type.
        val stringType = typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.VARCHAR), true)
        val (fieldsNames, columnTypes) =
            when (ddlNode.kind) {
                // Drop Queries
                SqlKind.CREATE_VIEW, SqlKind.CREATE_SCHEMA, SqlKind.DROP_SCHEMA, SqlKind.BEGIN, SqlKind.COMMIT,
                SqlKind.ROLLBACK, SqlKind.DROP_TABLE, SqlKind.DROP_VIEW,
                -> {
                    val fieldNames = listOf("STATUS")
                    // Note this is non-null
                    val types = listOf(stringType)
                    Pair(fieldNames, types)
                }

                // Describe Queries
                SqlKind.DESCRIBE_TABLE -> {
                    // We only return the first 7 arguments from Snowflake right now as other expressions may not generalize.
                    val fieldNames = listOf("NAME", "TYPE", "KIND", "NULL?", "DEFAULT", "PRIMARY_KEY", "UNIQUE_KEY")
                    val types = listOf(stringType, stringType, stringType, stringType, stringType, stringType, stringType)
                    Pair(fieldNames, types)
                }
                SqlKind.SHOW_OBJECTS -> {
                    val fieldNames =
                        listOf(
                            "CREATED_ON",
                            "NAME",
                            "SCHEMA_NAME",
                            "KIND",
                        )
                    // TODO: created_on type from Snowflake is TIMESTAMP_LTZ
                    val types =
                        listOf(
                            stringType,
                            stringType,
                            stringType,
                            stringType,
                        )
                    Pair(fieldNames, types)
                }
                SqlKind.SHOW_SCHEMAS -> {
                    val fieldNames =
                        listOf(
                            "CREATED_ON",
                            "NAME",
                            "SCHEMA_NAME",
                            "KIND",
                        )
                    // TODO: created_on type from Snowflake is TIMESTAMP_LTZ
                    val types =
                        listOf(
                            stringType,
                            stringType,
                            stringType,
                            stringType,
                        )
                    Pair(fieldNames, types)
                }
                else -> throw UnsupportedOperationException("Unsupported DDL operation: ${ddlNode.kind}")
            }
        return typeFactory.createStructType(columnTypes, fieldsNames)
    }
}
