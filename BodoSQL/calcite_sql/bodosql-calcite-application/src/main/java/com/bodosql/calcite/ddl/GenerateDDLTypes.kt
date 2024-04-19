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
                SqlKind.DROP_TABLE -> {
                    val fieldNames = listOf("STATUS")
                    // Note this is non-null
                    val types = listOf(stringType)
                    Pair(fieldNames, types)
                }
                SqlKind.DESCRIBE_TABLE -> {
                    // We only return the first 7 arguments from Snowflake right now as other expressions may not generalize.
                    val fieldNames = listOf("NAME", "TYPE", "KIND", "NULL?", "DEFAULT", "PRIMARY_KEY", "UNIQUE_KEY")
                    val types = listOf(stringType, stringType, stringType, stringType, stringType, stringType, stringType)
                    Pair(fieldNames, types)
                }
                else -> throw UnsupportedOperationException("Unsupported DDL operation: ${ddlNode.kind}")
            }
        return typeFactory.createStructType(columnTypes, fieldsNames)
    }
}
