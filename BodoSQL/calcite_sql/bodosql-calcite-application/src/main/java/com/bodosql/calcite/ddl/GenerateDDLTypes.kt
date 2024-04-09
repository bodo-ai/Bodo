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
        when (ddlNode.kind) {
            SqlKind.DROP_TABLE -> {
                val fieldNames = listOf("STATUS")
                // Note this is non-null
                val types = listOf(typeFactory.createSqlType(SqlTypeName.VARCHAR))
                return typeFactory.createStructType(types, fieldNames)
            }
            else -> throw UnsupportedOperationException("Unsupported DDL operation: ${ddlNode.kind}")
        }
    }
}
