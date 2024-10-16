package com.bodosql.calcite.ddl

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.type.SqlTypeName

/**
 * Represents the result of a DDL operation. The maps to a columnar DataFrame result.
 * TODO: Replace columnTypes list of strings with the SqlType Enum values.
 */
data class DDLExecutionResult(
    val columnNames: List<String>,
    val columnValues: List<List<Any?>>,
    val columnTypes: List<String>,
) {
    companion object {
        /**
         * Converts a RelDataType row to a list of strings representing the column types
         * that can be called from Python to the correct pandas type to execute after calling
         * exec in Python.
         */
        @JvmStatic
        fun toPandasDataTypeList(rowType: RelDataType): List<String> {
            val columnTypes = mutableListOf<String>()
            rowType.fieldList.forEach {
                val typeString =
                    if (SqlTypeName.CHAR_TYPES.contains(it.type.sqlTypeName)) {
                        "pd.ArrowDtype(pa.string())"
                    } else if (it.type.sqlTypeName == SqlTypeName.DECIMAL) {
                        "pd.ArrowDtype(pa.decimal128(${it.type.precision}, ${it.type.scale}))"
                    } else {
                        throw RuntimeException("Unsupported DDL return type: ${it.type.sqlTypeName}")
                    }
                columnTypes.add(typeString)
            }
            return columnTypes
        }
    }
}
