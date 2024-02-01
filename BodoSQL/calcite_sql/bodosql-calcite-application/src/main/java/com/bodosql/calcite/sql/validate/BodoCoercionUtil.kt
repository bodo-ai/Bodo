package com.bodosql.calcite.sql.validate

import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.TZAwareSqlType
import org.apache.calcite.sql.type.VariantSqlType
import java.lang.RuntimeException

class BodoCoercionUtil {

    companion object {
        data class CoercionTableEntry(val input: SqlTypeName, val validCoercions: List<SqlTypeFamily>, val canCoerceToVariant: Boolean)
        var coercionTable: MutableList<CoercionTableEntry> = mutableListOf(
            CoercionTableEntry(SqlTypeName.ARRAY, listOf(), true),
            CoercionTableEntry(SqlTypeName.BOOLEAN, listOf(SqlTypeFamily.STRING), true),
            CoercionTableEntry(SqlTypeName.DATE, listOf(SqlTypeFamily.TIMESTAMP, SqlTypeFamily.STRING), false),
            CoercionTableEntry(SqlTypeName.MAP, listOf(), true),
            CoercionTableEntry(SqlTypeName.TIME, listOf(SqlTypeFamily.STRING), false),
            CoercionTableEntry(SqlTypeName.TIMESTAMP, listOf(SqlTypeFamily.DATE, SqlTypeFamily.TIME, SqlTypeFamily.STRING), false),
            CoercionTableEntry(SqlTypeName.VARCHAR, listOf(SqlTypeFamily.BOOLEAN, SqlTypeFamily.DATE, SqlTypeFamily.NUMERIC, SqlTypeFamily.TIME, SqlTypeFamily.TIMESTAMP), false),
        )

        init {
            for (approxType in SqlTypeName.APPROX_TYPES) {
                coercionTable.add(CoercionTableEntry(approxType, listOf(SqlTypeFamily.BOOLEAN, SqlTypeFamily.NUMERIC, SqlTypeFamily.STRING), true))
            }

            for (approxType in SqlTypeName.EXACT_TYPES) {
                coercionTable.add(CoercionTableEntry(approxType, listOf(SqlTypeFamily.BOOLEAN, SqlTypeFamily.NUMERIC, SqlTypeFamily.TIMESTAMP, SqlTypeFamily.STRING), true))
            }
        }
        fun coercable(input: RelDataType, target: RelDataType): Boolean {
            if (input is VariantSqlType) {
                return true
            }

            val targetIsVariant = target is VariantSqlType

            for (row in BodoCoercionUtil.coercionTable) {
                if (row.input == input.sqlTypeName || (row.input == SqlTypeName.TIMESTAMP && input is TZAwareSqlType)) {
                    if (targetIsVariant) {
                        return row.canCoerceToVariant
                    }

                    for (validCoercion in row.validCoercions) {
                        if (validCoercion.contains(target)) {
                            return true
                        }
                    }
                }
            }
            return false
        }

        fun getCastFunction(target: RelDataType): SqlOperator {
            if (target is VariantSqlType) {
                return CastingOperatorTable.TO_VARIANT
            }

            if (SqlTypeFamily.STRING.contains(target)) {
                return CastingOperatorTable.TO_VARCHAR
            } else if (SqlTypeFamily.DATE.contains(target)) {
                return CastingOperatorTable.TO_DATE
            } else if (SqlTypeFamily.TIMESTAMP.contains(target)) {
                return CastingOperatorTable.TO_TIMESTAMP
            } else if (SqlTypeFamily.TIME.contains(target)) {
                return CastingOperatorTable.TO_TIME
            } else if (SqlTypeFamily.BOOLEAN.contains(target)) {
                return CastingOperatorTable.TO_BOOLEAN
            } else if (SqlTypeName.APPROX_TYPES.contains(target.sqlTypeName)) {
                return CastingOperatorTable.TO_DOUBLE
            } else if (SqlTypeName.EXACT_TYPES.contains(target.sqlTypeName)) {
                return CastingOperatorTable.TO_NUMBER
            }

            throw RuntimeException("Unexpected target type")
        }
    }
}
