package com.bodosql.calcite.sql.validate

import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.type.ArraySqlType
import org.apache.calcite.sql.type.MapSqlType
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.TZAwareSqlType
import org.apache.calcite.sql.type.VariantSqlType
import java.lang.RuntimeException

/**
 * Coercion utils based upon an investigation into Snowflake UDFs:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1565098005/Snowflake+UDF+Overload+Investigation#Conclusions.1
 */
class BodoCoercionUtil {

    companion object {

        /**
         * Determine if the input type can be used for the target type. If coerce=true
         * then we can use implicit casts to reach this result. This is the equivalent
         * to SqlTypeUtil.canCastFrom except with UDF restrictions.
         */
        fun canCastFromUDF(input: RelDataType, target: RelDataType, coerce: Boolean): Boolean {
            return if (coerce) {
                val (type, _) = coercableInUDF(input, target)
                return type
            } else {
                assignableInUDF(input, target)
            }
        }

        /**
         * Gets the casting score for a particular argument
         * for computing a signature score. If a cast is not possible it will
         * return -1, although we expect callers to only uses this function
         * after validating casting is possible.
         *
         * Each score is relative to the source type. The full type ordering can be found here:
         * https://bodo.atlassian.net/wiki/spaces/B/pages/1565098005/Snowflake+UDF+Overload+Investigation#Conclusions.1
         */
        fun getCastingMatchScore(input: RelDataType, target: RelDataType): Int {
            val (_, score) = coercableInUDF(input, target)
            return score
        }

        /**
         * Determine if the input type can be assigned to target without a cast in a
         * UDF. There is no need to cast for nullability or differing precision
         * in a UDF.
         */
        private fun assignableInUDF(input: RelDataType, target: RelDataType): Boolean {
            if (target is ArraySqlType) {
                // Array
                return input is ArraySqlType
            } else if (SqlTypeFamily.BINARY.contains(target)) {
                // Binary
                return SqlTypeFamily.BINARY.contains(input)
            } else if (SqlTypeFamily.BOOLEAN.contains(target)) {
                // Boolean
                return SqlTypeFamily.BOOLEAN.contains(input)
            } else if (SqlTypeFamily.DATE.contains(target)) {
                // Date
                return SqlTypeFamily.DATE.contains(input)
            } else if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(target)) {
                // Float/Double
                return SqlTypeFamily.APPROXIMATE_NUMERIC.contains(input)
            } else if (SqlTypeFamily.EXACT_NUMERIC.contains(target)) {
                // Number
                return SqlTypeFamily.EXACT_NUMERIC.contains(input)
            } else if (target is MapSqlType) {
                // Object
                return input is MapSqlType
            } else if (SqlTypeFamily.TIME.contains(target)) {
                // Time
                return SqlTypeFamily.TIME.contains(input)
            } else if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                // Timestamp_NTZ. Note we must use typename here because LTZ
                // conversion should require a cast.
                return input.sqlTypeName == SqlTypeName.TIMESTAMP
            } else if (target is TZAwareSqlType) {
                // Timestamp_LTZ
                return input is TZAwareSqlType
            } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                // Varchar
                return SqlTypeFamily.CHARACTER.contains(input)
            } else if (target is VariantSqlType) {
                // Variant
                return input is VariantSqlType
            } else {
                return false
            }
        }

        /**
         * Determines if input can be converted to a target either because the types are assignable
         * or through an implicit cast. This function returns two values:
         * - If the cast is possible.
         * - The "score" or "distance" for the particular cast, which is the simple type precedence
         *   for casting input to target. If the cast is not possible this is -1 and if no cast is required
         *   the result is 0. Each score is relative to the source type. The full type ordering can be found here:
         *   https://bodo.atlassian.net/wiki/spaces/B/pages/1565098005/Snowflake+UDF+Overload+Investigation#Conclusions.1
         *
         * Note: Since we are trying to find the "best match" for the input type, this section is order by input
         * or "source type" and the score is computed by checking the target.
         */
        private fun coercableInUDF(input: RelDataType, target: RelDataType): Pair<Boolean, Int> {
            if (assignableInUDF(input, target)) {
                return Pair(true, 0)
            } else if (input is ArraySqlType) {
                // Array
                if (target is VariantSqlType) {
                    return Pair(true, 1)
                }
            } else if (SqlTypeFamily.BOOLEAN.contains(input)) {
                // Boolean
                if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 1)
                } else if (target is VariantSqlType) {
                    return Pair(true, 2)
                }
            } else if (SqlTypeFamily.DATE.contains(input)) {
                // Date
                if (target is TZAwareSqlType) {
                    return Pair(true, 1)
                } else if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                    return Pair(true, 2)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 3)
                } else if (target is VariantSqlType) {
                    return Pair(true, 4)
                }
            } else if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(input)) {
                // Float/Double
                if (SqlTypeFamily.BOOLEAN.contains(target)) {
                    return Pair(true, 1)
                } else if (target is VariantSqlType) {
                    return Pair(true, 2)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 3)
                } else if (SqlTypeFamily.EXACT_NUMERIC.contains(target)) {
                    return Pair(true, 4)
                }
            } else if (SqlTypeFamily.EXACT_NUMERIC.contains(input)) {
                // Number
                if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(target)) {
                    return Pair(true, 1)
                } else if (SqlTypeFamily.BOOLEAN.contains(target)) {
                    return Pair(true, 2)
                } else if (target is VariantSqlType) {
                    return Pair(true, 3)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 4)
                }
            } else if (input is MapSqlType) {
                // Object
                if (target is VariantSqlType) {
                    return Pair(true, 1)
                }
            } else if (SqlTypeFamily.TIME.contains(input)) {
                // Time
                if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 1)
                }
            } else if (input.sqlTypeName == SqlTypeName.TIMESTAMP) {
                // Timestamp_NTZ. Note we must use typename here because LTZ
                // conversion should require a cast.
                if (target is TZAwareSqlType) {
                    return Pair(true, 1)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 2)
                } else if (SqlTypeFamily.DATE.contains(target)) {
                    return Pair(true, 3)
                } else if (SqlTypeFamily.TIME.contains(target)) {
                    return Pair(true, 4)
                } else if (target is VariantSqlType) {
                    return Pair(true, 5)
                }
            } else if (input is TZAwareSqlType) {
                // Timestamp_LTZ
                if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                    return Pair(true, 1)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 2)
                } else if (SqlTypeFamily.DATE.contains(target)) {
                    return Pair(true, 3)
                } else if (SqlTypeFamily.TIME.contains(target)) {
                    return Pair(true, 4)
                } else if (target is VariantSqlType) {
                    return Pair(true, 5)
                }
            } else if (SqlTypeFamily.CHARACTER.contains(input)) {
                // Varchar
                if (SqlTypeFamily.BOOLEAN.contains(target)) {
                    return Pair(true, 1)
                } else if (SqlTypeFamily.DATE.contains(target)) {
                    return Pair(true, 2)
                } else if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(target)) {
                    return Pair(true, 3)
                } else if (target is TZAwareSqlType) {
                    return Pair(true, 4)
                } else if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                    return Pair(true, 5)
                } else if (SqlTypeFamily.EXACT_NUMERIC.contains(target)) {
                    return Pair(true, 6)
                } else if (SqlTypeFamily.TIME.contains(target)) {
                    return Pair(true, 7)
                } else if (target is VariantSqlType) {
                    return Pair(true, 8)
                }
            } else if (input is VariantSqlType) {
                // Variant
                if (target is ArraySqlType) {
                    return Pair(true, 1)
                } else if (SqlTypeFamily.BOOLEAN.contains(target)) {
                    return Pair(true, 2)
                } else if (target is MapSqlType) {
                    return Pair(true, 3)
                } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                    return Pair(true, 4)
                } else if (SqlTypeFamily.DATE.contains(target)) {
                    return Pair(true, 5)
                } else if (SqlTypeFamily.TIME.contains(target)) {
                    return Pair(true, 6)
                } else if (target is TZAwareSqlType) {
                    return Pair(true, 7)
                } else if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                    return Pair(true, 8)
                } else if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(target)) {
                    return Pair(true, 9)
                } else if (SqlTypeFamily.EXACT_NUMERIC.contains(target)) {
                    return Pair(true, 10)
                }
            }
            // Fall through to all cases without a match.
            return Pair(false, -1)
        }

        fun getCastFunction(target: RelDataType): SqlOperator {
            if (target is ArraySqlType) {
                // Array
                return CastingOperatorTable.TO_ARRAY
            } else if (SqlTypeFamily.BOOLEAN.contains(target)) {
                // Boolean
                return CastingOperatorTable.TO_BOOLEAN
            } else if (SqlTypeFamily.DATE.contains(target)) {
                // Date
                return CastingOperatorTable.TO_DATE
            } else if (SqlTypeFamily.APPROXIMATE_NUMERIC.contains(target)) {
                // Float/Double
                return CastingOperatorTable.TO_DOUBLE
            } else if (SqlTypeFamily.EXACT_NUMERIC.contains(target)) {
                // Number
                return CastingOperatorTable.TO_NUMBER
            } else if (target is MapSqlType) {
                // Object
                return CastingOperatorTable.TO_OBJECT
            } else if (SqlTypeFamily.TIME.contains(target)) {
                // Time
                return CastingOperatorTable.TO_TIME
            } else if (target.sqlTypeName == SqlTypeName.TIMESTAMP) {
                // Timestamp_NTZ
                return CastingOperatorTable.TO_TIMESTAMP_NTZ
            } else if (target is TZAwareSqlType) {
                // Timestamp_LTZ
                return CastingOperatorTable.TO_TIMESTAMP_LTZ
            } else if (SqlTypeFamily.CHARACTER.contains(target)) {
                // Varchar
                return CastingOperatorTable.TO_VARCHAR
            } else if (target is VariantSqlType) {
                // Variant
                return CastingOperatorTable.TO_VARIANT
            } else {
                // Fall through to types that don't support implicit casting (e.g. Binary)
                throw RuntimeException("No implicit casting is allowed for target type")
            }
        }
    }
}
