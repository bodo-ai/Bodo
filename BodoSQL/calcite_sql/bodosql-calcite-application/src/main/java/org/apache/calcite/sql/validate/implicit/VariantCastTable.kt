package org.apache.calcite.sql.validate.implicit

import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.operatorTables.ThreeOperatorStringTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.`fun`.SqlLibraryOperators
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeName

internal class VariantCastTable {

    companion object {
        private val anyArgBooleanCast = {
                inType: RelDataType, factory: RelDataTypeFactory, _: Int ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.BOOLEAN),
                inType.isNullable,
            )
        }

        private val anyArgVarcharCast = {
                inType: RelDataType, factory: RelDataTypeFactory, _: Int ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.VARCHAR),
                inType.isNullable,
            )
        }

        private val anyArgIntegerCast = {
                inType: RelDataType, factory: RelDataTypeFactory, _: Int ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.INTEGER),
                inType.isNullable,
            )
        }

        private val anyArgDoubleCast = {
                inType: RelDataType, factory: RelDataTypeFactory, _: Int ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.DOUBLE),
                inType.isNullable,
            )
        }

        private val padCasting = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            val typeName = if (idx == 1) {
                SqlTypeName.INTEGER
            } else {
                SqlTypeName.VARCHAR
            }
            factory.createTypeWithNullability(
                factory.createSqlType(typeName),
                inType.isNullable,
            )
        }

        /**
         * Generate the new types for the given argument for any function that
         * has some arguments which need to be cast to string and other
         * arguments that need to be cast to the same numeric type (integer, bigInt, etc).
         *
         * @param inType: The original input type. This is returned if the argument number
         * is beyond the argument limit and is used to determine nullability.
         * @param factory: The factory used to create the output data type.
         * @param idx: The argument of the input to replace. This is used to determine which type
         * should be output.
         * @param argLimit: The maximum number of expected arguments for a function's signature.
         * If any index is passed in that suggests more arguments than accounted for then we do not cast
         * the variant type at all to avoid incorrect undefined behavior.
         * @param stringIndices A set of argument numbers that map to string types. All other valid arguments
         * map to the numeric type.
         * @param numericTypeName The typename for the numeric type bing cast to. All numeric arguments must cast
         * to the same type for this function
         *
         * @return The output data type that should be produced for the given variant argument.
         */
        private fun varcharNumericHelper(inType: RelDataType, factory: RelDataTypeFactory, idx: Int, argLimit: Int, stringIndices: Set<Int>, numericTypeName: SqlTypeName): RelDataType {
            if (idx >= argLimit) {
                return inType
            }
            val typeName = if (stringIndices.contains(idx)) {
                SqlTypeName.VARCHAR
            } else {
                numericTypeName
            }
            return factory.createTypeWithNullability(
                factory.createSqlType(typeName),
                inType.isNullable,
            )
        }

        /**
         * Casting for 2 argument functions where the first argument is varchar and the last
         * is tinyInt.
         */
        private val varcharTinyintCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 2, setOf(0), SqlTypeName.TINYINT)
        }

        /**
         * Casting for 2 argument functions where the first argument is varchar and the last
         * is integer.
         */
        private val varcharIntegerCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 2, setOf(0), SqlTypeName.INTEGER)
        }

        /**
         * Casting for 3 argument functions where the first 2 arguments are varchar and the last
         * is integer.
         */
        private val varcharVarcharIntegerCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 3, setOf(0, 1), SqlTypeName.INTEGER)
        }

        /**
         * Casting for 3 argument functions where the first arguments is varchar and the last
         * two are BIGINT.
         */
        private val varcharBigintBigintCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 3, setOf(0), SqlTypeName.BIGINT)
        }

        /**
         * Function that casts variant to string if they are found in the allowed indices
         * and just returns variant for all other arguments.
         */
        private val onlyStringCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int, stringIndices: Set<Int> ->
            if (stringIndices.contains(idx)) {
                factory.createTypeWithNullability(
                    factory.createSqlType(SqlTypeName.VARCHAR),
                    inType.isNullable,
                )
            } else {
                inType
            }
        }

        /**
         * Cast for functions that change arg0 to string but other variant
         * casting is not supported (either because it should never be supported
         * or we don't support it yet).
         */
        private val arg0VarcharCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            onlyStringCast(inType, factory, idx, setOf(0))
        }

        /**
         * Cast for insert, for which we currently only support casting the string
         * arguments.
         *
         * Insert casts to of a function signature
         * (VARCHAR, INTEGER, INTEGER, VARCHAR), but Snowflake's variant casting
         * for the integers does VARIANT:DOUBLE:INTEGER, so we don't support this yet
         * and only enable the VARCHAR
         */
        private val insertCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            onlyStringCast(inType, factory, idx, setOf(0, 3))
        }

        /**
         * Cast for (VARCHAR, VARCHAR, BIGINT, VARCHAR).
         */
        private val varcharVarcharBigIntVarcharCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 4, setOf(0, 1, 3), SqlTypeName.BIGINT)
        }

        /**
         * Cast for REGEXP_SUBSTR.
         *
         * REGEXP_SUBSTR casts to a function signature
         * (VARCHAR, VARCHAR, BIGINT, BIGINT, VARCHAR, BIGINT)
         */
        private val regexpSubstrCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 6, setOf(0, 1, 4), SqlTypeName.BIGINT)
        }

        /**
         * Cast for REGEXP_REPLACE.
         *
         * REGEXP_REPLACE casts to a function signature
         * (VARCHAR, VARCHAR, VARCHAR, BIGINT, BIGINT, VARCHAR)
         */
        private val regexpReplaceCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 6, setOf(0, 1, 2, 5), SqlTypeName.BIGINT)
        }

        /**
         * Cast for REGEXP_INSTR.
         *
         * REGEXP_INSTR casts to a function signature
         * (VARCHAR, VARCHAR, BIGINT, BIGINT, BIGINT, VARCHAR, BIGINT)
         */
        private val regexpInstrCast = {
                inType: RelDataType, factory: RelDataTypeFactory, idx: Int ->
            varcharNumericHelper(inType, factory, idx, 7, setOf(0, 1, 5), SqlTypeName.BIGINT)
        }

        /**
         * Mapping of function names to a lambda function used to derive the default variant type.
         *
         * TODO: Ensure these lists are exhaustive.
         */
        val variantNameMapping = mapOf(
            SqlStdOperatorTable.AND to anyArgBooleanCast,
            SqlStdOperatorTable.OR to anyArgBooleanCast,
            SqlStdOperatorTable.NOT to anyArgBooleanCast,
            SqlStdOperatorTable.LOWER to anyArgVarcharCast,
            SqlStdOperatorTable.UPPER to anyArgVarcharCast,
            SqlStdOperatorTable.TRIM to anyArgVarcharCast,
            SqlStdOperatorTable.ASCII to anyArgVarcharCast,
            SqlStdOperatorTable.CHAR_LENGTH to anyArgVarcharCast,
            SqlStdOperatorTable.CHARACTER_LENGTH to anyArgVarcharCast,
            StringOperatorTable.LEN to anyArgVarcharCast,
            StringOperatorTable.CONCAT_WS to anyArgVarcharCast,
            StringOperatorTable.CONCAT to anyArgVarcharCast,
            SqlStdOperatorTable.CONCAT to anyArgVarcharCast,
            StringOperatorTable.LENGTH to anyArgVarcharCast,
            StringOperatorTable.LTRIM to anyArgVarcharCast,
            StringOperatorTable.REVERSE to anyArgVarcharCast,
            StringOperatorTable.RTRIM to anyArgVarcharCast,
            StringOperatorTable.RTRIMMED_LENGTH to anyArgVarcharCast,
            StringOperatorTable.INITCAP to anyArgVarcharCast,
            StringOperatorTable.MD5 to anyArgVarcharCast,
            StringOperatorTable.MD5_HEX to anyArgVarcharCast,
            StringOperatorTable.STARTSWITH to anyArgVarcharCast,
            StringOperatorTable.ENDSWITH to anyArgVarcharCast,
            ThreeOperatorStringTable.LPAD to padCasting,
            ThreeOperatorStringTable.RPAD to padCasting,
            SqlStdOperatorTable.SUM to anyArgDoubleCast,
            SqlStdOperatorTable.SUM0 to anyArgDoubleCast,
            SqlStdOperatorTable.PLUS to anyArgDoubleCast,
            SqlStdOperatorTable.MINUS to anyArgDoubleCast,
            SqlStdOperatorTable.MULTIPLY to anyArgDoubleCast,
            SqlStdOperatorTable.DIVIDE to anyArgDoubleCast,
            SqlStdOperatorTable.ACOS to anyArgDoubleCast,
            NumericOperatorTable.ACOSH to anyArgDoubleCast,
            SqlStdOperatorTable.ASIN to anyArgDoubleCast,
            NumericOperatorTable.ASINH to anyArgDoubleCast,
            SqlStdOperatorTable.ATAN to anyArgDoubleCast,
            SqlStdOperatorTable.ATAN2 to anyArgDoubleCast,
            NumericOperatorTable.ATANH to anyArgDoubleCast,
            SqlStdOperatorTable.CBRT to anyArgDoubleCast,
            SqlStdOperatorTable.COS to anyArgDoubleCast,
            NumericOperatorTable.COSH to anyArgDoubleCast,
            SqlStdOperatorTable.COT to anyArgDoubleCast,
            SqlStdOperatorTable.DEGREES to anyArgDoubleCast,
            NumericOperatorTable.DIV0 to anyArgDoubleCast,
            SqlStdOperatorTable.EXP to anyArgDoubleCast,
            NumericOperatorTable.HAVERSINE to anyArgDoubleCast,
            SqlStdOperatorTable.LN to anyArgDoubleCast,
            NumericOperatorTable.LOG to anyArgDoubleCast,
            NumericOperatorTable.POW to anyArgDoubleCast,
            SqlStdOperatorTable.POWER to anyArgDoubleCast,
            SqlStdOperatorTable.RADIANS to anyArgDoubleCast,
            CondOperatorTable.REGR_VALX to anyArgDoubleCast,
            CondOperatorTable.REGR_VALY to anyArgDoubleCast,
            SqlStdOperatorTable.SIGN to anyArgDoubleCast,
            SqlStdOperatorTable.SIN to anyArgDoubleCast,
            NumericOperatorTable.SINH to anyArgDoubleCast,
            SqlStdOperatorTable.SQRT to anyArgDoubleCast,
            NumericOperatorTable.SQUARE to anyArgDoubleCast,
            SqlStdOperatorTable.TAN to anyArgDoubleCast,
            NumericOperatorTable.TANH to anyArgDoubleCast,
            SqlStdOperatorTable.ABS to anyArgDoubleCast,
            SqlStdOperatorTable.AVG to anyArgDoubleCast,
            NumericOperatorTable.CORR to anyArgDoubleCast,
            SqlStdOperatorTable.COVAR_POP to anyArgDoubleCast,
            SqlStdOperatorTable.COVAR_SAMP to anyArgDoubleCast,
            NumericOperatorTable.KURTOSIS to anyArgDoubleCast,
            NumericOperatorTable.RATIO_TO_REPORT to anyArgDoubleCast,
            NumericOperatorTable.SKEW to anyArgDoubleCast,
            SqlStdOperatorTable.STDDEV to anyArgDoubleCast,
            SqlStdOperatorTable.STDDEV_POP to anyArgDoubleCast,
            SqlStdOperatorTable.STDDEV_SAMP to anyArgDoubleCast,
            SqlStdOperatorTable.VARIANCE to anyArgDoubleCast,
            SqlStdOperatorTable.VAR_SAMP to anyArgDoubleCast,
            SqlStdOperatorTable.VAR_POP to anyArgDoubleCast,
            NumericOperatorTable.VARIANCE_POP to anyArgDoubleCast,
            NumericOperatorTable.VARIANCE_SAMP to anyArgDoubleCast,
            CondOperatorTable.ZEROIFNULL to anyArgDoubleCast,
            StringOperatorTable.CHAR to anyArgIntegerCast,
            StringOperatorTable.CHR to anyArgIntegerCast,
            StringOperatorTable.SPACE to anyArgIntegerCast,
            StringOperatorTable.SPLIT to anyArgVarcharCast,
            StringOperatorTable.STRTOK to varcharVarcharIntegerCast,
            StringOperatorTable.STRTOK_TO_ARRAY to anyArgVarcharCast,
            StringOperatorTable.SPLIT_PART to varcharVarcharIntegerCast,
            SqlStdOperatorTable.LIKE to anyArgVarcharCast,
            SqlStdOperatorTable.NOT_LIKE to anyArgVarcharCast,
            SqlLibraryOperators.ILIKE to anyArgVarcharCast,
            SqlLibraryOperators.NOT_ILIKE to anyArgVarcharCast,
            SqlLibraryOperators.RLIKE to anyArgVarcharCast,
            SqlLibraryOperators.NOT_RLIKE to anyArgVarcharCast,
            SqlBodoOperatorTable.ANY_LIKE to arg0VarcharCast,
            SqlBodoOperatorTable.ANY_ILIKE to arg0VarcharCast,
            SqlBodoOperatorTable.ALL_LIKE to arg0VarcharCast,
            SqlBodoOperatorTable.ALL_ILIKE to arg0VarcharCast,
            StringOperatorTable.SHA2 to arg0VarcharCast,
            StringOperatorTable.SHA2_HEX to arg0VarcharCast,
            StringOperatorTable.HEX_DECODE_STRING to arg0VarcharCast,
            StringOperatorTable.HEX_DECODE_BINARY to arg0VarcharCast,
            StringOperatorTable.TRY_HEX_DECODE_STRING to arg0VarcharCast,
            StringOperatorTable.TRY_HEX_DECODE_BINARY to arg0VarcharCast,
            StringOperatorTable.BASE64_ENCODE to arg0VarcharCast,
            StringOperatorTable.BASE64_DECODE_STRING to arg0VarcharCast,
            StringOperatorTable.BASE64_DECODE_BINARY to arg0VarcharCast,
            StringOperatorTable.TRY_BASE64_DECODE_STRING to arg0VarcharCast,
            StringOperatorTable.TRY_BASE64_DECODE_BINARY to arg0VarcharCast,
            StringOperatorTable.HEX_ENCODE to varcharTinyintCast,
            StringOperatorTable.LEFT to varcharIntegerCast,
            StringOperatorTable.RIGHT to varcharIntegerCast,
            StringOperatorTable.CONTAINS to anyArgVarcharCast,
            StringOperatorTable.JAROWINKLER_SIMILARITY to anyArgVarcharCast,
            SqlLibraryOperators.TRANSLATE3 to anyArgVarcharCast,
            StringOperatorTable.CHARINDEX to varcharVarcharIntegerCast,
            StringOperatorTable.EDITDISTANCE to varcharVarcharIntegerCast,
            StringOperatorTable.POSITION to varcharVarcharIntegerCast,
            SqlStdOperatorTable.REPLACE to anyArgVarcharCast,
            SqlStdOperatorTable.SUBSTRING to varcharBigintBigintCast,
            StringOperatorTable.SUBSTR to varcharBigintBigintCast,
            StringOperatorTable.INSERT to insertCast,
            StringOperatorTable.REGEXP_LIKE to anyArgVarcharCast,
            StringOperatorTable.REGEXP_COUNT to varcharVarcharBigIntVarcharCast,
            StringOperatorTable.REGEXP_SUBSTR to regexpSubstrCast,
            StringOperatorTable.REGEXP_INSTR to regexpInstrCast,
            StringOperatorTable.REGEXP_REPLACE to regexpReplaceCast,
        ).mapKeys { it.key.name }
    }
}
