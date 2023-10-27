package org.apache.calcite.sql.validate.implicit

import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.operatorTables.ThreeOperatorStringTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
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
        ).mapKeys { it.key.name }
    }
}
