package com.bodosql.calcite.prepare

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.rex.RexNamedParam
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import com.bodosql.calcite.sql.func.SqlLikeQuantifyOperator
import com.bodosql.calcite.sql.func.SqlNamedParameterOperator
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.`fun`.SqlLibraryOperators
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql2rel.SqlRexContext
import org.apache.calcite.sql2rel.SqlRexConvertlet
import org.apache.calcite.sql2rel.StandardConvertletTable
import org.apache.calcite.sql2rel.StandardConvertletTableConfig

/**
 * Custom convertlet table for Bodo code generation. Handles custom functions
 * along with overriding the standard behavior for certain functions that can
 * be handled natively.
 */
class BodoConvertletTable(config: StandardConvertletTableConfig) : StandardConvertletTable(config) {
    init {
        registerOp(SqlNamedParameterOperator.INSTANCE, this::convertNamedParam)
        addAlias(CastingOperatorTable.TO_NUMERIC, CastingOperatorTable.TO_NUMBER)
        addAlias(CastingOperatorTable.TO_DECIMAL, CastingOperatorTable.TO_NUMBER)
        addAlias(CastingOperatorTable.TRY_TO_NUMERIC, CastingOperatorTable.TRY_TO_NUMBER)
        addAlias(CastingOperatorTable.TRY_TO_DECIMAL, CastingOperatorTable.TRY_TO_NUMBER)
        addAlias(CastingOperatorTable.DATE, CastingOperatorTable.TO_DATE)
        addAlias(CastingOperatorTable.TIME, CastingOperatorTable.TO_TIME)
        addAlias(CastingOperatorTable.TO_CHAR, CastingOperatorTable.TO_VARCHAR)
        addAlias(DatetimeOperatorTable.DATEFROMPARTS, DatetimeOperatorTable.DATE_FROM_PARTS)
        addAlias(DatetimeOperatorTable.TIMEFROMPARTS, DatetimeOperatorTable.TIME_FROM_PARTS)
        addAlias(DatetimeOperatorTable.TIMESTAMPFROMPARTS, DatetimeOperatorTable.TIMESTAMP_FROM_PARTS)
        addAlias(DatetimeOperatorTable.TIMESTAMPNTZFROMPARTS, DatetimeOperatorTable.TIMESTAMP_NTZ_FROM_PARTS)
        addAlias(DatetimeOperatorTable.TIMESTAMPLTZFROMPARTS, DatetimeOperatorTable.TIMESTAMP_LTZ_FROM_PARTS)
        addAlias(DatetimeOperatorTable.TIMESTAMPTZFROMPARTS, DatetimeOperatorTable.TIMESTAMP_TZ_FROM_PARTS)
        addAlias(StringOperatorTable.LEN, StringOperatorTable.LENGTH)
        addAlias(StringOperatorTable.SHA2_HEX, StringOperatorTable.SHA2)
        addAlias(StringOperatorTable.MD5_HEX, StringOperatorTable.MD5)
        addAlias(CondOperatorTable.IF_FUNC, CondOperatorTable.IFF_FUNC)
        addAlias(CondOperatorTable.NVL, SqlStdOperatorTable.COALESCE)
        addAlias(CondOperatorTable.IFNULL_FUNC, SqlStdOperatorTable.COALESCE)
        addAlias(StringOperatorTable.CHR, StringOperatorTable.CHAR)
        addAlias(StringOperatorTable.ORD, SqlStdOperatorTable.ASCII)
        addAlias(StringOperatorTable.LCASE, SqlStdOperatorTable.LOWER)
        addAlias(StringOperatorTable.UCASE, SqlStdOperatorTable.UPPER)
        addAlias(DatetimeOperatorTable.SYSDATE, DatetimeOperatorTable.UTC_TIMESTAMP)
        addAlias(SqlStdOperatorTable.LOCALTIME, SqlStdOperatorTable.CURRENT_TIME)
        addAlias(DatetimeOperatorTable.CURDATE, SqlStdOperatorTable.CURRENT_DATE)
        addAlias(SqlBodoOperatorTable.CURRENT_TIMESTAMP, DatetimeOperatorTable.GETDATE)
        addAlias(DatetimeOperatorTable.NOW, DatetimeOperatorTable.GETDATE)
        addAlias(SqlBodoOperatorTable.LOCALTIMESTAMP, DatetimeOperatorTable.GETDATE)
        addAlias(DatetimeOperatorTable.SYSTIMESTAMP, DatetimeOperatorTable.GETDATE)
        addAlias(DatetimeOperatorTable.DATE_ADD, DatetimeOperatorTable.DATEADD)
        addAlias(DatetimeOperatorTable.TIMEADD, DatetimeOperatorTable.DATEADD)
        addAlias(SqlBodoOperatorTable.TIMESTAMP_ADD, DatetimeOperatorTable.DATEADD)
        addAlias(DatetimeOperatorTable.TIMEDIFF, DatetimeOperatorTable.DATEDIFF)
        addAlias(SqlBodoOperatorTable.TIMESTAMP_DIFF, DatetimeOperatorTable.DATEDIFF)
        addAlias(SqlStdOperatorTable.POSITION, StringOperatorTable.CHARINDEX)
        addAlias(StringOperatorTable.POSITION, StringOperatorTable.CHARINDEX)
        addAlias(StringOperatorTable.SUBSTR, SqlStdOperatorTable.SUBSTRING)
        addAlias(StringOperatorTable.MID, SqlStdOperatorTable.SUBSTRING)
        addAlias(SqlLibraryOperators.RLIKE, StringOperatorTable.REGEXP_LIKE)
        addAlias(StringOperatorTable.RLIKE, StringOperatorTable.REGEXP_LIKE)
        registerOp(StringOperatorTable.LENGTH, this::simpleConversion)
    }

    constructor() : this(StandardConvertletTableConfig(true, true))

    private fun convertNamedParam(cx: SqlRexContext, call: SqlCall): RexNode {
        val name = call.operand<SqlLiteral>(0).getValueAs(String::class.java)
            .trimStart('$', '@')
        val returnType = cx.validator.getValidatedNodeType(call)
        return RexNamedParam(returnType, name)
    }

    /**
     * Convert an operator from Sql To Rex direclty. This is used in case our convertlet
     * conflicts with Calcite.
     */
    private fun simpleConversion(cx: SqlRexContext, call: SqlCall): RexNode {
        val operands = call.operandList.map { op -> cx.convertExpression(op) }
        return cx.rexBuilder.makeCall(call.operator, operands)
    }

    override fun get(call: SqlCall): SqlRexConvertlet? {
        return when (call.kind) {
            // LEAST and GREATEST default to expanding into case statements
            // in the standard convertlet table. We natively support these
            // operations so avoid converting them to another pattern.
            SqlKind.LEAST, SqlKind.GREATEST -> AliasConverter
            SqlKind.LIKE -> if (call.operator is SqlLikeQuantifyOperator) {
                LikeQuantifyConverter
            } else {
                super.get(call)
            }

            SqlKind.OTHER_FUNCTION -> {
                when (call.operator.name) {
                    "TRUNC" -> DateTruncConverter
                    "NULLIFZERO" -> NullIfZeroConverter
                    else -> super.get(call)
                }
            }
            else -> super.get(call)
        }
    }

    private object DateTruncConverter : SqlRexConvertlet {
        override fun convertCall(cx: SqlRexContext, call: SqlCall): RexNode {
            val operands = call.operandList.map { op -> cx.convertExpression(op) }

            return when (cx.validator.getValidatedNodeType(call).family) {
                SqlTypeFamily.NUMERIC -> cx.rexBuilder.makeCall(SqlStdOperatorTable.TRUNCATE, operands)
                else -> cx.rexBuilder.makeCall(DatetimeOperatorTable.DATE_TRUNC, operands)
            }
        }
    }

    private object NullIfZeroConverter : SqlRexConvertlet {
        override fun convertCall(cx: SqlRexContext, call: SqlCall): RexNode {
            // Convert NULLIFZERO(a) into IFF(a = 0, NULL, a)
            val originalOperands = call.operandList.map { op -> cx.convertExpression(op) }
            val arg0 = originalOperands[0]
            val literalZero = cx.rexBuilder.makeZeroLiteral(arg0.type)
            val equalityCheck = cx.rexBuilder.makeCall(SqlStdOperatorTable.EQUALS, listOf(arg0, literalZero))
            // NULL returned in the "then" branch
            val nullValue = cx.rexBuilder.makeNullLiteral(arg0.type)
            return cx.rexBuilder.makeCall(CondOperatorTable.IFF_FUNC, listOf(equalityCheck, nullValue, arg0))
        }
    }
    private object AliasConverter : SqlRexConvertlet {
        override fun convertCall(cx: SqlRexContext, call: SqlCall): RexNode {
            val operands = call.operandList.map { op -> cx.convertExpression(op) }
            return cx.rexBuilder.makeCall(call.operator, operands)
        }
    }

    /**
     * Handles the conversion of LIKE ANY/ALL operators to the RexNode equivalent.
     *
     * At the moment, this convertlet only handles sets such as LIKE ANY ('%a%', '%b%')
     * and converts that to an equivalent AND/OR expression with a comparison
     * against each part of the set.
     *
     * It does not handle subqueries which are valid in this context. Subqueries
     * would require further support from the planner and decorrelator to support
     * and can't be done with a simple expansion transformation.
     */
    private object LikeQuantifyConverter : SqlRexConvertlet {
        override fun convertCall(cx: SqlRexContext, call: SqlCall): RexNode {
            assert(call.operandCount() >= 2)

            val op = call.operator as SqlLikeQuantifyOperator
            val likeOp = if (op.caseSensitive) {
                // Case insensitive is not standard SQL so it's in the library operators.
                SqlStdOperatorTable.LIKE
            } else {
                SqlLibraryOperators.ILIKE
            }

            val arg0 = cx.convertExpression(call.operandList[0])
            val arg2 = call.operandList.getOrNull(2)?.let { sqlNode -> cx.convertExpression(sqlNode) }
            val arg1 = when (val arg1 = call.operandList[1]) {
                is SqlNodeList -> arg1.list.map { n ->
                    val expr = cx.convertExpression(n)
                    if (arg2 != null) {
                        cx.rexBuilder.makeCall(likeOp, arg0, expr, arg2)
                    } else {
                        cx.rexBuilder.makeCall(likeOp, arg0, expr)
                    }
                }
                else -> throw BodoSQLCodegenException("Unsupported argument to $op: $arg1")
            }

            return when (op.comparisonKind) {
                SqlKind.SOME -> RexUtil.composeDisjunction(cx.rexBuilder, arg1)
                SqlKind.ALL -> RexUtil.composeConjunction(cx.rexBuilder, arg1)
                else -> throw IllegalStateException()
            }
        }
    }
}
