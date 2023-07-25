package com.bodosql.calcite.prepare

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.BodoSQLOperatorTables.DatetimeOperatorTable
import com.bodosql.calcite.rex.RexNamedParam
import com.bodosql.calcite.sql.`fun`.SqlBodoOperatorTable
import com.bodosql.calcite.sql.`fun`.SqlLikeQuantifyOperator
import com.bodosql.calcite.sql.`fun`.SqlNamedParameterOperator
import com.google.common.collect.ImmutableList
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.*
import org.apache.calcite.sql.`fun`.SqlLibraryOperators
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.validate.SqlValidatorImpl
import org.apache.calcite.sql2rel.*
import org.checkerframework.checker.initialization.qual.UnknownInitialization

/**
 * Custom convertlet table for Bodo code generation. Handles custom functions
 * along with overriding the standard behavior for certain functions that can
 * be handled natively.
 */
class BodoConvertletTable(config: StandardConvertletTableConfig) : StandardConvertletTable(config) {
    init {
        registerOp(SqlBodoOperatorTable.TRY_CAST, this::convertTryCast)
        registerOp(SqlNamedParameterOperator.INSTANCE, this::convertNamedParam)
    }

    constructor() : this(StandardConvertletTableConfig(true, true))

    private fun convertTryCast(cx: SqlRexContext, call: SqlCall): RexNode {
        val typeFactory = cx.typeFactory
        assert(call.kind == SqlKind.TRY_CAST)
        val left = call.operand<SqlNode>(0)
        val right = call.operand<SqlNode>(1)
        val dataType = right as SqlDataTypeSpec
        var type = dataType.deriveType(cx.validator)
        if (type == null) {
            type = cx.validator.getValidatedNodeType(dataType.typeName)
        }
        val arg = cx.convertExpression(left)
        if (arg.type.isNullable) {
            type = typeFactory.createTypeWithNullability(type, true)
        }
        if (SqlUtil.isNullLiteral(left, false)) {
            val validator = cx.validator as SqlValidatorImpl
            validator.setValidatedNodeType(left, type)
            return cx.convertExpression(left)
        }
        return cx.rexBuilder.makeCall(type, SqlBodoOperatorTable.TRY_CAST, ImmutableList.of(arg))
    }

    private fun convertNamedParam(cx: SqlRexContext, call: SqlCall): RexNode {
        val name = call.operand<SqlLiteral>(0).getValueAs(String::class.java)
            .trimStart('$', '@')
        val returnType = cx.validator.getValidatedNodeType(call)
        return RexNamedParam(returnType, name)
    }

    override fun get(call: SqlCall): SqlRexConvertlet? {
        return when (call.kind) {
            // LEAST and GREATEST default to expanding into case statements
            // in the standard convertlet table. We natively support these
            // operations so avoid converting them to another pattern.
            SqlKind.LEAST, SqlKind.GREATEST -> AliasConverter
            SqlKind.LIKE -> if (call.operator is SqlLikeQuantifyOperator)
                LikeQuantifyConverter else super.get(call)

            SqlKind.OTHER_FUNCTION -> {
                when (call.operator.name) {
                    "TRUNC" -> DateTruncConverter
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
            val likeOp = if (op.caseSensitive)
                // Case insensitive is not standard SQL so it's in the library operators.
                SqlStdOperatorTable.LIKE else SqlLibraryOperators.ILIKE

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
