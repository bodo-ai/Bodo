package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.ir.Expr
import com.google.common.collect.Range
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexCorrelVariable
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexFieldAccess
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexPatternFieldRef
import org.apache.calcite.rex.RexRangeRef
import org.apache.calcite.rex.RexSubQuery
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.rex.RexVisitor
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.Sarg

typealias Filter = List<Triple<String, String, Expr>>

private val BASIC_UNARY_CALL_TO_OPSTR =
    mapOf(
        SqlKind.IS_NULL to ("is" to Expr.StringLiteral("NULL")),
        SqlKind.IS_NOT_NULL to ("is not" to Expr.StringLiteral("NULL")),
        SqlKind.IS_TRUE to ("==" to Expr.BooleanLiteral(true)),
        SqlKind.IS_NOT_TRUE to ("!=" to Expr.BooleanLiteral(true)),
        SqlKind.IS_FALSE to ("==" to Expr.BooleanLiteral(false)),
        SqlKind.IS_NOT_FALSE to ("!=" to Expr.BooleanLiteral(false)),
    )

private val BASIC_BIN_CALL_TO_OPSTR =
    mapOf(
        SqlKind.EQUALS to "==",
        SqlKind.NOT_EQUALS to "!=",
        SqlKind.LESS_THAN to "<",
        SqlKind.LESS_THAN_OR_EQUAL to "<=",
        SqlKind.GREATER_THAN to ">",
        SqlKind.GREATER_THAN_OR_EQUAL to ">=",
        SqlKind.IN to "in",
    )

private val GENERIC_CALL_TO_OPSTR =
    mapOf(
        StringOperatorTable.STARTSWITH.name to "startswith",
    )

class IcebergFilterVisitor(private val topNode: IcebergTableScan) : RexVisitor<Filter> {
    // / Convert a RexInputRef in a filter, representing a boolean column
    // / into a Bodo compiler DNF expression [[(colName, "==", true)]]
    override fun visitInputRef(var1: RexInputRef): Filter {
        val colName = topNode.deriveRowType().fieldNames[var1.index]
        return listOf(Triple(colName, "==", Expr.BooleanLiteral(true)))
    }

    // / It's impossible at codegen for a filter to only contain a RexLiteral
    override fun visitLiteral(var1: RexLiteral): Filter {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLiteral")
    }

    // / Convert a Search RexCall inside a RelFilter into a
    // / Bodo compiler DNF expression
    fun visitSearchCall(search: RexCall): Filter {
        // By construction op0 must be the column and op1 the search argument
        val op0 = search.operands[0]
        val op1 = search.operands[1]
        // Add a defensive check against a complex expression for arg0
        val (colOp, sargOp) =
            when {
                op0 is RexInputRef && op1 is RexLiteral -> op0 to op1
                else -> throw IllegalStateException()
            }
        val colName = topNode.deriveRowType().fieldNames[colOp.index]

        val sarg = sargOp.value as Sarg<*>
        val rangeSet = sarg.rangeSet
        val literalList: MutableList<Expr> = ArrayList()
        // Note: Only encountering discrete values is enforced by SearchArgExpandProgram
        for (range in rangeSet.asRanges()) {
            literalList.add(LiteralCodeGen.sargValToPyLiteral((range as Range<*>).lowerEndpoint()))
        }
        return listOf(Triple(colName, "in", Expr.List(literalList)))
    }

    // / Convert a RexCall in a filter, representing most filters
    // / into a Bodo compiler DNF expression
    override fun visitCall(var1: RexCall): Filter {
        return if (var1.kind in BASIC_UNARY_CALL_TO_OPSTR) {
            val colOp = var1.operands[0]
            if (colOp !is RexInputRef) {
                throw IllegalStateException("Impossible for filter on non-input col")
            }

            val colName = topNode.deriveRowType().fieldNames[colOp.index]
            val (opStr, value) = BASIC_UNARY_CALL_TO_OPSTR[var1.kind]!!
            listOf(Triple(colName, opStr, value))
        } else if (var1.kind in BASIC_BIN_CALL_TO_OPSTR ||
            (
                (var1.kind == SqlKind.OTHER_FUNCTION || var1.kind == SqlKind.OTHER) &&
                    var1.operator.name in GENERIC_CALL_TO_OPSTR
            )
        ) {
            val op0 = var1.operands[0]
            val op1 = var1.operands[1]
            val (colOp, constOp) =
                when {
                    op0 is RexInputRef && op1 is RexLiteral -> (op0 to op1)
                    op1 is RexInputRef && op0 is RexLiteral -> (op1 to op0)
                    else -> throw IllegalStateException("Codegen with comparison between 2 columns is not supported")
                }

            val colName = topNode.deriveRowType().fieldNames[colOp.index]
            val constExpr = LiteralCodeGen.generateLiteralCode(constOp, null)
            val opStr =
                if (var1.kind in BASIC_BIN_CALL_TO_OPSTR) {
                    BASIC_BIN_CALL_TO_OPSTR[var1.kind]!!
                } else {
                    GENERIC_CALL_TO_OPSTR[var1.operator.name]!!
                }
            listOf(Triple(colName, opStr, constExpr))
        } else if (var1.kind == SqlKind.AND) {
            var1.operands.map { op -> op.accept(this) }.flatten()
        } else if (var1.kind == SqlKind.SEARCH) {
            visitSearchCall(var1)
        } else {
            TODO("Codegen for Call Kind ${var1.kind.name} Implemented Yet")
        }
    }

    override fun visitLocalRef(var1: RexLocalRef): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLocalRef")
    }

    override fun visitOver(var1: RexOver): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLocalRef")
    }

    override fun visitCorrelVariable(var1: RexCorrelVariable): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexCorrelVariable")
    }

    override fun visitDynamicParam(var1: RexDynamicParam): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexDynamicParam")
    }

    override fun visitRangeRef(var1: RexRangeRef): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexRangeRef")
    }

    override fun visitFieldAccess(var1: RexFieldAccess): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexFieldAccess")
    }

    override fun visitSubQuery(var1: RexSubQuery): Filter? {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexSubQuery")
    }

    override fun visitTableInputRef(p0: RexTableInputRef): Filter {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexTableInputRef")
    }

    override fun visitPatternFieldRef(p0: RexPatternFieldRef?): Filter {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexPatternFieldRef")
    }
}
