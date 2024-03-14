package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Variable
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexCorrelVariable
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexFieldAccess
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexPatternFieldRef
import org.apache.calcite.rex.RexRangeRef
import org.apache.calcite.rex.RexSubQuery
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.rex.RexVisitor
import org.apache.calcite.sql.SqlKind

typealias Filter = List<Triple<String, String, Expr>>

private val BASIC_UNARY_CALL_TO_OPSTR =
    mapOf(
        SqlKind.IS_NULL to ("is" to Expr.StringLiteral("NULL")),
        SqlKind.IS_NOT_NULL to ("is not" to Expr.StringLiteral("NULL")),
        SqlKind.IS_TRUE to ("==" to Expr.BooleanLiteral(true)),
        SqlKind.IS_FALSE to ("==" to Expr.BooleanLiteral(false)),
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

class IcebergFilterVisitor(private val topNode: IcebergTableScan, private val ctx: PandasRel.BuildContext) : RexVisitor<Filter> {
    private val scalarTranslator = ctx.scalarRexTranslator()

    /**
     * Helper function to convert a RexNode scalar, generate the
     * expression, create an assignment which is added to the code
     * generation module and return the variable representing the
     * scalar.
     */
    private fun createScalarAssignment(scalar: RexNode): Variable {
        // Generate a new code generation frame because we are inside streaming
        ctx.builder().startCodegenFrame()
        val scalarExpr = scalar.accept(scalarTranslator)
        val scalarVar = ctx.builder().symbolTable.genGenericTempVar()
        val assignment = Op.Assign(scalarVar, scalarExpr)
        ctx.builder().add(assignment)
        // Pop the frame and add to initialization
        val frame = ctx.builder().endFrame()
        val frameOp = Op.InsertFrame(frame)
        if (ctx.builder().isStreamingFrame()) {
            ctx.builder().getCurrentStreamingPipeline().addInitialization(frameOp)
        } else {
            ctx.builder().add(frameOp)
        }
        return scalarVar
    }

    /**
     * Convert a RexInputRef in a filter, representing a boolean column
     * into a Bodo compiler DNF expression [[(colName, "==", true)]].
     * This assumes the reference to a column is a boolean column.
     * @param var1 The RexInputRef to convert.
     * @return The DNF expression representing the RexInputRef.
     */
    override fun visitInputRef(var1: RexInputRef): Filter {
        val colName = topNode.deriveRowType().fieldNames[var1.index]
        return listOf(Triple(colName, "==", Expr.BooleanLiteral(true)))
    }

    /**
     * Unreachable code because we do not accept filters that only contain a RexLiteral.
     */
    override fun visitLiteral(var1: RexLiteral): Filter {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLiteral")
    }

    /**
     * Convert a Search RexCall in a filter, representing a search
     * into a Bodo compiler DNF expression [[(colName, "in", scalarVar)]].
     * @param search The RexCall to convert.
     * @return The DNF expression representing the RexCall.
     */
    fun visitSearchCall(search: RexCall): Filter {
        // By construction op0 must be the column and op1 the search argument
        val colOp = search.operands[0] as RexInputRef
        // Note: This must be a scalar.
        val op1 = search.operands[1]
        val colName = topNode.deriveRowType().fieldNames[colOp.index]
        val scalarVar = createScalarAssignment(op1)
        return listOf(Triple(colName, "in", scalarVar))
    }

    /**
     * Convert a RexCall in a filter, representing most filters
     * into a Bodo compiler DNF expression.
     * @param var1 The RexCall to convert.
     * @return The DNF expression representing the RexCall.
     */
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
            val (colOp, scalarOp) =
                when {
                    op0 is RexInputRef && IsScalar.isScalar(op1) -> (op0 to op1)
                    op1 is RexInputRef && IsScalar.isScalar(op0) -> (op1 to op0)
                    else -> throw IllegalStateException("Codegen with comparison between 2 columns is not supported")
                }

            val colName = topNode.deriveRowType().fieldNames[colOp.index]
            val scalarVar = createScalarAssignment(scalarOp)
            val opStr =
                if (var1.kind in BASIC_BIN_CALL_TO_OPSTR) {
                    BASIC_BIN_CALL_TO_OPSTR[var1.kind]!!
                } else {
                    GENERIC_CALL_TO_OPSTR[var1.operator.name]!!
                }
            listOf(Triple(colName, opStr, scalarVar))
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
