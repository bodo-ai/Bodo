package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
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
import org.apache.calcite.rex.RexLambda
import org.apache.calcite.rex.RexLambdaRef
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

private val BASIC_UNARY_CALL_TO_OPSTR =
    mapOf(
        SqlKind.IS_NULL to ("IS_NULL" to null),
        SqlKind.IS_NOT_NULL to ("IS_NOT_NULL" to null),
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
        SqlKind.IN to "IN",
    )

private val GENERIC_CALL_TO_OPSTR =
    mapOf(
        StringOperatorTable.STARTSWITH.name to "STARTS_WITH",
    )

// Operators that require a rewrite in code generation.
private val REWRITE_UNARY_KINDS = setOf(SqlKind.IS_NOT_TRUE, SqlKind.IS_NOT_FALSE)
private val REWRITE_BINOP_KINDS = setOf(SqlKind.IS_DISTINCT_FROM, SqlKind.IS_NOT_DISTINCT_FROM)

class IcebergFilterVisitor(private val topNode: IcebergTableScan, private val ctx: BodoPhysicalRel.BuildContext) : RexVisitor<Expr> {
    private val scalarTranslator = ctx.scalarRexTranslator()

    // Static Class that can be accessed outside this class
    companion object {
        // Construct a bodo.ir.filter.Ref for column accesses
        fun ref(colName: String): Expr = Expr.Call("bif.make_ref", Expr.StringLiteral(colName))

        // Construct a bodo.ir.filter.Scalar for wrapping scalar values
        fun scalar(value: Expr): Expr = Expr.Call("bif.make_scalar", value)

        fun op(
            op: String,
            args: List<Expr>,
        ) = Expr.Call("bif.make_op", listOf(Expr.StringLiteral(op)) + args)

        // Construct a bodo.ir.filter.Op for compute or comparison operations
        // inside of filters
        fun op(
            op: String,
            vararg args: Expr,
        ) = op(op, args.toList())

        // Default filter value in special cases like
        // - When there are no filters
        // - With scalar conditions that are always true (TODO)
        fun default() = op("ALWAYS_TRUE")
    }

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
     * Generate the expr for any Unary expression that needs to be rewritten
     * into multiple expressions.
     * For example IS_NOT_TRUE(A) is equivalent to (A == False OR A IS NULL).
     * The exact generated code depends on details like nullability.
     * @param var1 The RexCall that requires multiple expressions
     * to write.
     * @return The rewritten expression.
     */
    private fun visitUnaryRewriteCall(var1: RexCall): Expr {
        val colOp =
            when (val op0 = var1.operands[0]) {
                is RexInputRef -> op0
                else -> throw IllegalStateException("Codegen with comparison between 2 columns is not supported")
            }
        val colName = topNode.deriveRowType().fieldNames[colOp.index]
        val columnRef = ref(colName)
        // We can omit certain checks we know an input is not nullable,
        // which is common for scalars.
        val colIsNullable = colOp.type.isNullable
        // Return the opposite scalar so we can use equality.
        val scalar =
            if (var1.kind == SqlKind.IS_NOT_TRUE) {
                Expr.BooleanLiteral(false)
            } else {
                Expr.BooleanLiteral(true)
            }
        // This is rewritten as (A == Scalar) OR A IS NULL
        val equalsCall = op("==", columnRef, scalar)
        return if (colIsNullable) {
            val nullable = op("IS_NULL", columnRef)
            op("OR", equalsCall, nullable)
        } else {
            equalsCall
        }
    }

    /**
     * Generate the expr for any Binops that needs to be rewritten
     * into multiple expressions.
     * For example IS_NOT_DISTINCT_FROM(A, B) is equivalent to
     * (A == B OR (A IS NULL AND B IS NULL)).
     * The exact generated code depends on details like nullability
     * @param var1 The RexCall that requires multiple expressions
     * to write.
     * @return The rewritten expression.
     */
    private fun visitBinopRewriteCall(var1: RexCall): Expr {
        val op0 = var1.operands[0]
        val op1 = var1.operands[1]
        val (colOp, constOp) =
            when {
                op0 is RexInputRef && op1 is RexLiteral -> (op0 to op1)
                op1 is RexInputRef && op0 is RexLiteral -> (op1 to op0)
                else -> throw IllegalStateException("Codegen with comparison between 2 columns is not supported")
            }
        val colName = topNode.deriveRowType().fieldNames[colOp.index]
        val columnRef = ref(colName)
        val constVar = createScalarAssignment(constOp)
        val scalarRef = scalar(constVar)
        // We can omit certain checks we know an input is not nullable,
        // which is common for scalars.
        val colIsNullable = colOp.type.isNullable
        val scalarIsNullable = constOp.type.isNullable
        return if (var1.kind == SqlKind.IS_DISTINCT_FROM) {
            // Equivalent to A != B AND (A IS NOT NULL OR B IS NOT NULL)
            val notEqualOp = op("!=", columnRef, scalarRef)
            if (colIsNullable && scalarIsNullable) {
                op("AND", notEqualOp, op("OR", op("IS_NOT_NULL", columnRef), op("IS_NOT_NULL", scalarRef)))
            } else if (colIsNullable) {
                op("AND", notEqualOp, op("IS_NOT_NULL", columnRef))
            } else if (scalarIsNullable) {
                op("AND", notEqualOp, op("IS_NOT_NULL", scalarRef))
            } else {
                notEqualOp
            }
        } else {
            // IS NOT DISTINCT FROM is equivalent to A == B OR (A IS NULL AND B IS NULL)
            val equalOp = op("==", columnRef, scalarRef)
            if (colIsNullable && scalarIsNullable) {
                op("OR", equalOp, op("AND", op("IS_NULL", columnRef), op("IS_NULL", scalarRef)))
            } else {
                // If either is non-nullable they must be equal.
                equalOp
            }
        }
    }

    /**
     * Convert a RexInputRef in a filter, representing a boolean column
     * into a Bodo compiler filter expression Op("==", ref(colName), scalar(True)).
     * This assumes the reference to a column is a boolean column.
     * @param var1 The RexInputRef to convert.
     * @return The filter expression representing the RexInputRef.
     */
    override fun visitInputRef(var1: RexInputRef): Expr {
        val colName = topNode.deriveRowType().fieldNames[var1.index]
        return op("==", ref(colName), scalar(Expr.BooleanLiteral(true)))
    }

    /**
     * Convert a Search RexCall in a filter, representing a search
     * into a Bodo compiler filter expression [[(colName, "in", scalarVar)]].
     * @param search The RexCall to convert.
     * @return The filter expression representing the RexCall.
     */
    private fun visitSearchCall(search: RexCall): Expr {
        // By construction op0 must be the column and op1 the search argument
        val colOp = search.operands[0] as RexInputRef
        // Note: This must be a scalar.
        val sargOp = search.operands[1]
        val colName = topNode.deriveRowType().fieldNames[colOp.index]
        val scalarVar = createScalarAssignment(sargOp)
        return op("IN", ref(colName), scalar(scalarVar))
    }

    /**
     * Convert a RexCall in a filter, representing most filters
     * into a Bodo compiler filter expression.
     * @param var1 The RexCall to convert.
     * @return The filter expression representing the RexCall.
     */
    override fun visitCall(var1: RexCall): Expr {
        // Complex Operators with Special Handling
        return if (var1.kind == SqlKind.SEARCH) {
            visitSearchCall(var1)
        } else if (var1.kind in REWRITE_UNARY_KINDS) {
            visitUnaryRewriteCall(var1)
        } else if (var1.kind in REWRITE_BINOP_KINDS) {
            visitBinopRewriteCall(var1)
        } else if (var1.kind in BASIC_UNARY_CALL_TO_OPSTR) {
            // Basic Operators with Normal Handling
            val colOp = var1.operands[0]
            if (colOp !is RexInputRef) {
                throw IllegalStateException("Impossible for filter on non-input col")
            }

            val colName = topNode.deriveRowType().fieldNames[colOp.index]
            val (opStr, value) = BASIC_UNARY_CALL_TO_OPSTR[var1.kind]!!

            if (value == null) {
                op(opStr, ref(colName))
            } else {
                op(opStr, ref(colName), scalar(value))
            }
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
            op(opStr, ref(colName), scalar(scalarVar))

            // Logical Operators
        } else if (var1.kind == SqlKind.AND) {
            op("AND", var1.operands.map { op -> op.accept(this) })
        } else if (var1.kind == SqlKind.OR) {
            op("OR", var1.operands.map { op -> op.accept(this) })
        } else if (var1.kind == SqlKind.NOT) {
            op("NOT", var1.operands[0].accept(this))
        } else if (var1.kind == SqlKind.SEARCH) {
            visitSearchCall(var1)
        } else {
            TODO("Codegen for Call Kind ${var1.kind.name} Implemented Yet")
        }
    }

    /**
     * Unreachable code because we do not accept filters that only contain a RexLiteral.
     */
    override fun visitLiteral(var1: RexLiteral): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLiteral")
    }

    // ------ It's impossible at codegen for a filter to contain any of the following ------
    override fun visitLocalRef(var1: RexLocalRef): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLocalRef")
    }

    override fun visitOver(var1: RexOver): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexLocalRef")
    }

    override fun visitCorrelVariable(var1: RexCorrelVariable): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexCorrelVariable")
    }

    override fun visitDynamicParam(var1: RexDynamicParam): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexDynamicParam")
    }

    override fun visitRangeRef(var1: RexRangeRef): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexRangeRef")
    }

    override fun visitFieldAccess(var1: RexFieldAccess): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexFieldAccess")
    }

    override fun visitSubQuery(var1: RexSubQuery): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexSubQuery")
    }

    override fun visitTableInputRef(p0: RexTableInputRef): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexTableInputRef")
    }

    override fun visitPatternFieldRef(p0: RexPatternFieldRef): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a RexPatternFieldRef")
    }

    override fun visitLambda(var1: RexLambda): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a Lambda")
    }

    override fun visitLambdaRef(var1: RexLambdaRef): Expr {
        throw NotImplementedError("IcebergFilterVisitor in Codegen should not see a LambdaRef")
    }
}
