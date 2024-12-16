package com.bodosql.calcite.application.logicalRules

import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCallBinding
import org.apache.calcite.rex.RexFieldCollation
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexWindowBound
import org.apache.calcite.rex.RexWindowBounds
import org.apache.calcite.sql.SqlAggFunction
import org.apache.calcite.sql.SqlWindow
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import java.math.BigDecimal

/**
 * Extracts any OVER calls from the join condition.
 *
 * We cannot handle an OVER call inside a join, so we need to extract
 * it from the condition into its own column and then remove the column after
 * the join has been performed.
 */
abstract class AbstractWindowDecomposeRule protected constructor(
    config: Config,
) : RelRule<AbstractWindowDecomposeRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall) {
        val project: Project = call.rel(0)
        val input: RelNode = project.input
        val decomposer = WindowDecomposer(input.rowType.fieldCount, call.builder().rexBuilder)
        val rewrittenNodes = project.projects.map { it.accept(decomposer) }

        if (decomposer.doneRewrite) {
            val builder = call.builder()
            builder.push(project.input)

            // If the decomposer created new child nodes, place those into a new child
            // projection before the projection with the decomposed window functions.
            if (decomposer.newNodes.isNotEmpty()) {
                val newInputs: MutableList<RexNode> = mutableListOf()
                input.rowType.fieldList.forEachIndexed { idx, field ->
                    newInputs.add(RexInputRef(idx, field.type))
                }
                newInputs.addAll(decomposer.newNodes)
                builder.project(newInputs)
            }

            builder.project(rewrittenNodes)
            call.transformTo(builder.build())
        }
    }

    interface Config : RelRule.Config

    companion object {
        /**
         * A predicate rule indicating whether a RexOver is allowed to be decomposed. This rule should
         * be invoked on function types with the policy that we always rewrite unless the window function
         * is in the form `FUNC(...) OVER ()` (no partition columns, no order columns).
         */
        fun allowNonBlankRewriteRule(over: RexOver): Boolean {
            val window = over.window
            return window.partitionKeys.size > 0 ||
                window.orderKeys.size > 0 &&
                window.partitionKeys.all {
                    it !is RexLiteral
                } &&
                window.orderKeys.all { it.left !is RexLiteral }
        }

        /**
         * A predicate rule indicating whether a RexOver is allowed to be decomposed. This rule should
         * be invoked on function types with the policy that we always rewrite unless the window function
         * is in the form FUNC(DISTINCT X) OVER (...)
         */
        fun allowNonDistinctRewriteRule(over: RexOver): Boolean = !over.isDistinct

        /**
         * Copies a RexOver node but with a different operator and argument nodes.
         *
         * @param builder The rex builder used to create new rex nodes.
         * @param over The original RexOver call being duplicated.
         * @param typ The new output data type.
         * @param operator The new operator to use.
         * @param args The new operands to use.
         */
        fun duplicateOverWithNewOperator(
            builder: RexBuilder,
            over: RexOver,
            op: SqlAggFunction,
            args: List<RexNode>,
        ): RexNode {
            // Infer the output type of the new window function
            val binding: RexCallBinding =
                object : RexCallBinding(
                    builder.typeFactory,
                    op,
                    args,
                    ImmutableList.of(),
                ) {
                    override fun getGroupCount(): Int =
                        if (SqlWindow.isAlwaysNonEmpty(over.window.lowerBound, over.window.upperBound)) 1 else 0
                }
            val outTyp = op.returnTypeInference!!.inferReturnType(binding)
            // Create the new call
            return builder.makeOver(
                outTyp,
                op,
                args,
                over.window.partitionKeys,
                over.window.orderKeys,
                over.window.lowerBound,
                over.window.upperBound,
                over.window.isRows,
                true,
                false,
                over.isDistinct,
                over.ignoreNulls(),
            )
        }

        /**
         * Copies a RexOver node but with a different operator and argument nodes, and with
         * new window bounds.
         *
         * @param builder The rex builder used to create new rex nodes.
         * @param over The original RexOver call being duplicated.
         * @param typ The new output data type.
         * @param operator The new operator to use.
         * @param args The new operands to use.
         * @param lower: The new lower bound to use.
         * @param upper: The new upper bound to use.
         */
        fun duplicateOverWithNewOperatorAndBounds(
            builder: RexBuilder,
            over: RexOver,
            op: SqlAggFunction,
            args: List<RexNode>,
            lower: RexWindowBound,
            upper: RexWindowBound,
        ): RexNode {
            // Infer the output type of the new window function
            val binding: RexCallBinding =
                object : RexCallBinding(
                    builder.typeFactory,
                    op,
                    args,
                    ImmutableList.of(),
                ) {
                    override fun getGroupCount(): Int = if (SqlWindow.isAlwaysNonEmpty(lower, upper)) 1 else 0
                }
            val outTyp = op.returnTypeInference!!.inferReturnType(binding)
            // Create the new call
            return builder.makeOver(
                outTyp,
                op,
                args,
                over.window.partitionKeys,
                over.window.orderKeys,
                lower,
                upper,
                over.window.isRows,
                true,
                false,
                over.isDistinct,
                over.ignoreNulls(),
            )
        }

        /**
         * Copies a RexOver node but with a different operator and argument nodes, and with
         * new window partition/order terms.
         *
         * @param builder The rex builder used to create new rex nodes.
         * @param over The original RexOver call being duplicated.
         * @param typ The new output data type.
         * @param operator The new operator to use.
         * @param args The new operands to use.
         * @param partition: The new partition keys to use.
         * @param order: The new order keys to use.
         */
        fun duplicateOverWithNewOperatorAndWindow(
            builder: RexBuilder,
            over: RexOver,
            op: SqlAggFunction,
            args: List<RexNode>,
            partition: List<RexNode>,
            order: ImmutableList<RexFieldCollation>,
        ): RexNode {
            // Infer the output type of the new window function
            val binding: RexCallBinding =
                object : RexCallBinding(
                    builder.typeFactory,
                    op,
                    args,
                    ImmutableList.of(),
                ) {
                    override fun getGroupCount(): Int =
                        if (SqlWindow.isAlwaysNonEmpty(over.window.lowerBound, over.window.upperBound)) 1 else 0
                }
            val outTyp = op.returnTypeInference!!.inferReturnType(binding)
            // Create the new call
            return builder.makeOver(
                outTyp,
                op,
                args,
                partition,
                order,
                over.window.lowerBound,
                over.window.upperBound,
                over.window.isRows,
                true,
                false,
                over.isDistinct,
                over.ignoreNulls(),
            )
        }

        /**
         * Rewrites an AVG(X) OVER (...) to decompose into a call to SUM and COUNT as window functions.
         */
        fun rewriteAvg(
            builder: RexBuilder,
            over: RexOver,
        ): RexNode {
            val sum = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, over.operands)
            val count = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, over.operands)
            return builder.makeCall(SqlStdOperatorTable.DIVIDE, sum, count)
        }

        /**
         * Rewrites the functions STDDEV_SAMP, STDDEV_POP, VAR_SAMP, and VAR_POP as window functions
         * into combinations of SUM(X), SUM(X*X), and COUNT(X) as window functions.
         *
         * @param isSample: true if doing STDDEV_SAMP or VAR_SAMP, in which case we subtract 1 from the
         * denominator of COUNT(X).
         * @param root: true if doing STDDEV_SAMP or STDDEV_POP, in which case we take the square root
         * of the final answer.
         */
        fun rewriteStd(
            isSample: Boolean,
            root: Boolean,
        ): (RexBuilder, RexOver) -> RexNode =
            { builder: RexBuilder, over: RexOver ->
                // Calculate SUM(X), SUM(X*X), and COUNT(X)
                val x = over.operands[0]
                val xSquared = builder.makeCall(SqlStdOperatorTable.MULTIPLY, x, x)
                val sumX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(x))
                val sumSquareX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(xSquared))
                val countX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, listOf(x))
                val sumXSquared = builder.makeCall(SqlStdOperatorTable.MULTIPLY, sumX, sumX)

                // Formula for variance: (sum(x * x) - sum(x) * sum(x) / count(x)) / count(x)
                val numerator =
                    builder.makeCall(
                        SqlStdOperatorTable.MINUS,
                        sumSquareX,
                        builder.makeCall(SqlStdOperatorTable.DIVIDE, sumXSquared, countX),
                    )
                val denominator =
                    if (isSample) {
                        // If doing sample variance, subtract 1 from the count in the denominator (and make sure
                        // it is null if the result is not > 0).
                        val oneArg = builder.makeExactLiteral(BigDecimal.ONE)
                        val nullArg = builder.makeNullLiteral(countX.type)
                        val cmpArg = builder.makeCall(SqlStdOperatorTable.LESS_THAN_OR_EQUAL, countX, oneArg)
                        val newDenomArg = builder.makeCall(SqlStdOperatorTable.MINUS, countX, oneArg)
                        builder.makeCall(CondOperatorTable.IFF_FUNC, cmpArg, nullArg, newDenomArg)
                    } else {
                        countX
                    }
                val quotient = builder.makeCall(SqlStdOperatorTable.DIVIDE, numerator, denominator)
                val castedQuotient = builder.makeCast(over.type, quotient)

                // If doing standard deviation, take the square root of the result
                if (root) {
                    val half = builder.makeLiteral(0.5, over.type)
                    builder.makeCall(SqlStdOperatorTable.POWER, castedQuotient, half)
                } else {
                    castedQuotient
                }
            }

        /**
         * Rewrites the functions COVAR_SAMP and COVAR_POP as window functions into combinations of SUM(X),
         * SUM(Y), SUM(X*Y), and COUNT(X*Y) as window functions.
         *
         * @param isSample: true if doing COVAR_SAMP or VAR_SAMP, in which case we subtract 1 from the
         * denominator of COUNT(X*Y).
         */
        fun rewriteCovar(isSample: Boolean): (RexBuilder, RexOver) -> RexNode =
            { builder: RexBuilder, over: RexOver ->
                // Pre-process X and Y to null-out any rows where X or Y is null.
                val rawX = over.operands[0]
                val rawY = over.operands[1]
                val nullX = builder.makeNullLiteral(rawX.type)
                val nullY = builder.makeNullLiteral(rawY.type)
                val x = builder.makeCall(CondOperatorTable.IFF_FUNC, builder.makeCall(SqlStdOperatorTable.IS_NULL, rawY), nullX, rawX)
                val y = builder.makeCall(CondOperatorTable.IFF_FUNC, builder.makeCall(SqlStdOperatorTable.IS_NULL, rawX), nullY, rawY)

                // Calculate SUM(X), SUM(Y), SUM(X*Y) and COUNT(X*Y)
                val xy = builder.makeCall(SqlStdOperatorTable.MULTIPLY, x, y)
                val sumX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(x))
                val sumY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(y))
                val sumXY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(xy))
                val countXY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, listOf(xy))

                // Formula for covariance: (sum(x * y) - sum(x) * sum(y) / count(x * y)) / count(x * y)
                val sumXsumY = builder.makeCall(SqlStdOperatorTable.MULTIPLY, sumX, sumY)
                val numerator =
                    builder.makeCall(
                        SqlStdOperatorTable.MINUS,
                        sumXY,
                        builder.makeCall(SqlStdOperatorTable.DIVIDE, sumXsumY, countXY),
                    )
                val denominator =
                    if (isSample) {
                        // If doing sample covariance, subtract 1 from the count in the denominator (and make sure
                        // it is null if the result is not > 0).
                        val oneArg = builder.makeExactLiteral(BigDecimal.ONE)
                        val nullArg = builder.makeNullLiteral(countXY.type)
                        val cmpArg = builder.makeCall(SqlStdOperatorTable.LESS_THAN_OR_EQUAL, countXY, oneArg)
                        val newDenomArg = builder.makeCall(SqlStdOperatorTable.MINUS, countXY, oneArg)
                        builder.makeCall(CondOperatorTable.IFF_FUNC, cmpArg, nullArg, newDenomArg)
                    } else {
                        countXY
                    }
                val quotient = builder.makeCall(SqlStdOperatorTable.DIVIDE, numerator, denominator)
                builder.makeCast(over.type, quotient)
            }

        /**
         * Rewrites the function CORR as window functions into a combination of SUM(X), SUM(Y), SUM(X*Y), and
         * COUNT(X*Y) as window functions.
         */
        fun rewriteCorr(
            builder: RexBuilder,
            over: RexOver,
        ): RexNode {
            // Pre-process X and Y to null-out any rows where X or Y is null.
            val rawX = over.operands[0]
            val rawY = over.operands[1]
            val nullX = builder.makeNullLiteral(rawX.type)
            val nullY = builder.makeNullLiteral(rawY.type)
            val x = builder.makeCall(CondOperatorTable.IFF_FUNC, builder.makeCall(SqlStdOperatorTable.IS_NULL, rawY), nullX, rawX)
            val y = builder.makeCall(CondOperatorTable.IFF_FUNC, builder.makeCall(SqlStdOperatorTable.IS_NULL, rawX), nullY, rawY)

            // Calculate SUM(X), SUM(Y), SUM(X*X), SUM(Y*Y), SUM(X*Y), COUNT(X*Y),
            // SUM(X)*SUM(X), SUM(Y)*SUM(Y), and SUM(X)*SUM(Y)
            val xx = builder.makeCall(SqlStdOperatorTable.MULTIPLY, x, x)
            val yy = builder.makeCall(SqlStdOperatorTable.MULTIPLY, y, y)
            val xy = builder.makeCall(SqlStdOperatorTable.MULTIPLY, x, y)
            val sumX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(x))
            val sumY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(y))
            val sumXX = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(xx))
            val sumYY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(yy))
            val sumXY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, listOf(xy))
            val countXY = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, listOf(xy))
            val sumXsumX = builder.makeCall(SqlStdOperatorTable.MULTIPLY, sumX, sumX)
            val sumYsumY = builder.makeCall(SqlStdOperatorTable.MULTIPLY, sumY, sumY)
            val sumXsumY = builder.makeCall(SqlStdOperatorTable.MULTIPLY, sumX, sumY)

            val zeroArg = builder.makeExactLiteral(BigDecimal.ZERO)
            val halfArg = builder.makeLiteral(0.5, over.type)

            // Calculate COVAR_POP(X, Y)
            val covarNumerator =
                builder.makeCall(
                    SqlStdOperatorTable.MINUS,
                    sumXY,
                    builder.makeCall(SqlStdOperatorTable.DIVIDE, sumXsumY, countXY),
                )
            val covarPopXY = builder.makeCall(SqlStdOperatorTable.DIVIDE, covarNumerator, countXY)

            // Calculate STDDEV_POP(X), ignoring rows where Y is null.
            val stdXNumerator =
                builder.makeCall(
                    SqlStdOperatorTable.MINUS,
                    sumXX,
                    builder.makeCall(SqlStdOperatorTable.DIVIDE, sumXsumX, countXY),
                )
            val stdXQuotient = builder.makeCast(over.type, builder.makeCall(SqlStdOperatorTable.DIVIDE, stdXNumerator, countXY))
            val stdX = builder.makeCall(SqlStdOperatorTable.POWER, stdXQuotient, halfArg)

            // Calculate STDDEV_POP(Y), ignoring rows where X is null.
            val stdYNumerator =
                builder.makeCall(
                    SqlStdOperatorTable.MINUS,
                    sumYY,
                    builder.makeCall(SqlStdOperatorTable.DIVIDE, sumYsumY, countXY),
                )
            val stdYQuotient = builder.makeCast(over.type, builder.makeCall(SqlStdOperatorTable.DIVIDE, stdYNumerator, countXY))
            val stdY = builder.makeCall(SqlStdOperatorTable.POWER, stdYQuotient, halfArg)

            // Formula for correlation: COVAR_POP(X, Y) / NULLIF(STDDEV_POP(X) * STDDEV_POP(Y), 0)
            val denominator =
                builder.makeCall(
                    CondOperatorTable.NULLIF,
                    builder.makeCall(SqlStdOperatorTable.MULTIPLY, stdX, stdY),
                    zeroArg,
                )
            val quotient = builder.makeCall(SqlStdOperatorTable.DIVIDE, covarPopXY, denominator)
            return builder.makeCast(over.type, quotient)
        }

        /**
         * Rewrites a `PERCENT_RANK() OVER (...)` call into the form
         * `DIV0(RANK() OVER (...) - 1, COUNT(*) OVER (...) - 1)`
         */
        fun rewritePercentRank(
            builder: RexBuilder,
            over: RexOver,
        ): RexNode {
            val rank =
                duplicateOverWithNewOperatorAndBounds(
                    builder,
                    over,
                    SqlStdOperatorTable.RANK,
                    listOf(),
                    RexWindowBounds.UNBOUNDED_PRECEDING,
                    RexWindowBounds.UNBOUNDED_FOLLOWING,
                )
            val rankMinus = builder.makeCall(SqlStdOperatorTable.MINUS, rank, builder.makeExactLiteral(BigDecimal.ONE))
            val count =
                duplicateOverWithNewOperatorAndBounds(
                    builder,
                    over,
                    SqlStdOperatorTable.COUNT,
                    listOf(),
                    RexWindowBounds.UNBOUNDED_PRECEDING,
                    RexWindowBounds.UNBOUNDED_FOLLOWING,
                )
            val countMinus = builder.makeCall(SqlStdOperatorTable.MINUS, count, builder.makeExactLiteral(BigDecimal.ONE))
            return builder.makeCall(NumericOperatorTable.DIV0, rankMinus, countMinus)
        }

        /**
         * Rewrites a `RATIO_TO_REPORT(X) OVER (...)` call into the form
         * `DIV0(RANK() OVER (...) - 1, SUM(X) OVER (...) - 1)`
         */
        fun rewriteRatioToReport(
            builder: RexBuilder,
            over: RexOver,
        ): RexNode {
            val sum =
                duplicateOverWithNewOperatorAndBounds(
                    builder,
                    over,
                    SqlStdOperatorTable.SUM,
                    over.operands,
                    RexWindowBounds.UNBOUNDED_PRECEDING,
                    RexWindowBounds.UNBOUNDED_FOLLOWING,
                )
            val denominator = builder.makeCall(CondOperatorTable.NULLIF, sum, builder.makeExactLiteral(BigDecimal.ZERO))
            return builder.makeCall(SqlStdOperatorTable.DIVIDE, over.operands[0], denominator)
        }

        val shouldRewriteRules =
            mapOf(
                "AVG" to AbstractWindowDecomposeRule::allowNonBlankRewriteRule,
                "VAR_SAMP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "VAR_POP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "STDDEV_SAMP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "STDDEV_POP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "COVAR_SAMP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "COVAR_POP" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "CORR" to AbstractWindowDecomposeRule::allowNonDistinctRewriteRule,
                "PERCENT_RANK" to { _ -> true },
                "CUME_DIST" to { _ -> true },
                "RATIO_TO_REPORT" to { _ -> true },
            )

        val rewriteFuncs =
            mapOf(
                "AVG" to AbstractWindowDecomposeRule::rewriteAvg,
                "VAR_SAMP" to rewriteStd(isSample = true, root = false),
                "VAR_POP" to rewriteStd(isSample = false, root = false),
                "STDDEV_SAMP" to rewriteStd(isSample = true, root = true),
                "STDDEV_POP" to rewriteStd(isSample = false, root = true),
                "COVAR_SAMP" to rewriteCovar(isSample = true),
                "COVAR_POP" to rewriteCovar(isSample = false),
                "CORR" to AbstractWindowDecomposeRule::rewriteCorr,
                "PERCENT_RANK" to AbstractWindowDecomposeRule::rewritePercentRank,
                "RATIO_TO_REPORT" to AbstractWindowDecomposeRule::rewriteRatioToReport,
            )

        val rewriteFuncsWithNewNodes =
            mapOf(
                "CUME_DIST" to WindowDecomposer::rewriteCumeDist,
            )

        class WindowDecomposer(
            val nInputs: Int,
            val builder: RexBuilder,
        ) : RexShuttle() {
            public var doneRewrite = false
            public val newNodes: MutableList<RexNode> = mutableListOf()

            override fun visitOver(over: RexOver): RexNode {
                val name = over.op.name
                // Verify that the window function call should be rewritten according to the predicates in
                // shouldRewriteRules.
                if (!shouldRewriteRules.containsKey(name) || !shouldRewriteRules[name]!!(over)) {
                    return super.visitOver(over)
                }
                // If we get this far, we can guarantee at least 1 rewrite has happened.
                doneRewrite = true

                // Invoke the appropriate rewrite function.
                return if (rewriteFuncsWithNewNodes.containsKey(name)) {
                    rewriteFuncsWithNewNodes[name]!!(this, over)
                } else {
                    rewriteFuncs[name]!!(builder, over)
                }
            }

            /**
             * Rewrites a `CUME_DIST() OVER (PARTITION BY A ORDER BY B)` call into the form
             * `MAX(ROW_NUMBER() OVER (PARTITION BY A ORDER BY B) OVER (PARTITION BY A, B)) / COUNT(*) OVER (PARTITION BY A)`.
             * Ensures that the row_number and count calls are added to a child projection.
             */
            fun rewriteCumeDist(over: RexOver): RexNode {
                val rowNumber = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.ROW_NUMBER, listOf())
                val rowNumberRef = RexInputRef(nInputs + newNodes.size, rowNumber.type)
                newNodes.add(rowNumber)
                val count = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, listOf())
                val countRef = RexInputRef(nInputs + newNodes.size, count.type)
                newNodes.add(count)
                val newPartition = over.window.partitionKeys + over.window.orderKeys.map { it.key }
                val max =
                    duplicateOverWithNewOperatorAndWindow(
                        builder,
                        over,
                        SqlStdOperatorTable.MAX,
                        listOf(rowNumberRef),
                        newPartition,
                        ImmutableList.of(),
                    )
                return builder.makeCall(SqlStdOperatorTable.DIVIDE, max, countRef)
            }
        }
    }
}
