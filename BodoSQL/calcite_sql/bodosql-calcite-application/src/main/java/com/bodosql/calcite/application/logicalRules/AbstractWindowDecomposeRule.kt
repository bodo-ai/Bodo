package com.bodosql.calcite.application.logicalRules

import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCallBinding
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.sql.SqlAggFunction
import org.apache.calcite.sql.SqlWindow
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable

/**
 * Extracts any OVER calls from the join condition.
 *
 * We cannot handle an OVER call inside a join, so we need to extract
 * it from the condition into its own column and then remove the column after
 * the join has been performed.
 */
abstract class AbstractWindowDecomposeRule protected constructor(config: Config) :
    RelRule<AbstractWindowDecomposeRule.Config>(config) {
        override fun onMatch(call: RelOptRuleCall) {
            val decomposer = WindowDecomposer(call.builder().rexBuilder)

            val project: Project = call.rel(0)
            val rewrittenNodes = project.projects.map { it.accept(decomposer) }

            if (decomposer.doneRewrite) {
                val builder = call.builder()
                builder.push(project.input)
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
                return window.partitionKeys.size > 0 || window.orderKeys.size > 0 &&
                    window.partitionKeys.all {
                        it !is RexLiteral
                    } && window.orderKeys.all { it.left !is RexLiteral }
            }

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
                        override fun getGroupCount(): Int {
                            return if (SqlWindow.isAlwaysNonEmpty(over.window.lowerBound, over.window.upperBound)) 1 else 0
                        }
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

            fun rewriteAvg(
                builder: RexBuilder,
                over: RexOver,
            ): RexNode {
                val sum = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.SUM, over.operands)
                val count = duplicateOverWithNewOperator(builder, over, SqlStdOperatorTable.COUNT, over.operands)
                return builder.makeCall(SqlStdOperatorTable.DIVIDE, sum, count)
            }

            val shouldRewriteRules =
                mapOf(
                    "AVG" to AbstractWindowDecomposeRule::allowNonBlankRewriteRule,
                )

            val rewriteFuncs =
                mapOf(
                    "AVG" to AbstractWindowDecomposeRule::rewriteAvg,
                )

            class WindowDecomposer(val builder: RexBuilder) : RexShuttle() {
                public var doneRewrite = false

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
                    return rewriteFuncs[name]!!(builder, over)
                }
            }
        }
    }
