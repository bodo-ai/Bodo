package com.bodosql.calcite.application.utils

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexSimplify
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlCastFunction

/**
 * Force each call to be normalized in a RexNode.
 */
class RexNormalizer private constructor(
    private val rexBuilder: RexBuilder,
) : RexShuttle() {
    override fun visitOver(over: RexOver): RexNode = over

    override fun visitCall(call: RexCall): RexNode {
        // Use an array to pass a boolean by reference and update
        // inside visitList.
        val update = booleanArrayOf(false)
        var newOperands = visitList(call.operands, update)
        // The order of AND/OR can create issues for deterministic planning.
        if (call.kind == SqlKind.AND || call.kind == SqlKind.OR) {
            val sortedOperands = newOperands.sortedBy { it.toString() }
            if (sortedOperands != newOperands) {
                newOperands = sortedOperands
                update[0] = true
            }
        }
        return if (update[0]) {
            // Cast functions have a separate API and cannot use the generic makeCall.
            if (call.op is SqlCastFunction) {
                rexBuilder.makeCast(call.getType(), newOperands[0])
            } else {
                rexBuilder.makeCall(call.op, newOperands)
            }
        } else {
            call
        }
    }

    companion object {
        @JvmStatic
        fun normalize(
            rexBuilder: RexBuilder,
            node: RexNode,
        ): RexNode = normalize(rexBuilder, node, false)

        @JvmStatic
        fun normalize(
            rexBuilder: RexBuilder,
            node: RexNode,
            matchType: Boolean,
        ): RexNode = normalize(rexBuilder, node, matchType, false)

        /**
         * Normalizes a filter or join condition and folds comparisons against a NULL literal
         * to FALSE (following [CALCITE-7070]).
         *
         * Filter and join conditions are collected into
         * [org.apache.calcite.plan.RelOptPredicateList.pulledUpPredicates] as conjunctions
         * without any further simplification, and Calcite rejects comparisons against NULL
         * there because they make the derived constant map inconsistent. A comparison against
         * NULL can appear after conversion when a rule inlines a constant projection into a
         * predicate, for example pushing a filter past a project that computes
         * `null::VARCHAR`. Conjuncts are folded individually because UNKNOWN is only
         * equivalent to FALSE for the top-level terms of a condition.
         */
        @JvmStatic
        fun normalizeCondition(
            rexBuilder: RexBuilder,
            node: RexNode,
        ): RexNode = normalize(rexBuilder, node, false, true)

        @JvmStatic
        fun normalize(
            rexBuilder: RexBuilder,
            node: RexNode,
            matchType: Boolean,
            unknownAsFalse: Boolean,
        ): RexNode {
            val normalizer = RexNormalizer(rexBuilder)
            val result = node.accept(normalizer)
            val folded =
                if (unknownAsFalse) {
                    foldNullComparisons(rexBuilder, result)
                } else {
                    result
                }
            return if (matchType && folded.type != node.type) {
                rexBuilder.makeCast(node.type, folded, true, false)
            } else {
                folded
            }
        }

        private fun foldNullComparisons(
            rexBuilder: RexBuilder,
            node: RexNode,
        ): RexNode {
            val conjuncts = RelOptUtil.conjunctions(node)
            var changed = false
            val folded =
                conjuncts.map {
                    val foldedConjunct = RexSimplify.simplifyComparisonWithNull(it, rexBuilder)
                    if (foldedConjunct !== it) {
                        changed = true
                    }
                    foldedConjunct
                }
            return if (changed) RexUtil.composeConjunction(rexBuilder, folded) else node
        }
    }
}
