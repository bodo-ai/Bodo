package com.bodosql.calcite.adapter.common

import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import java.util.function.Predicate

/**
 * Common utilities for converting a Filter from logical to a convention
 * (or across conventions).
 */
class FilterUtils private constructor() {
    companion object {
        /**
         * Extract the nodes for standard Filter conversion rules.
         */
        @JvmStatic
        fun <E> extractFilterNodes(call: RelOptRuleCall): Pair<Filter, E> =
            when (call.rels.size) {
                // Inputs are:
                // Filter ->
                //     XXXToPandasConverter ->
                //         XXXRel
                3 -> Pair(call.rel(0), call.rel(2))
                // Inputs are:
                // Filter ->
                //     XXXRel
                else -> Pair(call.rel(0), call.rel(1))
            }

        /**
         * @param condition The conditions to a filter that are being checked to see if it is a
         * conjunction of conditions that can be partially pushed into a target convention.
         * @param builder The RexBuilder used to generate new conditions.
         * @param pushablePredicate A predicate used to evaluate if a condition can be pushed into
         * the target convention. To support a new convention you provide an implementation that
         * defines the rules for conversion.
         * @param partialFilterDerivationFunction A function that can be used to derive a new filter
         * that is a subset of the original filter and may be possible to push. This requires keeping
         * the original filter as well.
         * @return A pair of conditions representing the subset of the condition that can be pushed into
         * the new dialect, and the subset that can not.
         */
        @JvmStatic
        fun extractPushableConditions(
            condition: RexNode,
            builder: RexBuilder,
            pushablePredicate: Predicate<RexNode>,
            partialFilterDerivationFunction: (RexNode) -> RexNode = { it },
        ): Pair<RexNode?, RexNode?> {
            val pushableConditions: MutableList<RexNode> = ArrayList()
            val nonPushableConditions: MutableList<RexNode> = ArrayList()
            for (node in RelOptUtil.conjunctions(condition)) {
                if (pushablePredicate.test(node)) {
                    pushableConditions.add(node)
                } else {
                    nonPushableConditions.add(node)
                    // If we cannot push the whole condition we may be able to derive
                    // a partial condition that can be pushed.
                    val derivedCondition = partialFilterDerivationFunction(node)
                    if (pushablePredicate.test(derivedCondition)) {
                        for (derivedNode in RelOptUtil.conjunctions(derivedCondition)) {
                            pushableConditions.add(derivedNode)
                        }
                    }
                }
            }
            // Construct the two new conjunctions.
            val pushedConditions = RexUtil.composeConjunction(builder, pushableConditions, true)
            val keptConditions = RexUtil.composeConjunction(builder, nonPushableConditions, true)
            return Pair(pushedConditions, keptConditions)
        }

        /**
         * Determine if at least 1 condition in a filter can be converted to a target convention
         * based on the required predicate for if a RexNode is pushable
         * @param filter The filter we are seeking to split.
         * @param pushablePredicate A predicate used to evaluate if a condition can be pushed into
         * the target convention. To support a new convention you provide an implementation that
         * defines the rules for conversion.
         * @param partialFilterDerivationFunction A function that can be used to derive a new filter
         * that is a subset of the original filter and may be possible to push. This requires keeping
         * the original filter as well.
         * @return Is at least 1 condition in a filter pushable?
         */
        @JvmStatic
        fun isPartiallyPushableFilter(
            filter: Filter,
            pushablePredicate: Predicate<RexNode>,
            partialFilterDerivationFunction: (RexNode) -> RexNode = { it },
        ): Boolean {
            // You cannot split a filter that contains an over.
            if (filter.containsOver()) {
                return false
            }
            val (pushedConditions, _) =
                extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    pushablePredicate,
                    partialFilterDerivationFunction,
                )
            return pushedConditions != null
        }

        /**
         * Determine if every condition in a filter can be converted to a target convention
         * based on the required predicate for if a RexNode is pushable
         * @param filter The filter we are seeking to split.
         * @param pushablePredicate A predicate used to evaluate if a condition can be pushed into
         * the target convention. To support a new convention you provide an implementation that
         * defines the rules for conversion.
         * @param partialFilterDerivationFunction A function that can be used to derive a new filter
         * that is a subset of the original filter and may be possible to push. This requires keeping
         * the original filter as well.
         * @return Is every condition in a filter pushable?
         */
        @JvmStatic
        fun isFullyPushableFilter(
            filter: Filter,
            pushablePredicate: Predicate<RexNode>,
            partialFilterDerivationFunction: (RexNode) -> RexNode = { it },
        ): Boolean {
            // You cannot split a filter that contains an over.
            if (filter.containsOver()) {
                return false
            }
            val (pushedConditions, keptConditions) =
                extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    pushablePredicate,
                    partialFilterDerivationFunction,
                )
            return pushedConditions != null && keptConditions == null
        }
    }
}
