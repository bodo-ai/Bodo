package com.bodosql.calcite.adapter.common

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import java.util.function.Predicate

/**
 * Common utilities for converting a Filter from logical to a convention
 * (or across conventions) by splitting a filter into a component supported
 * in the new convention and those that cannot be pushed down.
 */
class FilterSplitUtils {
    companion object {
        /**
         * @param condition The conditions to a filter that are being checked to see if it is a
         * conjunction of conditions that can be partially pushed into a target convention.
         * @param builder The RexBuilder used to generate new conditions.
         * @param pushablePredicate A predicate used to evaluate if a condition can be pushed into
         * the target convention. To support a new convention you provide an implementation that
         * defines the rules for conversion.
         * @return A pair of conditions representing the subset of the condition that can be pushed into
         * the new dialect, and the subset that can not.
         */
        @JvmStatic
        fun extractPushableConditions(
            condition: RexNode,
            builder: RexBuilder,
            pushablePredicate: Predicate<RexNode>,
        ): Pair<RexNode?, RexNode?> {
            // Identify which of the conditions in the conjunction are pushable.
            val (pushableConditions, nonPushableConditions) = RelOptUtil.conjunctions(condition).partition { pushablePredicate.test(it) }
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
         * @return Is at least 1 condition in a filter pushable?
         */
        @JvmStatic
        fun isPartiallyPushableFilter(
            filter: Filter,
            pushablePredicate: Predicate<RexNode>,
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
         * @return Is every condition in a filter pushable?
         */
        @JvmStatic
        fun isFullyPushableFilter(
            filter: Filter,
            pushablePredicate: Predicate<RexNode>,
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
                )
            return pushedConditions != null && keptConditions == null
        }
    }
}
