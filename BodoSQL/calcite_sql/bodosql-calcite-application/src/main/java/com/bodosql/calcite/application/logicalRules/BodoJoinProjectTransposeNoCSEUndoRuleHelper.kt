package com.bodosql.calcite.application.logicalRules

import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.util.ImmutableBitSet

/**
 * Selection of helper functions that are used exclusively in BodoJoinProjectTransposeNoCSEUndoRule.
 * This file exists because some functionality that I needed for the rule was very difficult to write in Java,
 * and converting the entire rule to Kotlin caused issues with the immutables library.
 */
class BodoJoinProjectTransposeNoCSEUndoRuleHelper {
    companion object {

        /**
         * Returns true if there exists a non-trivial expression that can be pulled above the join, that is not
         * used in the condition of the join.
         *
         * @param joinConditionUsages bitset containing the input indicies used by the join
         * @param leftProject The left input to the join. null if not a project.
         * @param rightProject The right input to the join. null if not a project.
         * @param leftJoinCount The number of fields in the left input of the join.
         */
        @JvmStatic
        fun bodoJoinProjectTransposeRuleCanMakeChange(joinConditionUsages: ImmutableBitSet, leftProject: Project?, rightProject: Project?, leftJoinCount: Int): Boolean {
            assert(leftProject != null || rightProject != null)

            // Find all input project expressions and their indices
            val leftProjectExprs = leftProject?.projects ?: listOf()
            val rightProjectExprs = rightProject?.projects ?: listOf()
            val leftProjectExprsWithIndex = leftProjectExprs.mapIndexed { idx, node -> Pair(idx, node) }
            val rightProjectExprsWithIndex = rightProjectExprs.mapIndexed { idx, node -> Pair(idx + leftJoinCount, node) }

            // If there is any non-trivial projectExpression that is NOT used in the join condition,
            // then we can pull it above the Join safely.
            return leftProjectExprsWithIndex.plus(rightProjectExprsWithIndex).any { idxAndNodePair ->
                !joinConditionUsages.contains(idxAndNodePair.first) && idxAndNodePair.second !is RexInputRef
            }
        }
    }
}
