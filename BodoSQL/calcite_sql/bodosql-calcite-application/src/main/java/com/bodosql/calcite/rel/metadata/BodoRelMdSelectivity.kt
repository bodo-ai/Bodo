package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.metadata.RelMdSelectivity
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import kotlin.math.max

class BodoRelMdSelectivity : RelMdSelectivity() {

    // Constant parameter to define what percent of estimated total groups we expect
    // each equality filter to keep in a Join.
    private val GROUP_SELECTIVITY_CONSTANT = 0.9

    /**
     * Determine if the predicate being used as a portion of the Join condition
     * can be used to extra better statistics via approx_count_distinct calls on the keys.
     */
    private fun isValidDistinctCountJoinPredicate(comparison: RexNode, leftCount: Int): Boolean {
        return if (comparison.kind == SqlKind.EQUALS) {
            val equalsCall = comparison as RexCall
            val firstOperand = equalsCall.operands[0]
            val secondOperand = equalsCall.operands[1]
            val bothInputRefs = firstOperand is RexInputRef && secondOperand is RexInputRef
            if (bothInputRefs) {
                val firstInputRef = firstOperand as RexInputRef
                val secondInputRef = secondOperand as RexInputRef
                val firstLeft = firstInputRef.index < leftCount
                val secondLeft = secondInputRef.index < leftCount
                firstLeft != secondLeft
            } else {
                false
            }
        } else {
            false
        }
    }

    /**
     * Determine the distinct count estimates for the keys used in the comparison. If the comparison is
     * not valid for this type of comparison then it returns null for each entry.
     *
     */
    private fun generateDistinctCountEstimate(rel: Join, mq: BodoRelMetadataQuery, comparison: RexNode): Pair<Double?, Double?> {
        val leftCount = rel.left.getRowType().fieldCount
        val valid = isValidDistinctCountJoinPredicate(comparison, leftCount)
        return if (valid) {
            val equalsCall = comparison as RexCall
            val firstOperand = equalsCall.operands[0] as RexInputRef
            val secondOperand = equalsCall.operands[1] as RexInputRef
            val (leftColumn, rightColumn) = if (firstOperand.index < leftCount) {
                Pair(firstOperand.index, secondOperand.index - leftCount)
            } else {
                Pair(secondOperand.index, firstOperand.index - leftCount)
            }
            Pair(mq.getColumnDistinctCount(rel.left, leftColumn), mq.getColumnDistinctCount(rel.right, rightColumn))
        } else {
            Pair(null, null)
        }
    }

    /**
     * Compute the selectivity for a Join based on the
     * ColumnDistinctCount query. This query attempts to compute
     * if any columns are guaranteed to be unique and otherwise
     * attempts to compute the approx_distinct_count() query
     * on each column in Snowflake.
     *
     * See https://bodo.atlassian.net/wiki/spaces/B/pages/1420722189/Join+Row+Count+Estimates
     * for the selectivity calculation.
     */
    private fun getJoinColumnDistinctCountBasedSelectivity(
        rel: Join,
        mq: BodoRelMetadataQuery,
        predicate: RexNode,
    ): Double {
        // If we have an AND fetch the operands. In any other case we can
        // only look at a single comparison. We do not add any insights for join.
        val comparisons: List<RexNode> = if (predicate.kind == SqlKind.AND) {
            (predicate as RexCall).operands
        } else {
            listOf(predicate)
        }
        // Selectivity for which we don't use any data
        var unsupportedSelectivity: Double = 1.0
        // Selectivity for which we use data
        var supportedSelectivity: Double = 1.0
        // Value to use to track the percent of groups we expect to keep.
        // We cap the maximum filter but not group selectivity.
        var groupSelectivity = 1.0
        for (comparison in comparisons) {
            // Generate the selectivity if we used the Calcite default.
            val defaultSelectivity = getSelectivity(rel as RelNode, mq, comparison) ?: 1.0
            val (leftDistinctCount, rightDistinctCount) = generateDistinctCountEstimate(rel, mq, comparison)
            if (leftDistinctCount == null && rightDistinctCount == null) {
                // The comparison is not supported or we have no usable
                unsupportedSelectivity *= defaultSelectivity
            } else {
                val leftDistinctNonNull = leftDistinctCount ?: 1.0
                val rightDistinctNonNull = rightDistinctCount ?: 1.0
                val proposedSelectivity = 1.0 / max(leftDistinctNonNull, rightDistinctNonNull)
                if ((leftDistinctCount == null || rightDistinctCount == null) && proposedSelectivity > defaultSelectivity) {
                    // If we only have data from one side of the table we only use it if it gives us a more restrictive
                    // estimate than the default. If its not then we assume the null info might be more restrictive.
                    unsupportedSelectivity *= defaultSelectivity
                } else {
                    supportedSelectivity *= proposedSelectivity
                    groupSelectivity *= GROUP_SELECTIVITY_CONSTANT
                }
            }
        }
        val maxSize = max(mq.getRowCount(rel.left), mq.getRowCount(rel.right))
        // Never let the selectivity drop below the smaller table.
        return max(supportedSelectivity * unsupportedSelectivity, 1.0 / maxSize) * groupSelectivity
    }

    override fun getSelectivity(
        rel: Join,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? {
        return if (rel.isSemiJoin) {
            super.getSelectivity(rel, mq, predicate)
        } else {
            // Utilize our custom metadata queries. If for some reason we have
            // a configuration issue, fall back to the default selectivity
            // for robustness.
            // TODO(njriasan): Update the no stats version of Join to be a more
            // reasonable estimate, such as taking the max like Dremio does
            // for hash join.
            if (mq is BodoRelMetadataQuery && predicate != null) {
                getJoinColumnDistinctCountBasedSelectivity(rel, mq, predicate)
            } else {
                getSelectivity(rel as RelNode, mq, predicate)
            }
        }
    }

    fun getSelectivity(
        rel: RelSubset,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? = getSelectivity(rel.getBestOrOriginal(), mq, predicate)
}
