package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMdRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil.getSelectivity
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.Util
import kotlin.math.max
import kotlin.math.min

class BodoRelMdRowCount : RelMdRowCount() {

    // TODO: extend the rest of the functions in RelMdRowCount, to cover all possible future Snowflake rels,
    // and/or move the snowflakeRel handling to a separate row count handler that runs before this one.
    // https://bodo.atlassian.net/browse/BSE-878
    override fun getRowCount(rel: RelNode, mq: RelMetadataQuery?): Double? {
        return if (rel is SnowflakeRel) {
            // If the rel is a Snowflake Rel, get the row count directly from snowflake,
            // if appropriate.
            // Note that the way that this is currently set up,
            // snowflake table scans and filters are the only two Rels for which this is enabled.
            // Both will both independently
            // submit metadata queries to get their row count
            // For now, we're considering this ok, but if/when we increase the number of RelNodes allowed to independently
            // submit metadata queries, the number of unused metadata queries will also increase
            // Therefore, should we ever extend this, we should look into a more robust way
            // of limiting the number what metadata queries we submit to snowflake
            (rel as SnowflakeRel).tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
        } else {
            // Otherwise, just use the default
            super.getRowCount(rel, mq)
        }
    }

    fun getRowCount(rel: SnowflakeFilter, mq: RelMetadataQuery?): Double? {
        return rel.tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
    }

    /**
     * This is a copy of RelMdRowCount, but it also handles named parameters.
     */
    override fun getRowCount(rel: Sort, mq: RelMetadataQuery): Double? {
        var rowCount: Double = mq.getRowCount(rel.input) ?: return null

        // This is the only change, allowing RexCall (the node type
        // that holds named parameters
        if (rel.offset is RexDynamicParam || rel.offset is RexCall) {
            return rowCount
        }
        val offset = if (rel.offset == null) {
            0
        } else {
            RexLiteral.intValue(rel.offset)
        }

        rowCount = (rowCount - offset).coerceAtLeast(0.0)
        if (rel.fetch != null) {
            if (rel.fetch is RexDynamicParam || rel.fetch is RexCall) {
                return rowCount
            }
            val limit = RexLiteral.intValue(rel.fetch)
            if (limit < rowCount) {
                return limit.toDouble()
            }
        }
        return rowCount
    }

    /**
     * Implementation of join. For the join cases we support we provide our own
     * calculation to determine the expected row count by accounting for distinctness
     * information.
     *
     * Note: A previous implementation tried to implement this as selectivity,
     * but that is not safe because we may call getSelectivity after we have already
     * calculated the row count, so we need to isolate this to Join's row count.
     */
    override fun getRowCount(rel: Join, mq: RelMetadataQuery): Double? {
        if (!rel.joinType.projectsRight() || mq !is BodoRelMetadataQuery) {
            // Fall back to the default implementation for cases we don't support
            // or possible configuration issues.
            // TODO(njriasan): Update the no stats version of Join to be a more
            // reasonable estimate, such as taking the max like Dremio does
            // for hash join.
            return super.getRowCount(rel, mq)
        } else {
            // Note: Most of this is copied from RelMdUtil.getJoinRowCount
            // Row count estimates of 0 will be rounded up to 1.
            // So, use maxRowCount where the product is very small.
            // Row count estimates of 0 will be rounded up to 1.
            // So, use maxRowCount where the product is very small.
            val left = mq.getRowCount(rel.getLeft())
            val right = mq.getRowCount(rel.getRight())
            if (left == null || right == null) {
                return null
            }
            if (left <= 1.0 || right <= 1.0) {
                val max = mq.getMaxRowCount(rel)
                if (max != null && max <= 1.0) {
                    return max
                }
            }

            val selectivity = getJoinColumnDistinctCountBasedSelectivity(rel, mq, rel.condition)

            val innerRowCount = left * right * selectivity
            return when (rel.joinType) {
                JoinRelType.INNER -> innerRowCount
                JoinRelType.LEFT -> left * (1.0 - selectivity) + innerRowCount
                JoinRelType.RIGHT -> right * (1.0 - selectivity) + innerRowCount
                JoinRelType.FULL -> (left + right) * (1.0 - selectivity) + innerRowCount
                else -> throw Util.unexpected<JoinRelType>(rel.joinType)
            }
        }
    }

    // Implementation of the join calculation.

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
        predicate: RexNode?,
    ): Double {
        // If there is no predicate this is a cross join.
        if (predicate == null) {
            return 1.0
        }
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
            val defaultSelectivity = mq.getSelectivity(rel, comparison) ?: 1.0
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
        // Never let the selectivity drop below the smaller table,
        // and cap the selectivity to [0, 1.0]
        return min(max(supportedSelectivity * unsupportedSelectivity, 1.0 / maxSize) * groupSelectivity, 1.0)
    }
}
