package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.pandas.PandasCostEstimator
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.application.operatorTables.AggOperatorTable
import com.bodosql.calcite.rel.core.Flatten
import com.bodosql.calcite.rel.core.RowSample
import com.bodosql.calcite.rel.core.TableFunctionScanBase
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.core.TableFunctionScan
import org.apache.calcite.rel.metadata.RelMdRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet
import org.apache.calcite.util.NumberUtil.multiply
import org.apache.calcite.util.Util
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

class BodoRelMdRowCount : RelMdRowCount() {
    // TODO: extend the rest of the functions in RelMdRowCount, to cover all possible future Snowflake rels,
    // and/or move the snowflakeRel handling to a separate row count handler that runs before this one.
    // https://bodo.atlassian.net/browse/BSE-878
    override fun getRowCount(
        rel: RelNode,
        mq: RelMetadataQuery?,
    ): Double? {
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
            rel.tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
        } else {
            // Otherwise, just use the default
            super.getRowCount(rel, mq)
        }
    }

    fun getRowCount(
        rel: RowSample,
        mq: RelMetadataQuery,
    ): Double? {
        val inputRowCount = mq.getRowCount(rel.input)
        return inputRowCount?.let { min(it, rel.rowSamplingParameters.numberOfRows.toDouble()) }
    }

    fun getRowCount(
        rel: SnowflakeFilter,
        mq: RelMetadataQuery?,
    ): Double? {
        return rel.tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
    }

    fun getRowCount(
        rel: Flatten,
        mq: RelMetadataQuery?,
    ): Double? {
        // For flatten, assume that each row of arrays has a certain number of elements
        // on average. However, flatten also drops null rows, so assume not every
        // row is copied over.
        var inputRowCount = mq?.getRowCount(rel.input)
        return inputRowCount?.times(PandasCostEstimator.AVG_ARRAY_ENTRIES_PER_ROW)
    }

    fun getRowCount(
        rel: TableFunctionScan,
        mq: RelMetadataQuery,
    ): Double? {
        return (rel as TableFunctionScanBase).estimateRowCount(mq)
    }

    /**
     * This is a copy of RelMdRowCount, but it also handles named parameters.
     */
    override fun getRowCount(
        rel: Sort,
        mq: RelMetadataQuery,
    ): Double? {
        var rowCount: Double = mq.getRowCount(rel.input) ?: return null

        // This is the only change, allowing RexCall (the node type
        // that holds named parameters
        if (rel.offset is RexDynamicParam || rel.offset is RexCall) {
            return rowCount
        }
        val offset =
            if (rel.offset == null) {
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
    override fun getRowCount(
        rel: Join,
        mq: RelMetadataQuery,
    ): Double? {
        val origLeft = mq.getRowCount(rel.left)
        val origRight = mq.getRowCount(rel.right)
        if (origLeft == null || origRight == null) {
            return null
        }
        // Row count estimates of < 1 will be rounded up to 1.
        val left = max(origLeft, 1.0)
        val right = max(origRight, 1.0)
        if (!rel.joinType.projectsRight() || mq !is BodoRelMetadataQuery) {
            // Note: Bodo doesn't support semi or anti joins (which is this
            // path). As a placeholder we opt to match the default case we have.
            //
            // Match the dremio default of selecting the larger table.
            return max(origLeft, origRight)
        } else {
            // Note: Most of this is copied from RelMdUtil.getJoinRowCount
            // Use maxRowCount where the product is very small.
            if (left <= 1.0 || right <= 1.0) {
                val max = mq.getMaxRowCount(rel)
                if (max != null && max <= 1.0) {
                    return max
                }
            }
            // If we have a cross join just return the cross product.
            if (rel.condition == null || rel.condition.isAlwaysTrue) {
                return left * right
            }

            return getDistinctnessBasedJoinRowCount(rel, mq, left, right)
        }
    }

    // Implementation of the join calculation.

    /**
     * Determine if the predicate being used as a portion of the Join condition
     * can be used to extra better statistics via approx_count_distinct calls on the keys.
     */
    private fun isValidDistinctCountJoinPredicate(
        comparison: RexNode,
        leftCount: Int,
    ): Boolean {
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
    private fun generateDistinctCountEstimate(
        rel: Join,
        mq: BodoRelMetadataQuery,
        comparison: RexNode,
    ): Pair<Double?, Double?> {
        val leftCount = rel.left.getRowType().fieldCount
        val valid = isValidDistinctCountJoinPredicate(comparison, leftCount)
        return if (valid) {
            val equalsCall = comparison as RexCall
            val firstOperand = equalsCall.operands[0] as RexInputRef
            val secondOperand = equalsCall.operands[1] as RexInputRef
            val (leftColumn, rightColumn) =
                if (firstOperand.index < leftCount) {
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
     * Compute the expected row count for a Join based on the
     * ColumnDistinctCount query. The function creates a distinction between the
     * equijoin case and the non-equijoin case. For the non-equijoin case, while
     * there may be some exceptions involving OR, we generally expect as a HEURISTIC
     * the total join size will grow. In contrast, when we have an equijoin we
     * expect the total join size to be no bigger than the larger table (the join should
     * be filtering) unless we have information directly rejected this.
     *
     * In this function we implement the functionality defined here:
     * https://bodo.atlassian.net/wiki/spaces/B/pages/1420722189/Join+Row+Count+Estimates
     *
     * At a high level we determine the uniqueness of each column by querying the
     * mq.getColumnDistinctCount() information to estimate the number of possible values for
     * the join. Then we use the information to estimate the number of matches and the resulting
     * output size. This information also contains special handling for when we don't have this
     * information for one or both tables, so please consult the design doc for more information.
     *
     * For functions with multiple filters we cap the impact of the distinctness results once
     * we reach the maximum number of distinct rows possible for a table (no larger than the size).
     * Beyond this threshold we apply a heuristic fixed sized filter to account for how we generally
     * expect having more filters to result in a smaller output size.
     *
     * @param rel The join whose size we are estimating.
     * @param mq The metadata query used to determine column uniqueness.
     * @param leftSize The expected number of rows in the left table.
     * @param rightSize The expected number of rows in the right table.
     * @return The expected row count for the output of the join.
     */
    private fun getDistinctnessBasedJoinRowCount(
        rel: Join,
        mq: BodoRelMetadataQuery,
        leftSize: Double,
        rightSize: Double,
    ): Double {
        // Look at all the clauses in the condition. In the future we may only want to look
        // at individual keys but this ensures we keep each equality clause.
        val comparisons: List<RexNode> =
            if (rel.condition.kind == SqlKind.AND) {
                (rel.condition as RexCall).operands
            } else {
                listOf(rel.condition)
            }
        // Selectivity impact of non equality conditions
        var nonEqualitySelectivity = 1.0
        // Determine the number of distinct entries. We use a list
        // so we can determine the "most impactful filter"
        var leftDistinctList: MutableList<Double> = ArrayList()
        var rightDistinctList: MutableList<Double> = ArrayList()
        // Track how many equality conditions lack any stats.
        var equalityNoStatsCount = 0
        for (comparison in comparisons) {
            if (comparison.kind == SqlKind.EQUALS) {
                val (leftDistinctCount, rightDistinctCount) = generateDistinctCountEstimate(rel, mq, comparison)
                // Compute a calculation if we have information about either table.
                if (leftDistinctCount != null || rightDistinctCount != null) {
                    val leftDistinctNonNull = leftDistinctCount ?: computeDefaultUniqueCount(leftSize)
                    val rightDistinctNonNull = rightDistinctCount ?: computeDefaultUniqueCount(rightSize)
                    leftDistinctList.add(leftDistinctNonNull)
                    rightDistinctList.add(rightDistinctNonNull)
                } else {
                    equalityNoStatsCount += 1
                }
            } else {
                // For non-equijoin just use the calcite default estimate.
                nonEqualitySelectivity *= mq.getSelectivity(rel, comparison) ?: 1.0
            }
        }
        if (equalityNoStatsCount > 0) {
            // If there is a comparison for which we have no information assume the table should be no larger than the larger
            // table, so we add this computation to our result.
            val noStatsDistinctCount = min(leftSize, rightSize)
            leftDistinctList.add(noStatsDistinctCount)
            rightDistinctList.add(noStatsDistinctCount)
            // Track that we have added this filter.
            equalityNoStatsCount -= 1
        }
        // Order by most impactful filters
        val jointDistinct = leftDistinctList.zip(rightDistinctList)
        // The formula for applying a single filter is:
        // EXPECTED_OVERLAP_FACTOR * min(left_distinct, right_distinct) * leftSize * rightSize / (left_distinct * right_distinct)
        // For sorting purposes, EXPECTED_OVERLAP_FACTOR, leftSize and rightSize are all constants, so we can just compute
        // min(left_distinct, right_distinct) / (left_distinct * right_distinct)
        // which is equivalent to 1 / max(left_distinct, right_distinct)
        val sortedJoinDistinct = jointDistinct.sortedByDescending { vals -> 1 / max(vals.first, vals.second) }
        // Apply the impact of each filter until either table has the maximum distinctness.
        var leftDistinctTotal = 1.0
        var rightDistinctTotal = 1.0
        var matchingDistinctTotal = 1.0
        var numFiltersApplied = 0
        val minTableSize = min(leftSize, rightSize)
        for (i in sortedJoinDistinct.indices) {
            numFiltersApplied += 1
            val distinctParts = sortedJoinDistinct[i]
            val leftDistinct = distinctParts.first
            val rightDistinct = distinctParts.second
            val matchingDistinct = min(leftDistinct, rightDistinct)
            leftDistinctTotal = min(leftDistinctTotal * leftDistinct, leftSize)
            rightDistinctTotal = min(rightDistinctTotal * rightDistinct, rightSize)
            matchingDistinctTotal = min(matchingDistinctTotal * matchingDistinct, minTableSize)
            // Cap the distinct values at the size of the table.
            if (leftDistinctTotal >= leftSize || rightDistinctTotal >= rightSize) {
                break
            }
        }

        // Determine the impact of all other filters
        val extraEqualitySelectivity =
            noStatsEqualityComparisonFactor.pow(((sortedJoinDistinct.size - numFiltersApplied) + equalityNoStatsCount).toDouble())
        val innerRowCount =
            expectedOverlapFactor * matchingDistinctTotal * leftSize * rightSize *
                extraEqualitySelectivity * nonEqualitySelectivity / (leftDistinctTotal * rightDistinctTotal)
        // Now estimate a left and right selectivity for outer join.
        // For non-equality or comparisons for which we have no stats assume both tables are equally impacted.
        val leftSelectivity =
            (matchingDistinctTotal / leftDistinctTotal) * sqrt(expectedOverlapFactor) *
                sqrt(nonEqualitySelectivity) * sqrt(extraEqualitySelectivity)
        val rightSelectivity =
            (matchingDistinctTotal / rightDistinctTotal) * sqrt(expectedOverlapFactor) *
                sqrt(nonEqualitySelectivity) * sqrt(extraEqualitySelectivity)
        return when (rel.joinType) {
            JoinRelType.INNER -> innerRowCount
            JoinRelType.LEFT -> leftSize * (1.0 - leftSelectivity) + innerRowCount
            JoinRelType.RIGHT -> rightSize * (1.0 - rightSelectivity) + innerRowCount
            JoinRelType.FULL -> leftSize * (1.0 - leftSelectivity) + rightSize * (1.0 - rightSelectivity) + innerRowCount
            else -> throw Util.unexpected(rel.joinType)
        }
    }

    // Constant parameter to define what percent of estimated total groups we expect
    // to keep for the final join.
    private val expectedOverlapFactor = 0.90

    // Constant parameter to check the impact of additional equality comparisons for
    // which we have no statistics or are not dominant.
    private val noStatsEqualityComparisonFactor = 0.70

    /**
     * Compute the default assumed unique count for a table without uniqueness estimates
     * but a given expected total size.
     *
     * Currently, we just assume the average group has size 1000.
     */
    private fun computeDefaultUniqueCount(tableSize: Double): Double {
        val defaultGroupSize = 1000.0
        return max(1.0, tableSize / defaultGroupSize)
    }

    /**
     * Estimates the row count after a MIN_ROW_NUMBER_FILTER based on the partition keys.
     */
    private fun getMinRowNumFilterRowCount(
        windowCond: RexOver,
        input: RelNode,
        mq: RelMetadataQuery,
    ): Double {
        // rowCount is the cardinality of the partition by columns, fallback to divided by 10
        // if anything goes wrong.
        val totalRows = mq.getRowCount(input)
        val default = totalRows / 10
        val partitionKeys: MutableList<Int> = mutableListOf()
        windowCond.window.partitionKeys.forEach {
            if (it is RexInputRef) {
                partitionKeys.add(it.index)
            } else {
                return default
            }
        }
        val partitionCountEstimate = mq.getDistinctRowCount(input, ImmutableBitSet.of(partitionKeys), null) ?: default
        // Cap the output at a certain ratio of the input so the row count at least decreases a little.
        return minOf(partitionCountEstimate, totalRows * 0.999)
    }

    override fun getRowCount(
        rel: Filter,
        mq: RelMetadataQuery,
    ): Double? {
        // Separate the conditions into window functions vs others
        val conjunctions = RelOptUtil.conjunctions(rel.condition)
        val windowConds: MutableList<RexOver> = mutableListOf()
        val otherConds: MutableList<RexNode> = mutableListOf()
        conjunctions.forEach {
            if (it is RexOver) {
                windowConds.add(it)
            } else {
                otherConds.add(it)
            }
        }

        // If there is not exactly 1 OVER condition and it is not an MRNF, fall back to the regular implementation
        if (otherConds.any { RexOver.containsOver(it) } || windowConds.size != 1) return super.getRowCount(rel, mq)
        val windowCond = windowConds[0]
        if (windowCond.aggOperator.name != AggOperatorTable.MIN_ROW_NUMBER_FILTER.name) return super.getRowCount(rel, mq)

        // Estimate how many rows remain based on the # of combinations of the partition keys
        val mrnfRowCount = getMinRowNumFilterRowCount(windowCond, rel.input, mq)
        if (otherConds.size == 0) return mrnfRowCount
        val otherConditionsCombined = otherConds[0]
        return multiply(mrnfRowCount, mq.getSelectivity(rel.input, otherConditionsCombined))
    }
}
