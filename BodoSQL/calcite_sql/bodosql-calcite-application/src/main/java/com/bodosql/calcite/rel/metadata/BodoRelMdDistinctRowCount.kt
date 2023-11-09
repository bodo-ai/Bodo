package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.rel.core.Flatten
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdDistinctRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.ImmutableBitSet
import kotlin.math.pow

class BodoRelMdDistinctRowCount : RelMdDistinctRowCount() {

    /**
     * Calculates an approximation of the number of distinct combinations of
     * several columns in a RelNode. This is useful for approximating how many
     * rows remain after an aggregation.
     *
     * The formula starts by finding the distinct row counts for each of the
     * relevant columns. If any cannot be found, return null. If there is only
     * one, then its distinct count is the answer. Otherwise, an approximation
     * can be made by multiplying all the distinct counts, as if a cartesian
     * product of all the grouping keys existed. The approximation can be refined
     * by assuming that not every combination exists. E.g. for columns (A, B, C),
     * we imagine only k1 percent of the combinations of unique rows  of A and B exist,
     * and for each of those only k2 percent of combinations with unique rows of C exist.
     * The simplest version of this calculation can be achieved by assuming that all k
     * values are the same constant between 0.0 and 1.0, here chosen to be 0.5. So,
     * for example, if we have columns with 40, 100, and 75 distinct rows respectively,
     * the cartesian product would assume that there are 40*100*75 = 300,000 distinct
     * combinations. The more selective calculation would assume that there are only
     *  (((40*100)*0.5)*75)*0.5 = 75,000 combinations.
     *
     * @param rel The RelNode being analyzed
     * @param mq Metadta query interface
     * @param groupKey the columns that are being used to group by
     * @return An approximation of the number of distinct combinations of the
     * columns specified by groupKey.
     */
    private fun cartesianApproximationDistinctRowCount(
        rel: RelNode,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
    ): Double? {
        // Fetch the row count first, since we need it to provide an upper bound for the output.
        val rowCount = mq.getRowCount(rel)
        return rowCount?.let {
            // Fetch the number of distinct rows for each column in the group key, and halt if none all of them are known.
            val requestColDistinct = groupKey.map { k -> (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel, k) }
            val colDistinct = requestColDistinct.filterNotNull()
            if (colDistinct.isEmpty()) return null

            // The first approximation of the number of distinct combinations of the keys is assuming
            // a cartesian product, i.e. every possible combination of the distinct rows from each of
            // the input keys exists in the output. This approximation is refined via multiplying by
            // a constant for every additional grouping key (excluding the unique ones) to factor in
            // the likelihood that not every combination will exist.
            var distinctApprox = colDistinct.reduce { a, b -> a * b }
            val cartesianSelectivity = 0.5
            val selectivityApplications = maxOf(0.0, colDistinct.filter { c -> c > 1.0 }.size - 1.0)
            distinctApprox *= cartesianSelectivity.pow(selectivityApplications)

            // The number of distinct rows has to be at least as many as the number of distinct rows
            // in any key, but cannot exceed the number of rows in the original input.
            var minDistinct = colDistinct.maxWithOrNull(compareBy { x -> x })!!

            // If we are missing some grouping keys, also lower bound by rowCount / 10
            if (colDistinct.size < requestColDistinct.size) minDistinct = maxOf(minDistinct, it / 10.0)
            minOf(maxOf(distinctApprox, minDistinct), it)
        }
    }

    fun getDistinctRowCount(
        rel: SingleRel,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        return mq.getDistinctRowCount(rel.input, groupKey, predicate)
    }

    override fun getDistinctRowCount(
        rel: RelNode,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        // Try the regular implementation, and return its answer if it is non-null.
        val regularCalculation = super.getDistinctRowCount(rel, mq, groupKey, predicate)
        regularCalculation?.let { return it }
        // Fall back to the cartesian approximation
        return cartesianApproximationDistinctRowCount(rel, mq, groupKey)
    }

    override fun getDistinctRowCount(
        rel: Filter,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        return cartesianApproximationDistinctRowCount(rel, mq, groupKey)
    }

    override fun getDistinctRowCount(
        rel: Project,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        return cartesianApproximationDistinctRowCount(rel, mq, groupKey)
    }

    fun getDistinctRowCount(
        rel: Flatten,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        return cartesianApproximationDistinctRowCount(rel, mq, groupKey)
    }
}
