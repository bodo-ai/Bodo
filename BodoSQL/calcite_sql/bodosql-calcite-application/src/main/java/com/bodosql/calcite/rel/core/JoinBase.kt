package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.bodo.RexCostEstimator
import com.bodosql.calcite.application.utils.RexNormalizer
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.google.common.collect.ImmutableSet
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.Util

abstract class JoinBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    left: RelNode,
    right: RelNode,
    condition: RexNode,
    joinType: JoinRelType,
) : Join(cluster, traitSet, hints, left, right, RexNormalizer.normalize(cluster.rexBuilder, condition), ImmutableSet.of(), joinType) {
    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        // Join conditions are still applied on the cross product of the inputs.
        // While we don't materialize all of these rows, the condition cost should
        // reflect that fact.
        val conditionRows =
            inputs
                .map { mq.getRowCount(it) }
                .reduce { a, b -> a * b }
        val conditionCost =
            condition
                .accept(RexCostEstimator)
                .multiplyBy(conditionRows)

        // Compute the memory cost from each of the inputs. Join must use every column
        val probeRows = mq.getRowCount(this.left)
        val averageProbeRowSize = mq.getAverageRowSize(this.left) ?: 8.0
        val buildRows = mq.getRowCount(this.right)
        val averageBuildRowSize = mq.getAverageRowSize(this.right) ?: 8.0
        // Build cost should be higher than probe cost because we must make the hash table.
        // If we have a RIGHT JOIN and similar sizes we may want to do a LEFT JOIN because
        // there is overhead to tracking the misses.
        val baseBuildMultiplier = 1.3
        val buildMultiplier =
            if (this.joinType == JoinRelType.RIGHT) {
                baseBuildMultiplier * 1.0
            } else {
                baseBuildMultiplier
            }
        val buildCost = Cost(mem = averageBuildRowSize).multiplyBy(buildRows).multiplyBy(buildMultiplier)
        val probeCost = Cost(mem = averageProbeRowSize).multiplyBy(probeRows)

        // We now want to compute the expected cost of producing this join's output.
        // We do this by taking the output rows and multiplying by the number
        // of rows we are estimated to produce. The join condition itself will influence
        // the estimated row count.
        val rows = mq.getRowCount(this)
        val averageRowSize = mq.getAverageRowSize(this) ?: 8.0
        val outputCost = Cost(mem = averageRowSize).multiplyBy(rows)

        // Final cost is these three combined.
        val totalCost = conditionCost.plus(buildCost).plus(probeCost).plus(outputCost)
        return planner.makeCost(rows = rows, from = totalCost)
    }

    override fun estimateRowCount(mq: RelMetadataQuery): Double = Util.first(mq.getRowCount(this), 1.0)
}
