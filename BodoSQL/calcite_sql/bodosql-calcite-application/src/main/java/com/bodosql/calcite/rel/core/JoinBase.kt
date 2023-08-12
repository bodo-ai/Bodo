package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.pandas.RexCostEstimator
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

abstract class JoinBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    left: RelNode,
    right: RelNode,
    condition: RexNode,
    joinType: JoinRelType,
) : Join(cluster, traitSet, hints, left, right, condition.accept(RexNormalizer(cluster.rexBuilder)), ImmutableSet.of(), joinType) {
    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        // Join conditions are still applied on the cross product of the inputs.
        // While we don't materialize all of these rows, the condition cost should
        // reflect that fact.
        val conditionRows = inputs.map { mq.getRowCount(it) }
            .reduce { a, b -> a * b }
        val conditionCost = condition.accept(RexCostEstimator)
            .multiplyBy(conditionRows)

        // For Bodo the build side is always the left input. This table
        // influences the total cost because the build table needs to be collected
        // before the probe side and is inserted into any hashmaps.
        val buildRows = mq.getRowCount(this.left)
        val averageBuildRowSize = mq.getAverageRowSize(this.left)
        // Add a multiplier to try ensure the build cost isn't too impactful.
        val buildCost = Cost(mem = averageBuildRowSize ?: 0.0).multiplyBy(buildRows).multiplyBy(0.3)

        // We now want to compute the expected cost of producing this join's output.
        // We do this by taking the output rows and multiplying by the number
        // of rows we are estimated to produce. The join condition itself will influence
        // the estimated row count.
        val rows = mq.getRowCount(this)
        val averageRowSize = mq.getAverageRowSize(this)
        val outputCost = Cost(mem = averageRowSize ?: 0.0).multiplyBy(rows)

        // Final cost is these three combined.
        val totalCost = conditionCost.plus(buildCost).plus(outputCost)
        return planner.makeCost(rows = rows, from = totalCost)
    }
}
