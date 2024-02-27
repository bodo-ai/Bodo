package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.pandas.AggCostEstimator
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.util.ImmutableBitSet

open class AggregateBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : Aggregate(cluster, traitSet, hints, input, groupSet, groupSets, aggCalls) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): AggregateBase {
        return AggregateBase(cluster, traitSet, hints, input, groupSet, groupSets, aggCalls)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        val distinctCost = AggCostEstimator.estimateInputCost(input)
        val funcCost =
            if (aggCalls.isEmpty()) {
                Cost()
            } else {
                aggCalls.map { aggCall -> AggCostEstimator.estimateFunctionCost(aggCall) }
                    .reduce { l, r -> l.plus(r) as Cost }
            }
        val cost = distinctCost + funcCost
        return planner.makeCost(from = cost).multiplyBy(rows)
    }
}
