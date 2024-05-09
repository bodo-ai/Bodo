package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.bodo.RexCostEstimator
import com.bodosql.calcite.application.utils.RexNormalizer
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

open class FilterBase(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    child: RelNode,
    condition: RexNode,
) : Filter(cluster, traits, child, RexNormalizer.normalize(cluster.rexBuilder, condition)) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): FilterBase {
        return FilterBase(cluster, traitSet, input, condition)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val inputRows = mq.getRowCount(input)
        // Filter cost consists of two parts. The cost to compute the actual filter
        // and the cost to filter each column.
        val conditionCost = condition.accept(RexCostEstimator).multiplyBy(inputRows)
        // Compute the expected number of output rows
        val outputRows = mq.getRowCount(this)
        // We will assume the CPU cost to apply the boolean filter to each row is 0
        // and just consider the memory allocation of the new output.
        // val filterCost = Cost(mem = mq.getAverageRowSize(this) ?: 0.0).multiplyBy(outputRows)
        // TODO(jsternberg): Temporarily disable memory accounting costs for filters.
        // This sometimes causes some undesirable results with choosing not to use
        // a filter in a place where a filter might be useful. This is because we seem
        // to not have all of the correct planner rules included, specifically the
        // ProjectFilterTransposeRule. When we have transitioned to the VolcanoPlanner
        // as our primary development target and are making changes to that,
        // consider adding the filter's memory cost again.
        // If the memory cost is still unhelpful, just delete it entirely.
        val filterCost = planner.costFactory.makeZeroCost()
        val totalCost = (conditionCost + filterCost) as Cost
        return planner.makeCost(rows = outputRows, from = totalCost)
    }
}
