package com.bodosql.calcite.rel.core

import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

open class SortBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : Sort(cluster, traitSet, input, collation, offset, fetch) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): SortBase {
        return SortBase(cluster, traitSet, input, collation, offset, fetch)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        // Add a sorting cost:
        val cpu =
            if (collation.fieldCollations.isEmpty()) {
                0.0
            } else {
                1.0
            }
        val averageRowSize = mq.getAverageRowSize(this)
        val rowSize = averageRowSize?.times(rows) ?: 0.0
        return planner.makeCost(cpu = cpu, rows = rows, mem = rowSize)
    }
}
