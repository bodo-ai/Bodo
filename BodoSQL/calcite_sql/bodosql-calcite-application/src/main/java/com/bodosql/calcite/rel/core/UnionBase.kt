package com.bodosql.calcite.rel.core

import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Union
import org.apache.calcite.rel.metadata.RelMetadataQuery

open class UnionBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : Union(cluster, traitSet, inputs, all) {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
        all: Boolean,
    ): UnionBase {
        return UnionBase(cluster, traitSet, inputs, all)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        val averageRowSize = mq.getAverageRowSize(this)
        val rowSize = averageRowSize?.times(rows) ?: 0.0
        val cpu =
            if (this.all) {
                // No CPU cost, only memory cost. In reality there is a "concat"
                // but we ignore this for now.
                0.0
            } else {
                // If all = False we have to do a drop_duplicates.
                0.7
            }
        return planner.makeCost(rows = rows, mem = rowSize, cpu = cpu)
    }
}
