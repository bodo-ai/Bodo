package com.bodosql.calcite.rel.core

import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.util.ImmutableBitSet

/**
 * Base flatten node for defining the cost model.
 */
open class FlattenBase(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    call: RexCall,
    callType: RelDataType,
    usedColOutputs: ImmutableBitSet,
    repeatColumns: ImmutableBitSet,
) : Flatten(cluster, traits, input, call, callType, usedColOutputs, repeatColumns) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        call: RexCall,
        callType: RelDataType,
        usedColOutputs: ImmutableBitSet,
        repeatColumns: ImmutableBitSet,
    ): FlattenBase = FlattenBase(cluster, traitSet, input, call, callType, usedColOutputs, repeatColumns)

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        val averageRowSize = mq.getAverageRowSize(this)
        val rowMultiplier = rows ?: 1.0
        val rowCost = averageRowSize?.times(rowMultiplier) ?: 0.0
        return planner.makeCost(rows = rows, mem = rowCost)
    }
}
