package com.bodosql.calcite.traits

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.plan.Cost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.metadata.RelMetadataQuery

class CombineStreamsExchange(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) : SingleRel(cluster,  traits,  input), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
        assert(traits.contains(BatchingProperty.SINGLE_BATCH))
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): RelNode? {
        return CombineStreamsExchange(cluster, traitSet, inputs[0])
    }

    override fun isEnforcer(): Boolean {
        return true
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        // This should never be called, we need to handle CombineStreamsExchange
        // in the visitor itself due to needing to mess with the visitor state slightly
        TODO("Not yet implemented")
    }

    override fun splitCount(numRanks: Int): Int {
        return 1
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        val rows = mq.getRowCount(this)
        val averageRowSize = mq.getAverageRowSize(this)
        val rowSize = if (averageRowSize == null) {
            0.0
        } else {
            averageRowSize * rows
        }
        // No CPU cost, only memory cost. In reality there is a "concat",
        // but this is fixed cost.
        return Cost(mem = rowSize, rows = rows)
    }
}
