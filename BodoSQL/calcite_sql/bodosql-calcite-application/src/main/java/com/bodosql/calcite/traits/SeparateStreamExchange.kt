package com.bodosql.calcite.traits

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.SingleRel

class SeparateStreamExchange(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
) : SingleRel(
        cluster,
        traitSet.replace(BodoPhysicalRel.CONVENTION),
        input,
    ),
    BodoPhysicalRel {
    init {
        assert(traitSet.contains(BatchingProperty.STREAMING))
    }

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): RelNode? = SeparateStreamExchange(cluster, traitSet, inputs[0])

    override fun isEnforcer(): Boolean = true

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        // This should never be called, we need to handle SeparateStreamExchange
        // in the visitor itself due to needing to mess with the visitor state slightly
        TODO("Not yet implemented")
    }

    override fun splitCount(numRanks: Int): Int = 1

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty = BatchingProperty.STREAMING

    override fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty = BatchingProperty.SINGLE_BATCH

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }
}
