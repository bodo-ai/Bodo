package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.UnionBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import kotlin.math.ceil

class BodoPhysicalUnion(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : UnionBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), inputs, all), BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
        all: Boolean,
    ): BodoPhysicalUnion {
        return BodoPhysicalUnion(cluster, traitSet, inputs, all)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    /**
     * Get union build memory estimate for memory budget comptroller
     */
    fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // UNION ALL is implemented as a ChunkedTableBuilder and doesn't need tracked in the memory
        // budget comptroller
        assert(!all)

        // Union distinct is same as aggregation
        val distinctRows = mq.getRowCount(this)
        val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
        return ceil(distinctRows * averageBuildRowSize).toInt()
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            all: Boolean,
        ): BodoPhysicalUnion {
            return BodoPhysicalUnion(cluster, cluster.traitSet(), inputs, all)
        }
    }
}
