package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.MinRowNumberFilterBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.ImmutableBitSet

class BodoPhysicalMinRowNumberFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode,
    inputsToKeep: ImmutableBitSet,
) : MinRowNumberFilterBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), child, condition, inputsToKeep), BodoPhysicalRel {
    fun asBodoProjectFilter(): BodoPhysicalRel {
        val asBodoFilter = BodoPhysicalFilter(cluster, traitSet, input, condition)
        return if (inputsToKeep.cardinality() == input.rowType.fieldCount) {
            asBodoFilter
        } else {
            val projExprs =
                inputsToKeep.map {
                    RexInputRef(it, input.rowType.fieldList[it].type)
                }
            val parentProject = BodoPhysicalProject(cluster, traitSet, asBodoFilter, projExprs, getRowType())
            parentProject
        }
    }

    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): BodoPhysicalMinRowNumberFilter {
        return BodoPhysicalMinRowNumberFilter(cluster, traitSet, input, condition, inputsToKeep)
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    /**
     * Get memory estimate for memory budget comptroller.
     */
    fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        val averageBuildRowSize = mq.getAverageRowSize(this.getInput()) ?: 8.0
        val inRows = mq.getRowCount(this.getInput())
        return (averageBuildRowSize * inRows).toInt()
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            condition: RexNode,
            inputsToKeep: ImmutableBitSet,
        ): BodoPhysicalMinRowNumberFilter {
            return BodoPhysicalMinRowNumberFilter(cluster, traitSet, input, condition, inputsToKeep)
        }

        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            condition: RexNode,
            inputsToKeep: ImmutableBitSet,
        ): BodoPhysicalMinRowNumberFilter {
            val mq = cluster.metadataQuery
            val traitSet =
                cluster.traitSet().replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return create(cluster, traitSet, input, condition, inputsToKeep)
        }

        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            condition: RexNode,
        ): BodoPhysicalMinRowNumberFilter {
            val inputsToKeep = ImmutableBitSet.range(input.rowType.fieldCount)
            return create(cluster, input, condition, inputsToKeep)
        }
    }
}
