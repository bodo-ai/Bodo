package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.UnionBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode

class PandasUnion(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : UnionBase(cluster, traitSet.replace(PandasRel.CONVENTION), inputs, all), PandasRel {

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>, all: Boolean): PandasUnion {
        return PandasUnion(cluster, traitSet, inputs, all)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            all: Boolean,
        ): PandasUnion {
            return PandasUnion(cluster, cluster.traitSet(), inputs, all)
        }
    }
}
