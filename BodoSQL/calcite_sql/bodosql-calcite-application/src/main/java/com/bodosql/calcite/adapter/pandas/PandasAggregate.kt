package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.AggregateBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.util.ImmutableBitSet

class PandasAggregate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : AggregateBase(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), input, groupSet, groupSets, aggCalls), PandasRel {

    override fun copy(traitSet: RelTraitSet, input: RelNode, groupSet: ImmutableBitSet, groupSets: List<ImmutableBitSet>?, aggCalls: List<AggregateCall>): PandasAggregate {
        return PandasAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls)
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

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.aggregateProperty(groupSets, aggCalls, getRowType())
    }

    companion object {

        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            groupSet: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>,
            aggCalls: List<AggregateCall>,
        ): PandasAggregate {
            return PandasAggregate(cluster, cluster.traitSet(), input, groupSet, groupSets, aggCalls)
        }
    }
}
