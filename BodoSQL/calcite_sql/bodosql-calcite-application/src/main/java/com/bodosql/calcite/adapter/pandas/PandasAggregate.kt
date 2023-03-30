package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.util.ImmutableBitSet

class PandasAggregate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : Aggregate(cluster, traitSet, ImmutableList.of(), input, groupSet, groupSets, aggCalls), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, groupSet: ImmutableBitSet, groupSets: List<ImmutableBitSet>?, aggCalls: List<AggregateCall>): PandasAggregate {
        return PandasAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }
}
