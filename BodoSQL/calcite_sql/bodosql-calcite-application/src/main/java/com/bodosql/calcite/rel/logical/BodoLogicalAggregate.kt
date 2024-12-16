package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.AggregateBase
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.util.ImmutableBitSet

class BodoLogicalAggregate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : AggregateBase(cluster, traitSet, hints, input, groupSet, groupSets, aggCalls) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): BodoLogicalAggregate = BodoLogicalAggregate(cluster, traitSet, hints, input, groupSet, groupSets, aggCalls)

    companion object {
        @JvmStatic
        fun create(
            input: RelNode,
            hints: List<RelHint>,
            groupSet: ImmutableBitSet,
            groupSets: ImmutableList<ImmutableBitSet>,
            aggCalls: List<AggregateCall>,
        ): BodoLogicalAggregate {
            val cluster = input.cluster
            val traitSet = cluster.traitSet().replace(Convention.NONE)
            return BodoLogicalAggregate(cluster, traitSet, hints, input, groupSet, groupSets, aggCalls)
        }
    }
}
