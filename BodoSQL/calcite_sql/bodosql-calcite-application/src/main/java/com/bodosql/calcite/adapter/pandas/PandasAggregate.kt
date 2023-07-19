package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.Utils.AggHelpers.aggContainsFilter
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.traits.BatchingProperty
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
        // Require streaming if we have enabled streaming and it makes sense for this node.
        if (getStreamingTrait(this.groupSets, this.aggCalls) == BatchingProperty.STREAMING)
            assert(traitSet.containsIfApplicable(BatchingProperty.STREAMING))
    }

    companion object {
        /**
         * Determine the streaming trait that can be used for an aggregation.
         *
         * @param groupSets the grouping sets used by the aggregation node.
         * If there is more than one grouping in the sets, or the singleton
         * grouping is a no-groupby aggregation, then the aggregation
         * node is not supported with streaming.
         * @param aggCallList the aggregations being applied to the data. If
         * any of the aggregations require a filter (e.g. for a PIVOT query)
         * then the aggregation node is not supported with streaming.
         */
        fun getStreamingTrait(groupSets: List<ImmutableBitSet>, aggCallList: List<AggregateCall>): BatchingProperty {
            // If streaming is disabled or grouping sets are being used, do not use streaming
            if (!RelationalAlgebraGenerator.enableGroupbyStreaming || groupSets.size != 1) return BatchingProperty.SINGLE_BATCH
            // If the aggregation is a no-groupby agg, do not use streaming
            val groupSet = groupSets[0]
            if (groupSet.cardinality() == 0) return BatchingProperty.SINGLE_BATCH
            // If any of the aggregations has a filter (e.g. for a PIVOT query), do not use streaming
            if (aggContainsFilter(aggCallList)) return BatchingProperty.SINGLE_BATCH
            return BatchingProperty.STREAMING
        }
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, groupSet: ImmutableBitSet, groupSets: List<ImmutableBitSet>?, aggCalls: List<AggregateCall>): PandasAggregate {
        return PandasAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }
}
