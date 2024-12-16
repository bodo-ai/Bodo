package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.table.SnowflakeCatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.util.ImmutableBitSet

class SnowflakeAggregate private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
    private val catalogTable: SnowflakeCatalogTable,
) : Aggregate(cluster, traitSet.replace(SnowflakeRel.CONVENTION), listOf(), input, groupSet, groupSets, aggCalls),
    SnowflakeRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): Aggregate = SnowflakeAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls, catalogTable)

    override fun getCatalogTable(): SnowflakeCatalogTable = catalogTable

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost(cpu = 0.0, mem = 0.0, rows = rows)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            groupSet: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>?,
            aggCalls: List<AggregateCall>,
            catalogTable: SnowflakeCatalogTable,
        ): SnowflakeAggregate = SnowflakeAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls, catalogTable)
    }
}
