package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.table.CatalogTableImpl
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.util.ImmutableBitSet

class SnowflakeAggregate private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
    val catalogTable: CatalogTableImpl,
) :
    Aggregate(cluster, traitSet, listOf(), input, groupSet, groupSets, aggCalls),
    SnowflakeRel {

    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>
    ): Aggregate {
        return SnowflakeAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls, catalogTable)
    }

    override fun generatePythonConnStr(schema: String): String {
        val catalog = catalogTable.catalog as SnowflakeCatalogImpl
        return catalog.generatePythonConnStr(schema)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode,
            groupSet: ImmutableBitSet, groupSets: List<ImmutableBitSet>?, aggCalls: List<AggregateCall>,
            catalogTable: CatalogTableImpl
        ): SnowflakeAggregate {
            val newTraitSet = traitSet.replace(SnowflakeRel.CONVENTION)
            return SnowflakeAggregate(cluster, newTraitSet, input, groupSet, groupSets, aggCalls, catalogTable)
        }
    }
}
