package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.table.CatalogTableImpl
import com.bodosql.calcite.traits.ExpectedBatchingProperty
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
    private val catalogTable: CatalogTableImpl,
) :
    Aggregate(cluster, traitSet, listOf(), input, groupSet, groupSets, aggCalls),
    SnowflakeRel {

    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): Aggregate {
        return SnowflakeAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls, catalogTable)
    }

    override fun getCatalogTable(): CatalogTableImpl = catalogTable

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            groupSet: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>?,
            aggCalls: List<AggregateCall>,
            catalogTable: CatalogTableImpl,
        ): SnowflakeAggregate {
            // Fetch types from keys and aggCalls.
            // Note: Types may be lazily computed so use getRowType() instead of rowType
            // and getType() instead of type.
            val inputType = input.getRowType()
            val keyTypes = groupSet.toList().map { i -> inputType.fieldList[i].getType() }
            // Derive the agg types.
            val aggTypes = aggCalls.map { a -> a.getType() }

            val batchingProperty = ExpectedBatchingProperty.streamingIfPossibleProperty(keyTypes + aggTypes)
            val newTraitSet = traitSet.replace(SnowflakeRel.CONVENTION).replace(batchingProperty)
            return SnowflakeAggregate(cluster, newTraitSet, input, groupSet, groupSets, aggCalls, catalogTable)
        }
    }
}
