package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.table.CatalogTableImpl
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexNode

class SnowflakeFilter private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    condition: RexNode,
    private val catalogTable: CatalogTableImpl,
) : Filter(cluster, traitSet, input, condition), SnowflakeRel {

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): Filter {
        return SnowflakeFilter(cluster, traitSet, input, condition, catalogTable)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            condition: RexNode,
            catalogTable: CatalogTableImpl,
        ): SnowflakeFilter {
            // Note: Types may be lazily computed so use getRowType() instead of rowType
            val batchingProperty = ExpectedBatchingProperty.streamingIfPossibleProperty(input.getRowType())
            val newTraitSet = traitSet.replace(SnowflakeRel.CONVENTION).replace(batchingProperty)
            return SnowflakeFilter(cluster, newTraitSet, input, condition, catalogTable)
        }
    }

    override fun getCatalogTable(): CatalogTableImpl = catalogTable
}
