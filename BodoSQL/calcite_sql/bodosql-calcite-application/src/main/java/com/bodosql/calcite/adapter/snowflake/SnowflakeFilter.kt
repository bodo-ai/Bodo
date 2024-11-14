package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.table.SnowflakeCatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

class SnowflakeFilter private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    condition: RexNode,
    private val catalogTable: SnowflakeCatalogTable,
) : Filter(cluster, traitSet.replace(SnowflakeRel.CONVENTION), input, condition),
    SnowflakeRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): Filter = SnowflakeFilter(cluster, traitSet, input, condition, catalogTable)

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
            condition: RexNode,
            catalogTable: SnowflakeCatalogTable,
        ): SnowflakeFilter = SnowflakeFilter(cluster, traitSet, input, condition, catalogTable)
    }

    override fun getCatalogTable(): SnowflakeCatalogTable = catalogTable
}
