package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.table.CatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

class IcebergFilter private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    condition: RexNode,
    private val catalogTable: CatalogTable,
) : Filter(cluster, traitSet.replace(IcebergRel.CONVENTION), input, condition), IcebergRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): Filter {
        return IcebergFilter(cluster, traitSet, input, condition, catalogTable)
    }

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
            catalogTable: CatalogTable,
        ): IcebergFilter {
            return IcebergFilter(cluster, traitSet, input, condition, catalogTable)
        }
    }

    override fun getCatalogTable(): CatalogTable = catalogTable

    override fun containsIcebergSort(): Boolean {
        return (input as IcebergRel).containsIcebergSort()
    }
}
