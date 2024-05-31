package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.RuntimeJoinFilterBase
import com.bodosql.calcite.table.CatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery

class IcebergRuntimeJoinFilter private constructor(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    joinFilterIDs: List<Int>,
    filterColumns: List<List<Int>>,
    filterIsFirstLocations: List<List<Boolean>>,
    private val catalogTable: CatalogTable,
) : RuntimeJoinFilterBase(
        cluster,
        traits.replace(IcebergRel.CONVENTION),
        input,
        joinFilterIDs,
        filterColumns,
        filterIsFirstLocations,
    ),
    IcebergRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>,
    ): IcebergRuntimeJoinFilter {
        return copy(traitSet, sole(inputs), filterColumns)
    }

    /**
     * Return a new IcebergRuntimeJoinFilter with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<List<Int>>,
    ): IcebergRuntimeJoinFilter {
        return IcebergRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, newColumns, filterIsFirstLocations, catalogTable)
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
            input: RelNode,
            joinFilterIDs: List<Int>,
            filterColumns: List<List<Int>>,
            filterIsFirstLocations: List<List<Boolean>>,
            catalogTable: CatalogTable,
        ): IcebergRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return IcebergRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, filterColumns, filterIsFirstLocations, catalogTable)
        }
    }

    override fun getCatalogTable(): CatalogTable = catalogTable
}
