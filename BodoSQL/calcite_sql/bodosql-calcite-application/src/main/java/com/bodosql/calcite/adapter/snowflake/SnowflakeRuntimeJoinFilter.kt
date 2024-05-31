package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.RuntimeJoinFilterBase
import com.bodosql.calcite.table.SnowflakeCatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery

class SnowflakeRuntimeJoinFilter private constructor(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    joinFilterIDs: List<Int>,
    filterColumns: List<List<Int>>,
    filterIsFirstLocations: List<List<Boolean>>,
    private val catalogTable: SnowflakeCatalogTable,
) : RuntimeJoinFilterBase(
        cluster,
        traits.replace(SnowflakeRel.CONVENTION),
        input,
        joinFilterIDs,
        filterColumns,
        filterIsFirstLocations,
    ),
    SnowflakeRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>,
    ): SnowflakeRuntimeJoinFilter {
        return copy(traitSet, sole(inputs), filterColumns)
    }

    /**
     * Return a new SnowflakeRuntimeJoinFilter with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<List<Int>>,
    ): SnowflakeRuntimeJoinFilter {
        return SnowflakeRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, newColumns, filterIsFirstLocations, catalogTable)
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
            catalogTable: SnowflakeCatalogTable,
        ): SnowflakeRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return SnowflakeRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, filterColumns, filterIsFirstLocations, catalogTable)
        }
    }

    override fun getCatalogTable(): SnowflakeCatalogTable = catalogTable
}
