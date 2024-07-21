package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.prepare.NonEqualityJoinFilterColumnInfo
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
    equalityFilterColumns: List<List<Int>>,
    equalityIsFirstLocations: List<List<Boolean>>,
    nonEqualityFilterInfo: List<List<NonEqualityJoinFilterColumnInfo>>,
    private val catalogTable: CatalogTable,
) : RuntimeJoinFilterBase(
        cluster,
        traits.replace(IcebergRel.CONVENTION),
        input,
        joinFilterIDs,
        equalityFilterColumns,
        equalityIsFirstLocations,
        nonEqualityFilterInfo,
    ),
    IcebergRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>,
    ): IcebergRuntimeJoinFilter {
        return copy(traitSet, sole(inputs), equalityFilterColumns, nonEqualityFilterInfo)
    }

    /**
     * Return a new IcebergRuntimeJoinFilter with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newEqualityColumns: List<List<Int>>,
        newNonEqualityColumns: List<List<NonEqualityJoinFilterColumnInfo>>,
    ): IcebergRuntimeJoinFilter {
        return IcebergRuntimeJoinFilter(
            cluster,
            traitSet,
            input,
            joinFilterIDs,
            newEqualityColumns,
            equalityIsFirstLocations,
            newNonEqualityColumns,
            catalogTable,
        )
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
            equalityFilterColumns: List<List<Int>>,
            equalityIsFirstLocations: List<List<Boolean>>,
            nonEqualityFilterInfo: List<List<NonEqualityJoinFilterColumnInfo>>,
            catalogTable: CatalogTable,
        ): IcebergRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return IcebergRuntimeJoinFilter(
                cluster,
                traitSet,
                input,
                joinFilterIDs,
                equalityFilterColumns,
                equalityIsFirstLocations,
                nonEqualityFilterInfo,
                catalogTable,
            )
        }
    }

    override fun getCatalogTable(): CatalogTable = catalogTable
}
