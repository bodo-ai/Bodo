package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.table.CatalogTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

class SnowflakeSort private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
    private val catalogTable: CatalogTable,
) :
    Sort(cluster, traitSet.replace(SnowflakeRel.CONVENTION), input, collation, offset, fetch), SnowflakeRel {

    override fun copy(
        traitSet: RelTraitSet,
        newInput: RelNode,
        newCollation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): Sort {
        return SnowflakeSort(cluster, traitSet, newInput, newCollation, offset, fetch, catalogTable)
    }

    override fun getCatalogTable(): CatalogTable = catalogTable

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost(cpu = 0.0, mem = 0.0, rows = rows)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            collation: RelCollation,
            offset: RexNode?,
            fetch: RexNode?,
            catalogTable: CatalogTable,
        ): SnowflakeSort {
            return SnowflakeSort(cluster, traitSet, input, collation, offset, fetch, catalogTable)
        }
    }
}
