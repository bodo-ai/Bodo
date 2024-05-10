package com.bodosql.calcite.rel.core

import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.metadata.RelMetadataQuery

/**
 * Base implementation for a runtime join filter.
 * See the design here:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1632370739/Runtime+Join+Filters#BodoPhysicalRuntimeJoinFilter
 *
 */
open class RuntimeJoinFilterBase(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    val joinFilterIDs: List<Int>,
    val filterColumns: List<List<Int>>,
    val filterIsFirstLocations: List<List<Boolean>>,
) : SingleRel(cluster, traits, input) {
    /**
     * Return a new RuntimeJoinFilterBase with only a different set of columns.
     */
    open fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<List<Int>>,
    ): RuntimeJoinFilterBase {
        return RuntimeJoinFilterBase(cluster, traitSet, input, joinFilterIDs, newColumns, filterIsFirstLocations)
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        // Only display the new columns to avoid confusion in the plans.
        val displayedColumns =
            filterColumns.withIndex().map {
                    (idx, columns) ->
                columns.withIndex().filter { filterIsFirstLocations[idx][it.index] }.map { it.value }
            }
        val allKeysReady = filterColumns.map { colList -> colList.all { it != -1 } }
        return pw.item("input", getInput())
            .item("joinIDs", joinFilterIDs)
            .item("columnsList", displayedColumns)
            .itemIf("allKeysReady", allKeysReady, allKeysReady.any())
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost().multiplyBy(rows)
    }
}
