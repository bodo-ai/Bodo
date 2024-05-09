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
    val joinFilterID: Int,
    val columns: List<Int>,
    val isFirstLocation: List<Boolean>,
) : SingleRel(cluster, traits, input) {
    /**
     * Return a new RuntimeJoinFilterBase with only a different set of columns.
     */
    open fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<Int>,
    ): RuntimeJoinFilterBase {
        return RuntimeJoinFilterBase(cluster, traitSet, input, joinFilterID, newColumns, isFirstLocation)
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        // Only display the new columns to avoid confusion in the plans.
        val displayedColumns = columns.withIndex().filter { isFirstLocation[it.index] }.map { it.value }
        return pw.item("input", getInput())
            .item("joinID", joinFilterID)
            .item("columns", displayedColumns)
            .itemIf("allKeysReady", true, columns.all { it != -1 })
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost().multiplyBy(rows)
    }
}
