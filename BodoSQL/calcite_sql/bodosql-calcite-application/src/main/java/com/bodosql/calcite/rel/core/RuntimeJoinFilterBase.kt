package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRuntimeJoinFilter
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
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
 * For join ID, we store 1 value per left key in the source join. This ordering matters and is consistent as it
 * maps to each key location in the runtime join.
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

    companion object {
        /**
         * Generates the expression to pass in a tuple of runtime join filters
         * to an I/O call. The tuple contains tuples in the form
         * (joinState, filterColumns, isFirstBooleans).
         */
        fun getRuntimeJoinFilterTuple(
            ctx: BodoPhysicalRel.BuildContext,
            runtimeJoinFilters: List<RuntimeJoinFilterBase>,
        ): Expr {
            if (runtimeJoinFilters.isEmpty()) return Expr.None
            val rfjExprs: MutableList<Expr> = mutableListOf()
            // Iterate across each RuntimeJoinFilter node
            runtimeJoinFilters.forEach { rfjCollection ->
                rfjCollection.joinFilterIDs.forEachIndexed { idx, joinFilterId ->
                    val joinStateCache = ctx.builder().getJoinStateCache()
                    val (stateVar, keyLocations) = joinStateCache.getStreamingJoinInfo(joinFilterId)
                    val nKeyColumns = keyLocations.size
                    // If we don't have the state stored assume we have disabled
                    // streaming entirely and this is a no-op.
                    stateVar?.let {
                        val columnOrderedList = MutableList(nKeyColumns) { Expr.NegativeOne }
                        keyLocations.forEachIndexed { index, keyLocation ->
                            columnOrderedList[keyLocation] =
                                Expr.IntegerLiteral(rfjCollection.filterColumns[idx][index])
                        }
                        val columnsTuple = Expr.Tuple(columnOrderedList)
                        val tupleVar = ctx.lowerAsMetaType(columnsTuple)
                        val rfjExpr = Expr.Tuple(stateVar, tupleVar)
                        rfjExprs.add(rfjExpr)
                    }
                }
            }
            return Expr.Tuple(rfjExprs)
        }

        /**
         * Wrap a reader result in runtime join filter codegen
         *
         * @param rel The original RelNode generating the read call.
         * @param ctx The build context.
         * @param runtimeJoinFilters The filters being applied.
         * @param readerResult The expression generated from the reader call.
         */
        fun wrapResultInRuntimeJoinFilters(
            rel: RelNode,
            ctx: BodoPhysicalRel.BuildContext,
            runtimeJoinFilters: List<RuntimeJoinFilterBase>,
            readerResult: Expr,
        ): BodoEngineTable {
            val builder = ctx.builder()
            var result = readerResult
            runtimeJoinFilters.forEach { it ->
                val joinFilters =
                    it.joinFilterIDs.indices.map { idx ->
                        Triple(it.joinFilterIDs[idx], it.filterColumns[idx], it.filterIsFirstLocations[idx])
                    }
                val sortedJoinFilters = joinFilters.sortedByDescending { it.first }
                val joinFilterIDs = sortedJoinFilters.map { it.first }
                val columnsLists = sortedJoinFilters.map { it.second }
                val isFirstLocationLists = sortedJoinFilters.map { it.third }

                val rtjfResult =
                    BodoPhysicalRuntimeJoinFilter.generateRuntimeJoinFilterCode(
                        ctx,
                        joinFilterIDs,
                        columnsLists,
                        isFirstLocationLists,
                        result,
                    )
                rtjfResult?.let {
                    val tableChunkVar = builder.symbolTable.genTableVar()
                    builder.add(Op.Assign(tableChunkVar, rtjfResult))
                    result = tableChunkVar
                }
            }
            return BodoEngineTable(result.emit(), rel)
        }
    }
}
