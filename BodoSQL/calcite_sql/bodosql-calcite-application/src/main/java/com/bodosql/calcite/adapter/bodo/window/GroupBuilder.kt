package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.BodoEngineTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexWindow

internal class GroupBuilder(
    private val cluster: RelOptCluster,
    private val input: BodoEngineTable,
    expr: RexOver,
    private val window: RexWindow,
) {
    private val exprs = mutableListOf(expr)

    /**
     * Determines whether two [RexOver] expressions can be performed together
     * in the same grouping through groupby.apply or groupby.window.
     *
     * In order to determine whether two window functions can be processed
     * in the same window, we check if their partition keys and their order
     * keys are the same.
     *
     * The group itself holds a [RexWindow], but the window should only be
     * used for these two attributes. Different [RexOver] nodes may have
     * different bounds attributes or other differences if they exist.
     */
    fun canAdd(expr: RexOver): Boolean =
        expr.window.partitionKeys.equals(window.partitionKeys) &&
            expr.window.orderKeys.equals(window.orderKeys)

    fun add(expr: RexOver): Int {
        exprs.add(expr)
        return exprs.lastIndex
    }

    fun build(): Group = Group(cluster, input, exprs, window)
}
