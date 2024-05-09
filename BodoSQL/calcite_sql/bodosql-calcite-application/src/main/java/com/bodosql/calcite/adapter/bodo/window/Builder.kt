package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.application.utils.Utils
import com.bodosql.calcite.ir.BodoEngineTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.sql.SqlKind

internal class Builder(val cluster: RelOptCluster, val input: BodoEngineTable) : RexShuttle() {
    private val overExprs = mutableListOf<RexOver>()
    private var index = 0

    override fun visitOver(over: RexOver): RexNode {
        overExprs.add(over)
        return RexLocalRef(index++, over.type)
    }

    override fun visitCall(call: RexCall): RexNode =
        when (call.op.kind) {
            SqlKind.CASE -> {
                // In some circumstances, calcite will optimize windowed aggregation
                // functions into a case statement that checks if the window is valid.
                // We don't really need this and can remove it.
                // This would likely be better done either in the parsing or planning
                // step but we're performing it here for now.
                if (Utils.isWindowedAggFn(call)) {
                    call.operands[1].accept(this)
                } else {
                    null
                }
            }
            else -> null
        } ?: super.visitCall(call)

    fun build(): WindowAggregate {
        val groups = mutableListOf<GroupBuilder>()
        val index =
            overExprs.map { expr ->
                // Find an existing window that has the same window.
                val compatibleGroup = groups.indexOfFirst { it.canAdd(expr) }
                if (compatibleGroup != -1) {
                    val index = groups[compatibleGroup].add(expr)
                    GroupIndex(group = compatibleGroup, index = index)
                } else {
                    groups.add(GroupBuilder(cluster, input, expr, expr.window))
                    GroupIndex(group = groups.lastIndex, index = 0)
                }
            }
        return WindowAggregate(
            groups = groups.map { it.build() },
            index = index,
        )
    }
}
