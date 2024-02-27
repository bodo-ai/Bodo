package com.bodosql.calcite.application.utils

import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlCastFunction

/**
 * Force each call to be normalized in a RexNode.
 */
class RexNormalizer(private val rexBuilder: RexBuilder) : RexShuttle() {
    override fun visitOver(over: RexOver): RexNode {
        return over
    }

    override fun visitCall(call: RexCall): RexNode {
        // Use an array to pass a boolean by reference and update
        // inside visitList.
        val update = booleanArrayOf(false)
        var newOperands = visitList(call.operands, update)
        // The order of AND/OR can create issues for deterministic planning.
        if (call.kind == SqlKind.AND || call.kind == SqlKind.OR) {
            val sortedOperands = newOperands.sortedBy { it.toString() }
            if (sortedOperands != newOperands) {
                newOperands = sortedOperands
                update[0] = true
            }
        }
        return if (update[0]) {
            // Cast functions have a separate API and cannot use the generic makeCall.
            if (call.op is SqlCastFunction) {
                rexBuilder.makeCast(call.getType(), newOperands[0])
            } else {
                rexBuilder.makeCall(call.op, newOperands)
            }
        } else {
            call
        }
    }
}
