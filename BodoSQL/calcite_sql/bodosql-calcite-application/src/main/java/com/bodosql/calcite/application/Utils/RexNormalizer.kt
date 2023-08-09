package com.bodosql.calcite.application.Utils

import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.sql.SqlKind

/**
 * Force each call to be normalized in a RexNode.
 */
class RexNormalizer(private val rexBuilder: RexBuilder) : RexShuttle() {
    override fun visitCall(call: RexCall): RexNode {
        val operands = call.operands.map { op -> op.accept(this) }
        return when (call.kind) {
            SqlKind.AND, SqlKind.OR -> {
                // The order of AND/OR doesn't really matter but can
                // differ in plans due to different planner rule orders.
                // Use the digest of each operand to sort. The digest
                // should be deterministic.
                val sorted = operands.sortedBy { it.toString() }
                rexBuilder.makeCall(call.op, sorted)
            }
            else -> call
        }
    }
}
