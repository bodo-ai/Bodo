package com.bodosql.calcite.prepare

import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql2rel.SqlRexContext
import org.apache.calcite.sql2rel.SqlRexConvertlet
import org.apache.calcite.sql2rel.SqlRexConvertletTable

/**
 * Custom convertlet table for Bodo code generation. Handles custom functions
 * along with overriding the standard behavior for certain functions that can
 * be handled natively.
 */
class BodoConvertletTable(private val inner: SqlRexConvertletTable) : SqlRexConvertletTable {
    override fun get(call: SqlCall): SqlRexConvertlet? {
        return when (call.kind) {
            // LEAST and GREATEST default to expanding into case statements
            // in the standard convertlet table. We natively support these
            // operations so avoid converting them to another pattern.
            SqlKind.LEAST, SqlKind.GREATEST -> AliasConverter
            else -> inner.get(call)
        }
    }

    private object AliasConverter : SqlRexConvertlet {
        override fun convertCall(cx: SqlRexContext, call: SqlCall): RexNode {
            val operands = call.operandList.map { op -> cx.convertExpression(op) }
            return cx.rexBuilder.makeCall(call.operator, operands)
        }
    }
}
