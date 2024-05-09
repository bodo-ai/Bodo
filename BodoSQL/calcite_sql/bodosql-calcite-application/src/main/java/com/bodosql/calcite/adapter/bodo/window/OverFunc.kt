package com.bodosql.calcite.adapter.bodo.window

import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexWindow
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator

/**
 * Wrapper around [RexOver] to facilitate overwriting and controlling access
 * to certain attributes.
 *
 * In particular, Calcite doesn't allow us to create [RexOver] without a [RexBuilder]
 * to change the operands and it doesn't implement [RexCall.clone]. We need to
 * overwrite the operands so we reference the correct local ref inside of the generated
 * dataframe so we use this class to store that information.
 */
internal class OverFunc internal constructor(val over: RexOver, val operands: List<RexNode>) {
    val kind: SqlKind get() = over.kind
    val op: SqlOperator get() = over.op
    val window: RexWindow get() = over.window
}
