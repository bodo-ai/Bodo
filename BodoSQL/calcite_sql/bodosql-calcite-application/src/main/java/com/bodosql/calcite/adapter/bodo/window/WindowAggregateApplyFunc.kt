package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr
import org.apache.calcite.rex.RexOver

internal fun interface WindowAggregateApplyFunc {
    fun emit(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr
}
