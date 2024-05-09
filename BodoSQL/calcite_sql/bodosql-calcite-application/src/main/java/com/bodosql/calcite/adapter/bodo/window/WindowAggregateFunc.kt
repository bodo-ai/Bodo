package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr

internal fun interface WindowAggregateFunc {
    fun emit(
        call: OverFunc,
        resolve: OperandResolver,
    ): Expr
}
