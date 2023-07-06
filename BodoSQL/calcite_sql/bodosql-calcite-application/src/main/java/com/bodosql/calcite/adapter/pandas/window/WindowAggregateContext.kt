package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module

internal data class WindowAggregateContext(
    val builder: Module.Builder,
    val len: Expr,
    val orderKeys: List<Expr>,
    val bounds: Bounds,
)
