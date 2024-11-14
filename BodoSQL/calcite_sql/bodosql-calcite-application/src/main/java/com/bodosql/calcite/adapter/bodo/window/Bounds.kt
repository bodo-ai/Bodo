package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr

internal data class Bounds(
    val lower: Expr?,
    val upper: Expr?,
)
