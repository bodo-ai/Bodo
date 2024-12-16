package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr

internal data class PartitionKey(
    val field: String,
    val expr: Expr,
)
