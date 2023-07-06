package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.ir.Expr

internal data class PartitionKey(val field: String, val expr: Expr)
