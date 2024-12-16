package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import org.apache.calcite.sql.type.BodoTZInfo

internal data class WindowAggregateContext(
    val builder: Module.Builder,
    val len: Expr,
    val orderKeys: List<Expr>,
    val bounds: Bounds,
    val defaultTZInfo: BodoTZInfo,
)
