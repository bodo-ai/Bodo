package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr
import org.apache.calcite.rel.type.RelDataType

internal data class Field(val name: String, val expr: Expr, val type: RelDataType)
