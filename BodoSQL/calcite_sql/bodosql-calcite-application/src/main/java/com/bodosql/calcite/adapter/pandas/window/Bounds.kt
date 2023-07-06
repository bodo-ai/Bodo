package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.ir.Expr

internal data class Bounds(val lower: Expr?, val upper: Expr?)
