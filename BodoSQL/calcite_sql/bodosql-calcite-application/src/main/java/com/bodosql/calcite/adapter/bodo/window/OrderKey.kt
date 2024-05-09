package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.Expr

internal data class OrderKey(val field: String, val asc: Boolean, val nullsFirst: Boolean, val expr: Expr)

internal fun List<OrderKey>.fieldList(fn: (List<Expr>) -> Expr = Expr::List): Expr = fn(this.map { Expr.StringLiteral(it.field) })

internal fun List<OrderKey>.ascendingList(fn: (List<Expr>) -> Expr = Expr::List): Expr = fn(this.map { Expr.BooleanLiteral(it.asc) })

internal fun List<OrderKey>.nullPositionList(fn: (List<Expr>) -> Expr = Expr::List): Expr =
    fn(
        this.map {
            val nullPosition = if (it.nullsFirst) "first" else "last"
            Expr.StringLiteral(nullPosition)
        },
    )
