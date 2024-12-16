package com.bodosql.calcite.ir

open class Variable(
    val name: String,
) : Expr() {
    override fun emit(): String = name
}
