package com.bodosql.calcite.ir


data class Variable(val name: String) : Expr() {
    override fun emit(): String = name

}
