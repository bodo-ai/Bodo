package com.bodosql.calcite.ir

import org.apache.calcite.rel.RelNode

class Dataframe(name: String, val rel: RelNode) : Expr() {
    val variable = Variable(name)
    override fun emit(): String = variable.emit()
}
