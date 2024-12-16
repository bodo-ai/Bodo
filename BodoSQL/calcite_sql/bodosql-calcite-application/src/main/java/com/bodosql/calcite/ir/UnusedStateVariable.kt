package com.bodosql.calcite.ir

object UnusedStateVariable : StateVariable("") {
    override fun emit(): String = throw UnsupportedOperationException("This state variable is not used and should not be emitted.")
}
