package com.bodosql.calcite.rex

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexDynamicParam

class RexNamedParam(type: RelDataType, val paramName: String) : RexDynamicParam(type, -1) {
    init {
        digest = "@$paramName"
    }

    override fun getIndex(): Int {
        throw UnsupportedOperationException("Named parameters do not have an index")
    }

    override fun equals(obj: Any?): Boolean {
        return this === obj || (obj is RexNamedParam && type == obj.type && paramName == obj.paramName)
    }
}
