package com.bodosql.calcite.ir

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType

class BodoEngineTable(
    name: String,
    val rowType: RelDataType,
) : Variable(name) {
    constructor(name: String, rel: RelNode) : this(name, rel.rowType)
}
