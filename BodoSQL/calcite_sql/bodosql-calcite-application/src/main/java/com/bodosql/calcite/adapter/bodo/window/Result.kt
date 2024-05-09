package com.bodosql.calcite.adapter.bodo.window

import org.apache.calcite.rex.RexNode

data class Result(val windowAggregate: WindowAggregate, val expr: RexNode)
