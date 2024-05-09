package com.bodosql.calcite.adapter.bodo.window

import org.apache.calcite.rex.RexNode

data class MultiResult(val windowAggregate: WindowAggregate, val exprs: List<RexNode>)
