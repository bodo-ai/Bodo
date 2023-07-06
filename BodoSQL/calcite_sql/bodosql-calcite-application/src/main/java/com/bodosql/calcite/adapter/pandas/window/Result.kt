package com.bodosql.calcite.adapter.pandas.window

import org.apache.calcite.rex.RexNode

data class Result(val windowAggregate: WindowAggregate, val expr: RexNode)
