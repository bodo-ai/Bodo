package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.adapter.bodo.window.Builder
import com.bodosql.calcite.adapter.bodo.window.MultiResult
import com.bodosql.calcite.adapter.bodo.window.Result
import com.bodosql.calcite.ir.BodoEngineTable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver

/**
 * Extract the [RexOver] nodes from the list of expressions.
 *
 * The [RexOver] nodes will be replaced with [RexLocalRef] values that refer
 * to the index of the variables returned by [emit].
 *
 * [RexOver] calls that have the same partition key will be organized together
 * to reduce the number of windowed aggregate invocations.
 */
fun extractWindows(
    cluster: RelOptCluster,
    input: BodoEngineTable,
    exprs: List<RexNode>,
): MultiResult {
    val windowBuilder = Builder(cluster, input)
    val newExprs = exprs.map { exp -> exp.accept(windowBuilder) }
    return MultiResult(windowBuilder.build(), newExprs)
}

/**
 * Extract the [RexOver] nodes from the given expression.
 *
 * This function is identical to [extract] but only takes in one node instead
 * of multiple nodes.
 *
 * @see [extract]
 */
fun extractWindows(
    cluster: RelOptCluster,
    input: BodoEngineTable,
    expr: RexNode,
): Result {
    val (aggregate, exprs) = extractWindows(cluster, input, listOf(expr))
    return Result(aggregate, exprs[0])
}
