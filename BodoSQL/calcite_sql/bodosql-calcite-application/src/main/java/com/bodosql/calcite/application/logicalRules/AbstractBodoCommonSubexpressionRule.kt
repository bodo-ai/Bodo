package com.bodosql.calcite.application.logicalRules

import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.RelBuilder

/**
 * <p>Performs maximal common subexpression elimination on a Project node using a dynamic
 * programming algorithm on all RexCall and RexOver nodes (excluding any RexCall nodes
 * that are inputs to a CASE statement, due to short-circuiting evaluation).</p>
 *
 * <p>The first step is to calculate `count[node]`: the number of times each potentially
 * eject-able RexNode appears in the Project (ignoring occurrences inside of CASE unless the
 * node also occurs outside of a CASE).</p>
 *
 * <p>The second step is to calculate `maxCount[node]`: the maximum value of `count[x]`
 * for node `x` and all of its descendants.</p>
 *
 * <p>The idea behind the algorithm is as follows: if `count[node] > 1` it means we should
 * eject that `node` at some point due to a common subexpression, but we should not necessarily
 * do so right away since if we eject `node` now we may lose opportunities for future ejection,
 * particularly if there are common subexpressions inside of `node` that can also be ejected.
 * The fix for this is to only eject `node` if `count[node] = maxCount[node]` is also true,
 * that way we can guarantee that there is no benefit to ejecting any descendants of `node`
 * before ejecting `node` itself.</p>
 *
 * <p>For example, consider this projection:</p>
 *
 * <code>
 *     Project(A=[(($0 + 1) * ($0 - 1)) + 1], B=[(($0 + 1) * ($0 - 1)) - 1], C=[$0 + 1])
 * </code>
 *
 * <p>If we ejected (($0 + 1) * ($0 - 1)), we would be led to with the following path and
 * lose an opportunity to eliminate the common subexpression with C:</p>
 *
 * <code>
 *     Project(A=[($1 + 1], B=[$1 - 1], C=[$0 + 1])
 *        Project(EXPR0=[$0 + 1], EXPR1=[($0 + 1) * ($0 - 1)])
 * </code>
 *
 * <p>However, with the algorithm, we'd be able to use the following counts:</p>
 *
 * <code>
 * ($0 + 1): count=3, maxCount=3
 * ($0 - 1): count=2, maxCount=2
 * (($0 + 1) * ($0 - 1)): count=2, maxCount=3
 * (($0 + 1) * ($0 - 1)) + 1: count=1, maxCount=3
 * (($0 + 1) * ($0 - 1)) - 1: count=1, maxCount=3
 * </code>
 *
 * <p>This would mean we choose to eject ($0 + 1) and ($0 - 1) as follows:</p>
 *
 * <code>
 *     Project(A=[($0 * $1) + 1], B=[($0 * $1) - 1], C=[$0])
 *        Project(EXPR0=[$0 + 1], EXPR1=[$0 - 1])
 * </code>
 *
 * <p>Then we could run the algorithm again to get our optimal common subexpression
 * elimination pattern:</p>
 *
 * <code>
 *     Project(A=[$1 + 1], B=[$1 - 1], C=[$0])
 *        Project(EXPR0=[$0], EXPR1=($0 * $1))
 *           Project(EXPR0=[$0 + 1], EXPR1=[$0 - 1])
 * </code>
 */
abstract class AbstractBodoCommonSubexpressionRule protected constructor(
    config: Config,
) : RelRule<AbstractBodoCommonSubexpressionRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall) {
        val project: Project = call.rel(0)
        val result = applySubexpressionElimination(call.builder(), project)
        if (result != null) {
            call.transformTo(result)
        }
    }

    interface Config : RelRule.Config

    companion object {
        @JvmStatic
        fun applySubexpressionElimination(
            originalBuilder: RelBuilder,
            project: Project,
        ): RelNode? {
            val builder = originalBuilder.transform { b: RelBuilder.Config -> b.withBloat(-1) }

            // Count how many times each non-trivial RexNode appears in the projection,
            // as well as the maximum for any of their descendants.
            val callCounter = RexCallCounter()
            project.projects.forEach {
                it.accept(callCounter)
            }

            // Create the basic list of child input refs.
            val childNodes: MutableList<RexNode> = mutableListOf()
            project.input.rowType.fieldList.forEachIndexed { idx, field ->
                childNodes.add(RexInputRef(idx, field.type))
            }

            // Use the dynamic programming algorithm to eject nodes from
            // the original projection into the child nodes & create
            // the new nodes without the removed layer of common subexpressions.
            val nodesToEject: MutableSet<RexNode> = mutableSetOf()
            project.projects.forEach {
                getNodesToEject(
                    callCounter.nodeCounts,
                    callCounter.caseCounts,
                    callCounter.maxCounts,
                    it,
                    nodesToEject,
                )
            }

            // Abort if we can not eject any nodes.
            if (nodesToEject.isEmpty()) {
                return null
            }

            val ejector = SubexpressionEjector(nodesToEject, childNodes)
            val newExprs = project.projects.map { it.accept(ejector) }

            builder.push(project.input)
            builder.project(childNodes)
            builder.project(newExprs, project.rowType.fieldNames)
            val newProject = builder.build()
            return newProject
        }

        private fun getNodesToEject(
            counts: Map<RexNode, Int>,
            caseCounts: Map<RexNode, Int>,
            maxCounts: Map<RexNode, Int>,
            node: RexNode,
            ejectSet: MutableSet<RexNode>,
        ) {
            if (node is RexCall) {
                val count = counts[node]?.let { it + (caseCounts[node] ?: 0) }
                val maxCount = maxCounts[node]
                if (count != null && maxCount != null && count > 1 && count >= maxCount) {
                    ejectSet.add(node)
                } else {
                    node.operands.forEach { getNodesToEject(counts, caseCounts, maxCounts, it, ejectSet) }
                }
            }
        }

        /**
         * Visitor used to count the number of occurrences of
         * RexCall and RexOver nodes from a collection of RexNodes,
         * such as a projection, to help identify opportunities for
         * common subexpression elimination. Does not keep track of
         * base-case nodes like literals or input refs.
         *
         * Also keeps track of the maximum number of counts between
         * the node and any of its descendants.
         *
         * Keeps track of nodes instead of CASE statements separately.
         */
        class RexCallCounter : RexVisitorImpl<Unit>(true) {
            val nodeCounts: MutableMap<RexNode, Int> = mutableMapOf()
            val caseCounts: MutableMap<RexNode, Int> = mutableMapOf()
            val maxCounts: MutableMap<RexNode, Int> = mutableMapOf()
            var insideCase = false

            override fun visitCall(call: RexCall) {
                if (insideCase) {
                    val selfCount = (caseCounts[call] ?: 0) + 1
                    caseCounts[call] = selfCount
                } else {
                    val selfCount = (nodeCounts[call] ?: 0) + 1
                    nodeCounts[call] = selfCount
                }
                val curCount = nodeCounts[call]?.let { it + (caseCounts[call] ?: 0) } ?: 0
                val maxCount = maxOf(call.operands.map { maxCounts[it] ?: 0 }.maxOrNull() ?: 0, curCount)
                maxCounts[call] = maxCount
                val oldInsideCase = insideCase
                if (call.kind == SqlKind.CASE) {
                    insideCase = true
                }
                val result = super.visitCall(call)
                if (call.kind == SqlKind.CASE) {
                    insideCase = oldInsideCase
                }
                return result
            }

            override fun visitOver(over: RexOver) {
                if (insideCase) {
                    val selfCount = (caseCounts[over] ?: 0) + 1
                    caseCounts[over] = selfCount
                } else {
                    val selfCount = (nodeCounts[over] ?: 0) + 1
                    nodeCounts[over] = selfCount
                }
                super.visitCall(over)
            }
        }

        class SubexpressionEjector(
            val ejectNodes: Set<RexNode>,
            val childNodes: MutableList<RexNode>,
        ) : RexShuttle() {
            private val pushedWindows = ejectNodes.any { RexOver.containsOver(it) }

            override fun visitCall(call: RexCall): RexNode {
                if (ejectNodes.contains(call)) {
                    return lookupOrAdd(call)
                }
                return super.visitCall(call)
            }

            override fun visitOver(over: RexOver): RexNode {
                if (ejectNodes.contains(over)) {
                    return lookupOrAdd(over)
                }
                val result = super.visitOver(over)
                // If the window function call was not modified, we should push it down as well if any
                // other window function calls were pushed.
                if (pushedWindows && result == over) {
                    return lookupOrAdd(over)
                }
                return result
            }

            private fun lookupOrAdd(node: RexNode): RexNode {
                childNodes.forEachIndexed { idx, childNode ->
                    if (childNode == node) return RexInputRef(idx, node.type)
                }
                val idx = childNodes.size
                childNodes.add(node)
                return RexInputRef(idx, node.type)
            }
        }
    }
}
