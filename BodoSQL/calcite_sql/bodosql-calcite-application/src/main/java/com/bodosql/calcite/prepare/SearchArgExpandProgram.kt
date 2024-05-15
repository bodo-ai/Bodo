package com.bodosql.calcite.prepare

import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.rel.core.cachePlanContainers.CacheNodeSingleVisitHandler
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.tools.Program
import org.apache.calcite.util.Sarg

class SearchArgExpandProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        val cacheHandler = CacheNodeSingleVisitHandler()
        val visitor = RelVisitor(cacheHandler)
        val result = rel.accept(visitor)
        while (cacheHandler.isNotEmpty()) {
            val cacheNode = cacheHandler.pop()
            val cacheRoot = cacheNode.cachedPlan.plan
            val cachePlanRoot = cacheRoot.rel
            cacheNode.cachedPlan.plan = cacheRoot.withRel(cachePlanRoot.accept(visitor))
        }
        return result
    }

    private class RelVisitor(private val cacheHandler: CacheNodeSingleVisitHandler) : RelShuttleImpl() {
        /**
         * Note the RelShuttleImpl() is design for logical nodes and therefore
         * isn't designed to run on Physical nodes. It does not have reflection
         * support and as a result we cannot add methods for our individual
         * implementations. We could replace this with a custom ReflectiveVisitor,
         * but this doesn't seem useful given time constraints.
         */
        override fun visit(node: RelNode): RelNode {
            // All Snowflake nodes must go through here.
            // This dispatches on the correct implementation.
            return when (node) {
                is Filter -> {
                    visit(node)
                }

                is Project -> {
                    visit(node)
                }
                is CachedSubPlanBase -> {
                    cacheHandler.add(node)
                    node
                }
                else -> {
                    super.visit(node)
                }
            }
        }

        fun visit(node: Filter): Filter {
            val newInput = node.input.accept(this)
            val rexBuilder = node.cluster.rexBuilder
            val visitor = RexVisitor(rexBuilder)
            val newCondition = node.condition.accept(visitor)
            // Flatten the condition
            return node.copy(node.traitSet, newInput, newCondition)
        }

        fun visit(node: Project): Project {
            val newInput = node.input.accept(this)
            val rexBuilder = node.cluster.rexBuilder
            val visitor = RexVisitor(rexBuilder)
            val newProjects = node.projects.map { it.accept(visitor) }
            // Make sure all the types match exactly, including nullability.
            val newProjectsCast =
                newProjects.withIndex().map {
                    val expectedType = node.projects[it.index].type
                    val value = it.value
                    if (it.value.type == expectedType) {
                        value
                    } else {
                        node.cluster.rexBuilder.makeCast(expectedType, value, true, false)
                    }
                }
            return node.copy(node.traitSet, newInput, newProjectsCast, node.rowType)
        }
    }

    private class RexVisitor(private val rexBuilder: RexBuilder) : RexShuttle() {
        override fun visitCall(call: RexCall): RexNode {
            return when (call.op.kind) {
                SqlKind.AND, SqlKind.OR -> {
                    val result = super.visitCall(call)
                    // Make sure AND/OR are fully flattened.
                    return RexUtil.flatten(rexBuilder, result)
                }
                SqlKind.SEARCH -> {
                    // Search argument we may need to expand. We can only use the Sarg
                    // if it has supported discrete values and is not inside a case statement.
                    val searchArgType = call.operands[0].type
                    val supportedType =
                        SqlTypeFamily.CHARACTER.contains(searchArgType) || SqlTypeFamily.INTEGER.contains(searchArgType)
                    val canKeep =
                        if (supportedType) {
                            val sargNode = call.getOperands()[1] as RexLiteral
                            val sargVal = sargNode.value!! as Sarg<*>
                            val rangeSet = sargVal.rangeSet
                            // Check that values are discrete.
                            var isDiscrete = !rangeSet.isEmpty
                            for (range in rangeSet.asRanges()) {
                                if (!range.hasLowerBound() || !range.hasUpperBound() || range.lowerEndpoint() != range.upperEndpoint()) {
                                    isDiscrete = false
                                    break
                                }
                            }
                            isDiscrete
                        } else {
                            false
                        }
                    if (canKeep) {
                        // In case operand 0 is a boolean with a search arg.
                        // TODO: Replace with IN to make the plans and code generation
                        // simpler to understand.
                        super.visitCall(call)
                    } else {
                        val newNode = RexUtil.expandSearch(rexBuilder, null, call)
                        // Accept the visitor in case we have nested calls.
                        newNode.accept(this)
                    }
                }
                else -> {
                    super.visitCall(call)
                }
            }
        }
    }
}
