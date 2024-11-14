package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin
import com.bodosql.calcite.adapter.bodo.BodoPhysicalProject
import com.bodosql.calcite.rel.core.BodoPhysicalRelFactories
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.rules.JoinCommuteRule
import org.apache.calcite.tools.Program
import org.apache.calcite.tools.RelBuilder

/**
 * Apply any changes to the physical plan as dictated by the hints.
 */
object HintApplicationProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        val builder =
            BodoPhysicalRelFactories.BODO_PHYSICAL_BUILDER.create(
                rel.cluster,
                null,
            )
        val shuttle = Visitor(builder)
        return rel.accept(shuttle)
    }

    private class Visitor(
        private val builder: RelBuilder,
    ) : RelShuttleImpl() {
        /**
         * Note the RelShuttleImpl() is designed for logical nodes and therefore
         * isn't designed to run on Physical nodes. It does not have reflection
         * support and as a result we cannot add methods for our individual
         * implementations. We could replace this with a custom ReflectiveVisitor,
         * but this doesn't seem useful given time constraints
         */
        override fun visit(node: RelNode): RelNode {
            val visitedResult = super.visit(node)
            return if (visitedResult is BodoPhysicalJoin) {
                applyJoinHints(visitedResult)
            } else {
                visitedResult
            }
        }

        private fun applyJoinHints(join: BodoPhysicalJoin): RelNode {
            if (join.hints.isEmpty()) {
                return join
            }
            // We current only support BROADCAST and BUILD as hints. We
            // give broadcast priority over build.
            var hasLeftBroadcast = false
            var hasRightBroadcast = false
            var hasLeftBuild = false
            var hasRightBuild = false
            for (hint in join.hints) {
                if (hint.hintName == "BROADCAST") {
                    if (hint.listOptions.size == 1) {
                        val table = hint.listOptions[0]
                        if (table == "LEFT") {
                            hasLeftBroadcast = true
                        } else if (table == "RIGHT") {
                            hasRightBroadcast = true
                        }
                    }
                } else if (hint.hintName == "BUILD") {
                    if (hint.listOptions.size == 1) {
                        val table = hint.listOptions[0]
                        if (table == "LEFT") {
                            hasLeftBuild = true
                        } else if (table == "RIGHT") {
                            hasRightBuild = true
                        }
                    }
                }
            }
            // Broadcast an input if there is any broadcast hint and check for the hint location to
            // swap inputs.
            val (swapInputs, broadcast) =
                if (hasLeftBroadcast || hasRightBroadcast) {
                    val swap = hasLeftBroadcast && !hasRightBroadcast
                    val broadcast = true
                    Pair(swap, broadcast)
                } else {
                    val swap = hasLeftBuild && !hasRightBuild
                    val broadcast = false
                    Pair(swap, broadcast)
                }
            // Update the physical node information + clear the hints.
            return if (swapInputs) {
                val updatedOutput = JoinCommuteRule.swap(join, true, builder)
                if (updatedOutput == null) {
                    // If the swap fails, return the original
                    join
                } else {
                    // While we have swapped the inputs, we still need to update the hints and broadcast.
                    // JoinCommuteRule.swap will create a projection.
                    if (updatedOutput is BodoPhysicalProject && updatedOutput.getInput(0) is BodoPhysicalJoin) {
                        val newJoin = updatedOutput.getInput(0) as BodoPhysicalJoin
                        builder.push(newJoin.withHints(mutableListOf()).withBroadcastBuildSide(broadcast))
                        builder.project(updatedOutput.projects, updatedOutput.rowType.fieldNames).build()
                    } else {
                        // Return the original join if it doesn't match the expected pattern.
                        join
                    }
                }
            } else {
                join.withHints(mutableListOf()).withBroadcastBuildSide(broadcast)
            }
        }
    }
}
