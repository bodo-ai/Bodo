package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalProject
import com.bodosql.calcite.application.logicalRules.AbstractBodoCommonSubexpressionRule
import com.bodosql.calcite.rel.core.BodoPhysicalRelFactories
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.tools.Program
import org.apache.calcite.tools.RelBuilder

/**
 * Program that runs the SubColumnEliminationRule on the final physical plan.
 * This is done both to ensure any sub column elimination is a heuristic step
 * and because the rule is too expensive to run on large plans in the volcano
 * planner.
 */
object SubColumnEliminationProgram : Program {
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
        override fun visit(node: RelNode): RelNode =
            if (node is BodoPhysicalProject) {
                val result = AbstractBodoCommonSubexpressionRule.applySubexpressionElimination(builder, node)
                if (result != null) {
                    visit(result)
                } else {
                    super.visit(node)
                }
            } else {
                super.visit(node)
            }
    }
}
