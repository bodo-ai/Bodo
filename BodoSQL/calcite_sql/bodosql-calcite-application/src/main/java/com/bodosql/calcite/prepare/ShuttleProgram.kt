package com.bodosql.calcite.prepare

import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttle
import org.apache.calcite.tools.Program

open class ShuttleProgram(private val shuttle: RelShuttle) : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode = rel.accept(shuttle)
}
