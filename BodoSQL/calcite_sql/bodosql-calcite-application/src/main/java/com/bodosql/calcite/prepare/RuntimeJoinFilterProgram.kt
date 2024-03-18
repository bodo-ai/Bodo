package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.pandas.PandasJoin
import com.bodosql.calcite.adapter.pandas.PandasRuntimeJoinFilter
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.Join
import org.apache.calcite.tools.Program

object RuntimeJoinFilterProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        return if (RelationalAlgebraGenerator.enableRuntimeJoinFilters) {
            val shuttle = RuntimeJoinFilterShuttle()
            rel.accept(shuttle)
        } else {
            rel
        }
    }

    private class RuntimeJoinFilterShuttle() : RelShuttleImpl() {
        private var joinFilterKey: Int = 0

        override fun visit(rel: RelNode): RelNode {
            return if (rel is PandasJoin) {
                visit(rel)
            } else {
                super.visit(rel)
            }
        }

        fun visit(join: Join): RelNode {
            // If we have a RIGHT or Inner join we can generate
            // a runtime join filter.
            return if (!join.joinType.generatesNullsOnRight()) {
                val info = join.analyzeCondition()
                val columns = info.leftKeys
                if (columns.isEmpty()) {
                    super.visit(join)
                } else {
                    val leftInput = join.left.accept(this)
                    val rightInput = join.right.accept(this)
                    // We can generate a runtime join filter. We use the current ID as the
                    val runtimeFilter =
                        PandasRuntimeJoinFilter.create(
                            leftInput,
                            joinFilterKey,
                            columns,
                            MutableList(columns.size) { true },
                        )
                    val result = PandasJoin.create(runtimeFilter, rightInput, join.condition, join.joinType, joinFilterKey = joinFilterKey)
                    joinFilterKey++
                    result
                }
            } else {
                super.visit(join)
            }
        }
    }
}
