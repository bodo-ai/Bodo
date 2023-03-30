package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.google.common.collect.ImmutableList
import com.google.common.collect.ImmutableSet
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rex.RexNode

class PandasJoin(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    left: RelNode,
    right: RelNode,
    condition: RexNode,
    joinType: JoinRelType,
) : Join(cluster, traitSet, ImmutableList.of(), left, right, condition,
    ImmutableSet.of(), joinType), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(
        traitSet: RelTraitSet,
        conditionExpr: RexNode,
        left: RelNode,
        right: RelNode,
        joinType: JoinRelType,
        semiJoinDone: Boolean
    ): Join {
        return PandasJoin(cluster, traitSet, left, right, condition, joinType)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(left: RelNode, right: RelNode, condition: RexNode, joinType: JoinRelType): PandasJoin {
            val cluster = left.cluster
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
            return PandasJoin(cluster, traitSet, left, right, condition, joinType)
        }
    }
}
