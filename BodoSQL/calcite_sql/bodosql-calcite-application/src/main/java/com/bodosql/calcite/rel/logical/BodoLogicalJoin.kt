package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.JoinBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rex.RexNode

class BodoLogicalJoin(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    left: RelNode,
    right: RelNode,
    condition: RexNode,
    joinType: JoinRelType,
) : JoinBase(cluster, traitSet, hints, left, right, condition, joinType) {
    override fun copy(
        traitSet: RelTraitSet,
        conditionExpr: RexNode,
        left: RelNode,
        right: RelNode,
        joinType: JoinRelType,
        semiJoinDone: Boolean,
    ): Join {
        return BodoLogicalJoin(cluster, traitSet, hints, left, right, conditionExpr, joinType)
    }

    companion object {
        @JvmStatic
        fun create(
            left: RelNode,
            right: RelNode,
            hints: List<RelHint>,
            condition: RexNode,
            joinType: JoinRelType,
        ): BodoLogicalJoin {
            val cluster = left.cluster
            val traitSet = cluster.traitSetOf(Convention.NONE)
            return BodoLogicalJoin(cluster, traitSet, hints, left, right, condition, joinType)
        }
    }
}
