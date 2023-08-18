package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.JoinBase
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
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
    val rebalanceOutput: Boolean,
) : JoinBase(cluster, traitSet, ImmutableList.of(), left, right, condition, joinType), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    constructor(
        cluster: RelOptCluster,
        traitSet: RelTraitSet,
        left: RelNode,
        right: RelNode,
        condition: RexNode,
        joinType: JoinRelType,
    ) : this(cluster, traitSet, left, right, condition, joinType, false)

    override fun copy(
        traitSet: RelTraitSet,
        conditionExpr: RexNode,
        left: RelNode,
        right: RelNode,
        joinType: JoinRelType,
        semiJoinDone: Boolean,
    ): Join {
        assert(PandasJoinRule.isValidNode(conditionExpr))
        return PandasJoin(cluster, traitSet, left, right, conditionExpr, joinType, rebalanceOutput)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }

    override fun explainTerms(pw: RelWriter?): RelWriter {
        return super.explainTerms(pw)
            .itemIf("rebalanceOutput", rebalanceOutput, rebalanceOutput)
    }

    fun withRebalanceOutput(rebalanceOutput: Boolean): PandasJoin {
        return PandasJoin(cluster, traitSet, left, right, condition, joinType, rebalanceOutput)
    }

    companion object {
        fun create(left: RelNode, right: RelNode, condition: RexNode, joinType: JoinRelType): PandasJoin {
            val cluster = left.cluster
            // Note: Types may be lazily computed so use getRowType() instead of rowType
            val leftTypes = ExpectedBatchingProperty.rowTypeToTypes(left.getRowType())
            val rightTypes = ExpectedBatchingProperty.rowTypeToTypes(right.getRowType())
            val streamingTrait = ExpectedBatchingProperty.streamingIfPossibleProperty(leftTypes + rightTypes)
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION).replace(streamingTrait)
            return PandasJoin(cluster, traitSet, left, right, condition, joinType)
        }
    }
}
