package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.JoinCondVisitor.isBodoHashJoin
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.traits.BatchingProperty
import com.google.common.collect.ImmutableList
import com.google.common.collect.ImmutableSet
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
) : Join(cluster, traitSet, ImmutableList.of(), left, right, condition,
    ImmutableSet.of(), joinType), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    constructor(
        cluster: RelOptCluster,
        traitSet: RelTraitSet,
        left: RelNode,
        right: RelNode,
        condition: RexNode,
        joinType: JoinRelType
    ) : this(cluster, traitSet, left, right, condition, joinType, false)

    override fun copy(
        traitSet: RelTraitSet,
        conditionExpr: RexNode,
        left: RelNode,
        right: RelNode,
        joinType: JoinRelType,
        semiJoinDone: Boolean
    ): Join {
        assert(PandasJoinRule.isValidNode(conditionExpr))
        return PandasJoin(cluster, traitSet, left, right, conditionExpr, joinType, rebalanceOutput)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
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
            val streamingTrait = getStreamingTrait(condition, left)
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION).replace(streamingTrait)
            return PandasJoin(cluster, traitSet, left, right, condition, joinType)
        }

        /**
         * Determine the streaming Trait for a newly created Join.
         * We support Streaming for hash joins and otherwise
         * require Single-Batch.
         */
        fun getStreamingTrait(condition: RexNode, left: RelNode): BatchingProperty {
            // Note we call getRowType and getFieldNames explicitly because types are lazily evaluated.
            val isHashJoin = isBodoHashJoin(condition, left.getRowType().getFieldNames().size)
            //TODO: re-enable once codegen is added
//            return if (isHashJoin) BatchingProperty.STREAMING else BatchingProperty.SINGLE_BATCH
            return BatchingProperty.SINGLE_BATCH
        }
    }
}
