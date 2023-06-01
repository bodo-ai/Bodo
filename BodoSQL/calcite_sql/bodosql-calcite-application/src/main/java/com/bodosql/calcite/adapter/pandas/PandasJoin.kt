package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.google.common.collect.ImmutableList
import com.google.common.collect.ImmutableSet
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.metadata.RelMetadataQuery
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
        // Require streaming if we have enabled streaming.
        traitSet.containsIfApplicable(BatchingProperty.STREAMING)
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

    override fun emit(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        inputs: () -> List<Dataframe>
    ): Dataframe {
        TODO("Not yet implemented")
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        // Join conditions are still applied on the cross product of the inputs.
        // While we don't materialize all of these rows, the condition cost should
        // reflect that fact.
        val conditionRows = inputs.map { mq.getRowCount(it) }
            .reduce { a, b -> a * b }
        val conditionCost = condition.accept(RexCostEstimator)
            .multiplyBy(conditionRows)

        // We now want to compute the expected cost of producing this join's output.
        // We do this by taking the output rows and multiplying by the number
        // of rows we are estimated to produce. The join condition itself will influence
        // the estimated row count.
        val rows = mq.getRowCount(this)
        val outputCost = Cost(mem = mq.getAverageRowSize(this) ?: 0.0)
            .multiplyBy(rows)

        // Final cost is these two combined.
        val totalCost = conditionCost.plus(outputCost)
        return planner.makeCost(rows = rows, from = totalCost)
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
            val streamingTrait = BatchingProperty.STREAMING
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION).replace(streamingTrait)
            return PandasJoin(cluster, traitSet, left, right, condition, joinType)
        }

    }
}
