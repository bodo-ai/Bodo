package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.utils.BodoJoinConditionUtil
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.JoinBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import kotlin.math.ceil

class BodoPhysicalJoin(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    left: RelNode,
    right: RelNode,
    condition: RexNode,
    joinType: JoinRelType,
    val rebalanceOutput: Boolean,
    val joinFilterID: Int,
    val originalJoinFilterKeyLocations: List<Int>,
) : JoinBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), ImmutableList.of(), left, right, condition, joinType), BodoPhysicalRel {
    init {
        if (joinFilterID != -1) {
            assert(originalJoinFilterKeyLocations.isNotEmpty())
        }
    }

    override fun copy(
        traitSet: RelTraitSet,
        conditionExpr: RexNode,
        left: RelNode,
        right: RelNode,
        joinType: JoinRelType,
        semiJoinDone: Boolean,
    ): Join {
        assert(BodoJoinConditionUtil.isValidNode(conditionExpr))
        if (originalJoinFilterKeyLocations.isNotEmpty()) {
            assert(conditionExpr == condition)
        }
        return BodoPhysicalJoin(
            cluster,
            traitSet,
            left,
            right,
            conditionExpr,
            joinType,
            rebalanceOutput,
            joinFilterID,
            originalJoinFilterKeyLocations,
        )
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        return super.explainTerms(pw)
            .itemIf("rebalanceOutput", rebalanceOutput, rebalanceOutput)
            .itemIf("JoinID", joinFilterID, joinFilterID != -1)
    }

    fun withRebalanceOutput(rebalanceOutput: Boolean): BodoPhysicalJoin {
        return BodoPhysicalJoin(
            cluster,
            traitSet,
            left,
            right,
            condition,
            joinType,
            rebalanceOutput,
            joinFilterID,
            originalJoinFilterKeyLocations,
        )
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    /**
     * Get join build memory estimate for memory budget comptroller
     */
    fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // Streaming join needs the build table in memory, which is the right input
        val buildRows = mq.getRowCount(this.getRight())
        val averageBuildRowSize = mq.getAverageRowSize(this.getRight()) ?: 8.0
        // Account for hash table key/value pairs and group ids for each row (all int64)
        return ceil(buildRows * (averageBuildRowSize + (3 * 8))).toInt()
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            left: RelNode,
            right: RelNode,
            condition: RexNode,
            joinType: JoinRelType,
            rebalanceOutput: Boolean = false,
            joinFilterID: Int = -1,
            originalJoinFilterKeyLocations: List<Int> = listOf(),
        ): BodoPhysicalJoin {
            if (originalJoinFilterKeyLocations.isEmpty()) {
                assert(joinFilterID == -1)
            }
            return BodoPhysicalJoin(
                cluster,
                traitSet,
                left,
                right,
                condition,
                joinType,
                rebalanceOutput = rebalanceOutput,
                joinFilterID = joinFilterID,
                originalJoinFilterKeyLocations = originalJoinFilterKeyLocations,
            )
        }

        fun create(
            left: RelNode,
            right: RelNode,
            condition: RexNode,
            joinType: JoinRelType,
            rebalanceOutput: Boolean = false,
            joinFilterID: Int = -1,
            originalJoinFilterKeyLocations: List<Int> = listOf(),
        ): BodoPhysicalJoin {
            val cluster = left.cluster
            return create(
                cluster,
                cluster.traitSet(),
                left,
                right,
                condition,
                joinType,
                rebalanceOutput = rebalanceOutput,
                joinFilterID = joinFilterID,
                originalJoinFilterKeyLocations = originalJoinFilterKeyLocations,
            )
        }
    }
}
