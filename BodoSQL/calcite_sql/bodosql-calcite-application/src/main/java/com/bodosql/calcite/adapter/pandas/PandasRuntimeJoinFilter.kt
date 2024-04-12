package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.rel.core.RuntimeJoinFilterBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode

/**
 * Pandas implementation of a Runtime Join Filter.
 */
class PandasRuntimeJoinFilter private constructor(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    joinFilterID: Int,
    columns: List<Int>,
    isFirstLocation: List<Boolean>,
) : RuntimeJoinFilterBase(cluster, traits.replace(PandasRel.CONVENTION), input, joinFilterID, columns, isFirstLocation), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>,
    ): PandasRuntimeJoinFilter {
        return copy(traitSet, sole(inputs), columns)
    }

    /**
     * Return a new RuntimeJoinFilterBase with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<Int>,
    ): PandasRuntimeJoinFilter {
        return PandasRuntimeJoinFilter(cluster, traitSet, input, joinFilterID, newColumns, isFirstLocation)
    }

    /**
     * Match the input's batching property. This node can be used in streaming
     * and single batch.
     * @param inputBatchingProperty the input's batching property
     * @return the expected batching property of this node.
     */
    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        val inputVar = implementor.visitChild(this.input, 0)
        return implementor.buildStreaming(
            {
                    ctx ->
                initStateVariable(ctx)
            },
            {
                    ctx, _ ->
                val joinStateCache = ctx.builder().getJoinStateCache()
                val stateVar = joinStateCache.getStreamingJoinStateVariable(joinFilterID)
                if (stateVar == null) {
                    // If we don't have the state stored assume we have disabled
                    // streaming entirely and this is a no-op.
                    inputVar
                } else {
                    val columnsTuple = Expr.Tuple(this.columns.map { Expr.IntegerLiteral(it) })
                    val tupleVar = ctx.lowerAsMetaType(columnsTuple)
                    val isFirstLocationTuple = Expr.Tuple(this.isFirstLocation.map { Expr.BooleanLiteral(it) })
                    val isFirstLocationVar = ctx.lowerAsMetaType(isFirstLocationTuple)
                    val call =
                        Expr.Call(
                            "bodo.libs.stream_join.runtime_join_filter",
                            listOf(stateVar, inputVar, tupleVar, isFirstLocationVar),
                        )
                    val tableVar = ctx.builder().symbolTable.genTableVar()
                    val assign = Op.Assign(tableVar, call)
                    ctx.builder().add(assign)
                    BodoEngineTable(tableVar.emit(), this)
                }
            },
            {
                    ctx, stateVar ->
                deleteStateVariable(ctx, stateVar)
            },
        )
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        return UnusedStateVariable
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        // Do nothing
    }

    companion object {
        fun create(
            input: RelNode,
            joinFilterID: Int,
            columns: List<Int>,
            isFirstLocation: List<Boolean>,
        ): PandasRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return PandasRuntimeJoinFilter(cluster, traitSet, input, joinFilterID, columns, isFirstLocation)
        }
    }
}
