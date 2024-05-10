package com.bodosql.calcite.adapter.bodo

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
 * Bodo physical implementation of a Runtime Join Filter.
 */
class BodoPhysicalRuntimeJoinFilter private constructor(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    joinFilterIDs: List<Int>,
    filterColumns: List<List<Int>>,
    filterIsFirstLocations: List<List<Boolean>>,
) : RuntimeJoinFilterBase(
        cluster,
        traits.replace(BodoPhysicalRel.CONVENTION),
        input,
        joinFilterIDs,
        filterColumns,
        filterIsFirstLocations,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>,
    ): BodoPhysicalRuntimeJoinFilter {
        return copy(traitSet, sole(inputs), filterColumns)
    }

    /**
     * Return a new RuntimeJoinFilterBase with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newColumns: List<List<Int>>,
    ): BodoPhysicalRuntimeJoinFilter {
        return BodoPhysicalRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, newColumns, filterIsFirstLocations)
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
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        val inputVar = implementor.visitChild(this.input, 0)
        return implementor.buildStreaming(
            BodoPhysicalRel.ProfilingOptions(timeStateInitialization = false),
            {
                    ctx ->
                initStateVariable(ctx)
            },
            {
                    ctx, _ ->
                val joinStateCache = ctx.builder().getJoinStateCache()
                var currentTable: BodoEngineTable = inputVar
                for (i in joinFilterIDs.indices) {
                    val joinFilterID = joinFilterIDs[i]
                    val columns = filterColumns[i]
                    val isFirstLocation = filterIsFirstLocations[i]
                    val stateVar = joinStateCache.getStreamingJoinStateVariable(joinFilterID)
                    // If we don't have the state stored assume we have disabled
                    // streaming entirely and this is a no-op.
                    if (stateVar != null) {
                        val columnsTuple = Expr.Tuple(columns.map { Expr.IntegerLiteral(it) })
                        val tupleVar = ctx.lowerAsMetaType(columnsTuple)
                        val isFirstLocationTuple = Expr.Tuple(isFirstLocation.map { Expr.BooleanLiteral(it) })
                        val isFirstLocationVar = ctx.lowerAsMetaType(isFirstLocationTuple)
                        val call =
                            Expr.Call(
                                "bodo.libs.stream_join.runtime_join_filter",
                                listOf(stateVar, currentTable, tupleVar, isFirstLocationVar),
                            )
                        val tableVar = ctx.builder().symbolTable.genTableVar()
                        val assign = Op.Assign(tableVar, call)
                        ctx.builder().add(assign)
                        currentTable = BodoEngineTable(tableVar.emit(), this)
                    }
                }
                currentTable
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
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        return UnusedStateVariable
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        // Do nothing
    }

    companion object {
        fun create(
            input: RelNode,
            joinFilterIDs: List<Int>,
            filterColumns: List<List<Int>>,
            filterIsFirstLocations: List<List<Boolean>>,
        ): BodoPhysicalRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return BodoPhysicalRuntimeJoinFilter(cluster, traitSet, input, joinFilterIDs, filterColumns, filterIsFirstLocations)
        }
    }
}
