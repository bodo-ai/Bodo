package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.prepare.NonEqualityJoinFilterColumnInfo
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
    equalityFilterColumns: List<List<Int>>,
    equalityIsFirstLocations: List<List<Boolean>>,
    nonEqualityFilterInfo: List<List<NonEqualityJoinFilterColumnInfo>>,
) : RuntimeJoinFilterBase(
        cluster,
        traits.replace(BodoPhysicalRel.CONVENTION),
        input,
        joinFilterIDs,
        equalityFilterColumns,
        equalityIsFirstLocations,
        nonEqualityFilterInfo,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalRuntimeJoinFilter = copy(traitSet, sole(inputs), equalityFilterColumns, nonEqualityFilterInfo)

    /**
     * Return a new RuntimeJoinFilterBase with only a different set of columns.
     */
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        newEqualityColumns: List<List<Int>>,
        newNonEqualityColumns: List<List<NonEqualityJoinFilterColumnInfo>>,
    ): BodoPhysicalRuntimeJoinFilter =
        BodoPhysicalRuntimeJoinFilter(
            cluster,
            traitSet,
            input,
            joinFilterIDs,
            newEqualityColumns,
            equalityIsFirstLocations,
            newNonEqualityColumns,
        )

    /**
     * Match the input's batching property. This node can be used in streaming
     * and single batch.
     * @param inputBatchingProperty the input's batching property
     * @return the expected batching property of this node.
     */
    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty =
        ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        // TODO: Consider moving to separate stages, 1 per filter.
        val stage =
            OutputtingStageEmission(
                { ctx, _, table ->
                    var currentTable: BodoEngineTable = table!!

                    // zip lists of joinFilterID, filterColumns, isFirstLocation, sort by joinID
                    val joinFilters =
                        joinFilterIDs.indices.map {
                            Triple(joinFilterIDs[it], equalityFilterColumns[it], equalityIsFirstLocations[it])
                        }
                    val sortedJoinFilters = joinFilters.sortedByDescending { it.first }
                    val joinFilterIDs = sortedJoinFilters.map { it.first }
                    val columnsLists = sortedJoinFilters.map { it.second }
                    val isFirstLocationLists = sortedJoinFilters.map { it.third }

                    val rtjfExpr = generateRuntimeJoinFilterCode(ctx, joinFilterIDs, columnsLists, isFirstLocationLists, currentTable)
                    rtjfExpr?.let {
                        val tableVar = ctx.builder().symbolTable.genTableVar()
                        val assign = Op.Assign(tableVar, rtjfExpr)
                        ctx.builder().add(assign)
                        currentTable = BodoEngineTable(tableVar.emit(), this)
                    }
                    currentTable
                },
                reportOutTableSize = true,
            )
        val pipeline =
            OutputtingPipelineEmission(
                listOf(stage),
                false,
                input,
            )
        val operatorEmission =
            OperatorEmission(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                listOf(),
                pipeline,
                timeStateInitialization = false,
            )
        return implementor.buildStreaming(operatorEmission)!!
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable = UnusedStateVariable

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) = Unit

    companion object {
        fun create(
            input: RelNode,
            joinFilterIDs: List<Int>,
            equalityFilterColumns: List<List<Int>>,
            equalityIsFirstLocations: List<List<Boolean>>,
            nonEqualityFilterInfo: List<List<NonEqualityJoinFilterColumnInfo>>,
        ): BodoPhysicalRuntimeJoinFilter {
            val cluster = input.cluster
            val traitSet = cluster.traitSet()
            return BodoPhysicalRuntimeJoinFilter(
                cluster,
                traitSet,
                input,
                joinFilterIDs,
                equalityFilterColumns,
                equalityIsFirstLocations,
                nonEqualityFilterInfo,
            )
        }

        /**
         * Generate runtime join filter code as an expression in terms of the input expression.
         *
         * @param ctx The builder context
         * @param joinFilterIDs The ID of the join causing the filter.
         * @param equalityFilterColumns The list mapping columns of the join to the columns in the current table
         * @param equalityIsFirstLocations The list indicating for which of the columns is it the first filtering site
         * @param input The input expression
         * @return
         */
        @JvmStatic
        fun generateRuntimeJoinFilterCode(
            ctx: BodoPhysicalRel.BuildContext,
            joinFilterIDs: List<Int>,
            equalityFilterColumns: List<List<Int>>,
            equalityIsFirstLocations: List<List<Boolean>>,
            input: Expr,
        ): Expr? {
            val joinStateCache = ctx.builder().getJoinStateCache()
            // Output is basically a tuple of (stateVar, keyLocations) that we
            // care about.
            val joinStatesInfo = joinFilterIDs.map { joinStateCache.getStreamingJoinInfo(it) }

            val stateVars = mutableListOf<StateVariable>()
            val columnVars = mutableListOf<Variable>()
            val isFirstLocationVars = mutableListOf<Variable>()
            joinStatesInfo.forEachIndexed { stateIdx, (stateVar, keyLocations, _) ->
                stateVar.let {
                    val columnOrderedList = MutableList(equalityFilterColumns[stateIdx].size) { Expr.NegativeOne }
                    val isFirstList: MutableList<Expr.BooleanLiteral> =
                        MutableList(equalityIsFirstLocations[stateIdx].size) { Expr.BooleanLiteral(false) }
                    keyLocations.forEachIndexed { locIdx, keyLocation ->
                        columnOrderedList[keyLocation] = Expr.IntegerLiteral(equalityFilterColumns[stateIdx][locIdx])
                        isFirstList[keyLocation] = Expr.BooleanLiteral(equalityIsFirstLocations[stateIdx][locIdx])
                    }
                    val columnsTuple = Expr.Tuple(columnOrderedList)
                    val isFirstLocationTuple = Expr.Tuple(isFirstList)
                    columnVars.add(ctx.lowerAsMetaType(columnsTuple))
                    isFirstLocationVars.add(ctx.lowerAsMetaType(isFirstLocationTuple))
                    stateVars.add(stateVar)
                }
            }

            // If we don't have the state stored assume we have disabled
            // streaming entirely and this is a no-op.
            if (stateVars.isEmpty()) {
                return null
            }

            val stateVarsTuple = Expr.Tuple(stateVars)
            val columnVarsTuple = Expr.Tuple(columnVars)
            val isFirstLocationVarsTuple = Expr.Tuple(isFirstLocationVars)
            return Expr.Call(
                "bodo.libs.streaming.join.runtime_join_filter",
                listOf(stateVarsTuple, input, columnVarsTuple, isFirstLocationVarsTuple),
            )
        }
    }
}
