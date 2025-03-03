package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen
import com.bodosql.calcite.application.utils.AggHelpers
import com.bodosql.calcite.application.utils.Utils
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.codeGeneration.TerminatingPipelineEmission
import com.bodosql.calcite.codeGeneration.TerminatingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Op.Assign
import com.bodosql.calcite.ir.Op.Stmt
import com.bodosql.calcite.ir.Op.TupleAssign
import com.bodosql.calcite.ir.OperatorID
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.StreamingPipelineFrame
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.AggregateBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet
import org.apache.calcite.util.Pair
import kotlin.math.ceil

class BodoPhysicalAggregate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : AggregateBase(
        cluster,
        traitSet.replace(BodoPhysicalRel.CONVENTION),
        ImmutableList.of(),
        input,
        groupSet,
        groupSets,
        aggCalls,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): BodoPhysicalAggregate = BodoPhysicalAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls)

    /**
     * Determine if this aggregate should use the
     * grouping sets code paths.
     */
    private fun usesGroupingSets(): Boolean = groupSets.size > 1 || (groupSets[0] != groupSet)

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            emitStreaming(implementor)
        } else {
            emitSingleBatch(implementor)
        }

    private fun emitStreaming(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        val stage =
            TerminatingStageEmission { ctx, stateVar, table ->
                emitStreamingBuild(ctx, stateVar, table!!)
                null
            }
        val terminatingPipeline = TerminatingPipelineEmission(listOf(), stage, false, input)
        // The final pipeline generates the output from the state.
        val outputStage =
            OutputtingStageEmission(
                { ctx, stateVar, _ ->
                    emitStreamingProduceOutput(ctx, stateVar)
                },
                // Out table sizes are handled directly in C++.
                reportOutTableSize = false,
            )
        val outputPipeline = OutputtingPipelineEmission(listOf(outputStage), true, null)
        val operatorEmission =
            OperatorEmission(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                listOf(terminatingPipeline),
                outputPipeline,
                timeStateInitialization = true,
            )

        return implementor.buildStreaming(operatorEmission)!!
    }

    private fun emitStreamingBuild(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
        input: BodoEngineTable,
    ) {
        val builder = ctx.builder()
        val pipeline: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val batchExitCond: Variable = pipeline.getExitCond()
        val newExitCond: Variable = builder.symbolTable.genFinishedStreamingFlag()
        val inputRequest: Variable = builder.symbolTable.genInputRequestVar()
        val batchCall =
            if (usesGroupingSets()) {
                Expr.Call(
                    "bodo.libs.streaming.groupby.groupby_grouping_sets_build_consume_batch",
                    listOf(
                        stateVar,
                        input,
                        batchExitCond,
                    ),
                )
            } else {
                Expr.Call(
                    "bodo.libs.streaming.groupby.groupby_build_consume_batch",
                    listOf(
                        stateVar,
                        input,
                        batchExitCond,
                        // is_final_pipeline is always true for groupBy
                        Expr.BooleanLiteral(true),
                    ),
                )
            }
        builder.add(TupleAssign(listOf(newExitCond, inputRequest), batchCall))
        pipeline.addInputRequest(inputRequest)
        pipeline.endSection(newExitCond)
        // Only GroupBy build needs a memory budget since output only has a ChunkedTableBuilder
        builder.forceEndOperatorAtCurPipeline(ctx.operatorID(), pipeline)
        // Also end any nested operators.
        if (usesGroupingSets()) {
            val subOperatorIDs = generateSubOperatorIDs(ctx.operatorID())
            subOperatorIDs.forEach {
                builder.forceEndOperatorAtCurPipeline(it, pipeline)
            }
        }
    }

    private fun generateSubOperatorIDs(operatorID: OperatorID): List<OperatorID> =
        (1..groupSets.size).map {
            OperatorID(operatorID.id + it, operatorID.hide)
        }

    private fun emitStreamingProduceOutput(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ): BodoEngineTable {
        val builder = ctx.builder()
        val pipeline: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        // Add the output side
        val outTable: Variable = builder.symbolTable.genTableVar()
        val outputControl: Variable = builder.symbolTable.genOutputControlVar()
        pipeline.addOutputControl(outputControl)
        val functionName =
            if (usesGroupingSets()) {
                "bodo.libs.streaming.groupby.groupby_grouping_sets_produce_output_batch"
            } else {
                "bodo.libs.streaming.groupby.groupby_produce_output_batch"
            }
        val outputCall = Expr.Call(functionName, listOf(stateVar, outputControl))
        builder.add(TupleAssign(listOf(outTable, pipeline.getExitCond()), outputCall))
        val intermediateTable = BodoEngineTable(outTable.emit(), this)
        val filteredAggCallList = Utils.literalAggPrunedAggList(aggCallList)
        return if (filteredAggCallList.size !== aggCallList.size) {
            // Append any Literal data if it exists.
            Utils.concatenateLiteralAggValue(builder, ctx, intermediateTable, this)
        } else {
            intermediateTable
        }
    }

    private fun emitSingleBatch(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        (implementor::build)(inputs) { ctx, inputs ->
            val groupingVariables = groupSet.asList()
            val groups = groupSets
            val expectedOutputCols: List<String> = this.getRowType().fieldNames
            val outVar: Variable = ctx.builder().symbolTable.genDfVar()
            // Remove any LITERAL_AGG and GROUPING nodes for the non-streaming code path.
            val filteredAggregateCallList = Utils.literalAggPrunedAggList(aggCallList)
            val groupingPrunedAggregateCallList = Utils.groupingPrunedAggList(filteredAggregateCallList)

            // Expected output column names according to the calcite plan, contains any/all
            // of the expected aliases
            val aggCallNames: MutableList<String> = ArrayList()
            for (i in groupingPrunedAggregateCallList.indices) {
                val aggregateCall = groupingPrunedAggregateCallList[i]
                if (aggregateCall.getName() == null) {
                    aggCallNames.add(expectedOutputCols[groupingVariables.size + i])
                } else {
                    aggCallNames.add(aggregateCall.getName()!!)
                }
            }
            var finalOutVar = outVar
            val inputColumnNames: List<String> = input.rowType.fieldNames
            val inTable: BodoEngineTable = inputs[0]
            val inVar: Variable = ctx.convertTableToDf(inTable)
            val outputDfNames: MutableList<String> = java.util.ArrayList()
            // If any group is missing a column we may need to do a concat.
            var hasMissingColsGroup = false

            // Naive implementation for handling multiple aggregation groups, where we
            // repeatedly
            // call group by, and append the dataframes together
            for (curGroup in groups) {
                hasMissingColsGroup = hasMissingColsGroup || curGroup.cardinality() < groupingVariables.size
                var curGroupAggExpr: Expr?
                // First rename any input keys to the output
                // group without aggregation : e.g. select B from table1 groupby A
                if (groupingPrunedAggregateCallList.isEmpty()) {
                    curGroupAggExpr = AggCodeGen.generateAggCodeNoAgg(inVar, inputColumnNames, curGroup.toList())
                } else if (curGroup.isEmpty) {
                    curGroupAggExpr =
                        AggCodeGen.generateAggCodeNoGroupBy(
                            inVar,
                            inputColumnNames,
                            groupingPrunedAggregateCallList,
                            aggCallNames,
                            usesGroupingSets(),
                            ctx,
                        )
                } else {
                    val (key, prependOp) =
                        handleLogicalAggregateWithGroups(
                            inVar,
                            inputColumnNames,
                            groupingPrunedAggregateCallList,
                            aggCallNames,
                            curGroup.toList(),
                            ctx,
                        )
                    curGroupAggExpr = key
                    if (prependOp != null) {
                        ctx.builder().add(prependOp)
                    }
                }
                // assign each of the generated dataframes their own variable, for greater
                // clarity in
                // the
                // generated code
                val outDf: Variable = ctx.builder().symbolTable.genDfVar()
                ctx.builder().add(Assign(outDf, curGroupAggExpr!!))
                if (groupingPrunedAggregateCallList.size != filteredAggregateCallList.size) {
                    // We have at least 1 grouping call, so we need to concatenates the literal value
                    // for this group.
                    val tableColumns = curGroup.cardinality() + groupingPrunedAggregateCallList.size
                    val tableRepresentation: BodoEngineTable = ctx.convertDfToTable(outDf, tableColumns)
                    val tableVar =
                        Utils.appendLiteralGroupingValues(
                            ctx.builder(),
                            ctx,
                            tableRepresentation,
                            curGroup,
                            filteredAggregateCallList,
                            tableColumns,
                        )
                    val groupSetList = groupSet.toList()
                    val keptFields =
                        this
                            .getRowType()
                            .fieldList
                            .withIndex()
                            .filter {
                                (it.index < groupSetList.size && curGroup.contains(groupSetList[it.index])) ||
                                    (
                                        it.index >= groupingVariables.size &&
                                            aggCallList[it.index - groupingVariables.size].aggregation.kind != SqlKind.LITERAL_AGG
                                    )
                            }.map { it.value.type }
                    val keptNames =
                        this
                            .getRowType()
                            .fieldNames
                            .withIndex()
                            .filter {
                                (it.index < groupSetList.size && curGroup.contains(groupSetList[it.index])) ||
                                    (
                                        it.index >= groupingVariables.size &&
                                            aggCallList[it.index - groupingVariables.size].aggregation.kind != SqlKind.LITERAL_AGG
                                    )
                            }.map { it.value }
                    val outputRowType = cluster.typeFactory.createStructType(keptFields, keptNames)
                    val engineTable = BodoEngineTable(tableVar.emit(), outputRowType)
                    val finalDf = ctx.convertTableToDf(engineTable)
                    outputDfNames.add(finalDf.name)
                } else {
                    outputDfNames.add(outDf.name)
                }
            }
            // If we have multiple groups, append the dataframes together
            if (usesGroupingSets() || hasMissingColsGroup) {
                // It is not guaranteed that a particular input column exists in any of the
                // output
                // dataframes,
                // but Calcite expects
                // All input dataframes to be carried into the output. It is also not
                // guaranteed that the output dataframes contain the columns in the order
                // expected by
                // calcite.
                // In order to ensure that we have all the input columns in the output,
                // we create a dummy dataframe that has all the columns with
                // a length of 0. The ordering is handled by a loc after the concat

                // We initialize the dummy column like this, as Bodo will default these columns
                // to
                // string type if we initialize empty columns.
                val concatDfs: MutableList<String> = java.util.ArrayList()
                if (hasMissingColsGroup) {
                    val dummyDfVar: Variable = ctx.builder().symbolTable.genDfVar()
                    // TODO: Switch to proper IR
                    val dummyDfExpr: Expr = Expr.Raw(inVar.name + ".iloc[:0, :]")
                    // Assign the dummy df to a variable name,
                    ctx.builder().add(Assign(dummyDfVar, dummyDfExpr))
                    concatDfs.add(dummyDfVar.emit())
                }
                concatDfs.addAll(outputDfNames)

                // Generate the concatenation expression
                val concatExprRaw = StringBuilder(AggCodeGen.concatDataFrames(concatDfs).emit())

                // Sort the output dataframe, so that they are in the ordering expected by
                // Calcite
                // Needed in the case that the topmost dataframe in the concat does not contain
                // all
                // the
                // columns in the correct ordering
                concatExprRaw.append(".loc[:, [")
                for (i in expectedOutputCols.indices) {
                    concatExprRaw.append(Utils.makeQuoted(expectedOutputCols[i])).append(", ")
                }
                concatExprRaw.append("]]")

                // Generate the concatenation
                ctx.builder().add(
                    Assign(finalOutVar, Expr.Raw(concatExprRaw.toString())),
                )
            } else {
                finalOutVar = Variable(outputDfNames[0])
            }
            // Generate a table using just the node members that aren't LITERAL_AGG
            val numCols: Int = (
                this.getRowType().fieldCount -
                    (aggCallList.size - filteredAggregateCallList.size)
            )
            val intermediateTable: BodoEngineTable = ctx.convertDfToTable(finalOutVar, numCols)
            if (this.getRowType().fieldCount != numCols) {
                // Insert the LITERAL_AGG results
                Utils.concatenateLiteralAggValue(ctx.builder(), ctx, intermediateTable, this)
            } else {
                intermediateTable
            }
        }

    /**
     * Generates an expression for a Logical Aggregation with grouped variables. May return code to be
     * appended to the generated code (this is needed if the aggregation list contains a filter).
     *
     * @param inVar The input variable.
     * @param inputColumnNames The names of the columns of the input var.
     * @param aggCallList The list of aggregations to be performed.
     * @param aggCallNames The list of column names to be used for the output of the aggregations
     * @param group List of integer column indices by which to group
     * @param ctx The build context for lowering globals.
     * @return A pair of strings, the key is the expression that evaluates to the output of the
     *     aggregation, and the value is the code that needs to be appended to the generated code.
     */
    private fun handleLogicalAggregateWithGroups(
        inVar: Variable,
        inputColumnNames: List<String>,
        aggCallList: List<AggregateCall>,
        aggCallNames: List<String>,
        group: List<Int>,
        ctx: BodoPhysicalRel.BuildContext,
    ): Pair<Expr, Op?> =
        if (AggHelpers.aggContainsFilter(aggCallList)) {
            // If we have a Filter we need to generate a groupby apply
            AggCodeGen.generateApplyCodeWithGroupBy(
                inVar,
                inputColumnNames,
                group,
                aggCallList,
                aggCallNames,
                ctx.builder().symbolTable.genGroupbyApplyAggFnName(),
                ctx,
            )
        } else {
            // Otherwise generate groupby.agg
            val output =
                AggCodeGen.generateAggCodeWithGroupBy(inVar, inputColumnNames, group, aggCallList, aggCallNames)
            Pair(output, null)
        }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val stateVar = builder.symbolTable.genStateVar()
        val keyIndicesList = AggCodeGen.getStreamingGroupByKeyIndices(groupSet)
        val keyIndices: Variable = ctx.lowerAsMetaType(Expr.Tuple(keyIndicesList))
        val filteredAggCallList = Utils.literalAggPrunedAggList(aggCallList)
        val offsetAndCols = AggCodeGen.getStreamingGroupByOffsetAndCols(filteredAggCallList, ctx, keyIndicesList[0])
        val offset = offsetAndCols.left
        val cols = offsetAndCols.right
        val funcNames = AggCodeGen.getStreamingGroupbyFnames(filteredAggCallList, ctx)
        if (usesGroupingSets()) {
            // Generate the grouping state indices.
            val groupingSets =
                groupSets.map {
                    Expr.Tuple(AggCodeGen.getStreamingGroupByKeyIndices(it))
                }
            val groupingSetsVar = ctx.lowerAsMetaType(Expr.Tuple(groupingSets))
            val operatorId = ctx.operatorID()
            val subOperatorIDs = generateSubOperatorIDs(operatorId)
            val subOperatorIDCodegen = ctx.lowerAsMetaType(Expr.Tuple(subOperatorIDs.map { it.toExpr() }))
            // TODO: Add handling for if we are using GROUPING.
            val stateCall =
                Expr.Call(
                    "bodo.libs.streaming.groupby.init_grouping_sets_state",
                    operatorId.toExpr(),
                    subOperatorIDCodegen,
                    keyIndices,
                    groupingSetsVar,
                    funcNames,
                    offset,
                    cols,
                )
            val groupByInit = Assign(stateVar, stateCall)
            // Fetch the streaming pipeline
            val pipeline: StreamingPipelineFrame = ctx.builder().getCurrentStreamingPipeline()
            val mq: RelMetadataQuery = cluster.metadataQuery
            pipeline.initializeStreamingState(
                operatorId,
                groupByInit,
                OperatorType.GROUPING_SETS,
                0,
            )
            subOperatorIDs.withIndex().forEach {
                pipeline.startNestedOperator(it.value, OperatorType.GROUPBY, estimateGroupingSetMemory(mq, groupSets[it.index]))
            }
        } else {
            val stateCall =
                Expr.Call(
                    "bodo.libs.streaming.groupby.init_groupby_state",
                    ctx.operatorID().toExpr(),
                    keyIndices,
                    funcNames,
                    offset,
                    cols,
                )
            val groupByInit = Assign(stateVar, stateCall)
            // Fetch the streaming pipeline
            val pipeline: StreamingPipelineFrame = ctx.builder().getCurrentStreamingPipeline()
            val mq: RelMetadataQuery = cluster.metadataQuery
            pipeline.initializeStreamingState(
                ctx.operatorID(),
                groupByInit,
                OperatorType.GROUPBY,
                estimateBuildMemory(mq),
            )
        }
        return stateVar
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val finalizePipeline = ctx.builder().getCurrentStreamingPipeline()
        val functionName =
            if (usesGroupingSets()) {
                "bodo.libs.streaming.groupby.delete_grouping_sets_state"
            } else {
                "bodo.libs.streaming.groupby.delete_groupby_state"
            }
        val deleteState = Stmt(Expr.Call(functionName, listOf(stateVar)))
        finalizePipeline.addTermination(deleteState)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty =
        ExpectedBatchingProperty.aggregateProperty(groupSet, groupSets, aggCalls, getRowType())

    /**
     * Estimate the memory needed for each grouping set. If a key cannot be found then we select an equal fraction
     * of the total row count.
     */
    private fun estimateGroupingSetMemory(
        mq: RelMetadataQuery,
        groupKeys: ImmutableBitSet,
    ): Int {
        val distinctRows = mq.getDistinctRowCount(this, groupKeys, null) ?: mq.getRowCount(this) / groupSets.size
        val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
        return ceil(distinctRows * averageBuildRowSize).toInt()
    }

    /**
     * Get group by build memory estimate for memory budget comptroller
     */
    private fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // See if streaming group by will use accumulate or aggregate code path
        var isStreamAccumulate = false
        for (aggCall in aggCalls) {
            // Should match accumulate function check in C++:
            // https://github.com/bodo-ai/Bodo/blob/3c902f01b0aa0748793b00554304d8a051f511aa/bodo/libs/_stream_groupby.cpp#L1101
            if (!ExpectedBatchingProperty.streamingSupportedWithoutAccumulateAggFunction(aggCall)) {
                isStreamAccumulate = true
                break
            }
        }

        // Get the set of group by key indices to skip in type check below
        val keySet = mutableSetOf<Int>()
        for (i in 0 until groupSet.size()) {
            if (groupSet[i]) {
                keySet.add(i)
            }
        }
        // Accumulate code path needs all input in memory
        return if (isStreamAccumulate) {
            val buildRows = mq.getRowCount(input)
            val averageBuildRowSize = mq.getAverageRowSize(input) ?: 8.0
            // multiply by 3 to account for extra memory needed in update call at the end
            ceil(3 * buildRows * averageBuildRowSize).toInt()
        } else {
            // Use output row count for aggregate code path
            val distinctRows = mq.getRowCount(this)
            val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
            ceil(distinctRows * averageBuildRowSize).toInt()
        }
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            groupSet: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>,
            aggCalls: List<AggregateCall>,
        ): BodoPhysicalAggregate = BodoPhysicalAggregate(cluster, cluster.traitSet(), input, groupSet, groupSets, aggCalls)
    }
}
