package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen
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
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.StreamingPipelineFrame
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.UnionBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import kotlin.math.ceil

class BodoPhysicalUnion(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : UnionBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), inputs, all), BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
        all: Boolean,
    ): BodoPhysicalUnion {
        return BodoPhysicalUnion(cluster, traitSet, inputs, all)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        return if (isStreaming()) {
            emitStreaming(implementor)
        } else {
            emitSingleBatch(implementor)
        }
    }

    private fun emitStreaming(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        // Generate a pipeline for each input and pushing its result into the union state
        val terminatingPipelines =
            inputs.withIndex().map {
                    (idx, input) ->
                val isLast = idx == inputs.size - 1
                val forceEndOperator = isLast && !all
                // Terminating pipelines must end with a terminating state. Because we just
                // need to push the result of the input into the union state, we only need
                // a single state.
                val stage =
                    TerminatingStageEmission { ctx, stateVar, table ->
                        val inputVar = table!!
                        val builder = ctx.builder()
                        val pipeline: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
                        val batchExitCond = pipeline.getExitCond()
                        val consumeCall =
                            Expr.Call(
                                "bodo.libs.stream_union.union_consume_batch",
                                listOf(
                                    stateVar,
                                    inputVar,
                                    batchExitCond,
                                    Expr.BooleanLiteral(isLast),
                                ),
                            )
                        val newExitCond: Variable = builder.symbolTable.genFinishedStreamingFlag()
                        val inputRequest: Variable = builder.symbolTable.genInputRequestVar()
                        val exitAssign = Op.TupleAssign(listOf(newExitCond, inputRequest), consumeCall)
                        builder.add(exitAssign)
                        pipeline.endSection(newExitCond)
                        pipeline.addInputRequest(inputRequest)
                        // We need to reset non-blocking is_last sync state after each pipeline when using groupby
                        if (!all) {
                            val endBuild =
                                Op.Stmt(Expr.Call("bodo.libs.stream_union.end_union_consume_pipeline", listOf(stateVar)))
                            pipeline.addTermination(endBuild)
                        }
                        // For budget purposes we mark the end of the operator after the last stage.
                        // We only need to do this
                        if (forceEndOperator) {
                            builder.forceEndOperatorAtCurPipeline(ctx.operatorID(), pipeline)
                        }
                        null
                    }
                TerminatingPipelineEmission(listOf(), stage, false, input)
            }
        // The final pipeline generates the output from the state.
        val outputStage =
            OutputtingStageEmission(
                {
                        ctx, stateVar, _ ->
                    val builder = ctx.builder()
                    val pipeline = builder.getCurrentStreamingPipeline()
                    val outputControl: Variable = builder.symbolTable.genOutputControlVar()
                    pipeline.addOutputControl(outputControl)
                    val outputCall =
                        Expr.Call(
                            "bodo.libs.stream_union.union_produce_batch",
                            listOf(stateVar, outputControl),
                        )
                    val outTable: Variable = builder.symbolTable.genTableVar()
                    val finishedFlag = pipeline.getExitCond()
                    val outputAssign = Op.TupleAssign(listOf(outTable, finishedFlag), outputCall)
                    builder.add(outputAssign)
                    ctx.returns(outTable)
                },
                reportOutTableSize = true,
            )
        val outputPipeline = OutputtingPipelineEmission(listOf(outputStage), true, null)
        val operatorEmission =
            OperatorEmission(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                terminatingPipelines,
                outputPipeline,
                timeStateInitialization = false,
            )
        return implementor.buildStreaming(operatorEmission)!!
    }

    private fun emitSingleBatch(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        val columnNames = getRowType().fieldNames
        return (implementor::build)(inputs) {
                ctx, inputs ->
            val builder = ctx.builder()
            val dfs = inputs.map { input -> ctx.convertTableToDf(input) }
            val outVar = builder.symbolTable.genDfVar()
            val unionExpr = SetOpCodeGen.generateUnionCode(columnNames, dfs, all, ctx)
            builder.add(Assign(outVar, unionExpr))
            ctx.convertDfToTable(outVar, this)
        }
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val stateVar = builder.symbolTable.genStateVar()
        val isAll = Pair<String, Expr>("all", Expr.BooleanLiteral(this.all))
        val stateCall =
            Expr.Call(
                "bodo.libs.stream_union.init_union_state",
                listOf(ctx.operatorID().toExpr()),
                listOf(isAll),
            )
        val unionInit = Assign(stateVar, stateCall)
        val initialPipeline = builder.getCurrentStreamingPipeline()
        if (this.all) {
            // UNION ALL is implemented as a ChunkedTableBuilder and doesn't need tracked in the memory
            // budget comptroller
            initialPipeline.addInitialization(unionInit)
        } else {
            val mq: RelMetadataQuery = cluster.metadataQuery
            initialPipeline.initializeStreamingState(
                ctx.operatorID(),
                unionInit,
                OperatorType.UNION,
                estimateBuildMemory(mq),
            )
        }
        return stateVar
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val builder = ctx.builder()
        val finalPipeline = builder.getCurrentStreamingPipeline()
        val deleteState =
            Stmt(Expr.Call("bodo.libs.stream_union.delete_union_state", listOf(stateVar)))
        finalPipeline.addTermination(deleteState)
    }

    /**
     * Get union build memory estimate for memory budget comptroller
     */
    private fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // UNION ALL is implemented as a ChunkedTableBuilder and doesn't need tracked in the memory
        // budget comptroller
        assert(!all)

        // Union distinct is same as aggregation
        val distinctRows = mq.getRowCount(this)
        val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
        return ceil(distinctRows * averageBuildRowSize).toInt()
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            all: Boolean,
        ): BodoPhysicalUnion {
            return BodoPhysicalUnion(cluster, cluster.traitSet(), inputs, all)
        }
    }
}
