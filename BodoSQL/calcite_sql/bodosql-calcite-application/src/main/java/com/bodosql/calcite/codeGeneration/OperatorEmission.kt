package com.bodosql.calcite.codeGeneration

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import java.util.concurrent.atomic.AtomicInteger

class OperatorEmission(
    val stateInitFn: (BodoPhysicalRel.BuildContext) -> StateVariable,
    val deleteStateFn: (BodoPhysicalRel.BuildContext, StateVariable) -> Unit,
    private val terminatingPipelines: List<TerminatingPipelineEmission>,
    private val outputPipeline: OutputtingPipelineEmission?,
    val timeStateInitialization: Boolean,
) {
    private val firstPipeline = if (terminatingPipelines.isEmpty()) outputPipeline!! else terminatingPipelines.first()
    private val lastPipeline = outputPipeline ?: terminatingPipelines.last()
    private val stageNumber = AtomicInteger(0)

    /**
     * Initialize the first pipeline, so it's safe to generate timers.
     */
    fun initFirstPipeline(ctx: BodoPhysicalRel.BuildContext): BodoEngineTable? {
        return firstPipeline.initializePipeline(ctx, 0)
    }

    fun emitOperator(
        ctx: BodoPhysicalRel.BuildContext,
        timerInfo: StreamingRelNodeTimer,
        input: BodoEngineTable?,
    ): BodoEngineTable? {
        val builder = ctx.builder()
        val initializationStage = stageNumber.getAndIncrement()
        if (timeStateInitialization) {
            timerInfo.insertStateStartTimer(initializationStage)
        }
        val stateVar: StateVariable = stateInitFn(ctx)
        if (timeStateInitialization) {
            timerInfo.insertStateEndTimer(initializationStage)
        }
        for (pipelineInfo in terminatingPipelines.withIndex()) {
            val (idx, pipeline) = pipelineInfo
            val inputTable =
                if (pipeline == firstPipeline) {
                    input
                } else {
                    pipeline.initializePipeline(ctx, idx)
                }
            pipeline.emitPipeline(ctx, stateVar, inputTable, stageNumber, timerInfo)
            // Don't terminate the last pipeline, so we can delete state at the end.
            if (pipeline != lastPipeline) {
                val streamingFrame = builder.endCurrentStreamingPipeline()
                builder.add(Op.StreamingPipeline(streamingFrame))
            }
        }
        val outputTable =
            if (outputPipeline != null) {
                val inputTable =
                    if (outputPipeline != firstPipeline) {
                        outputPipeline.initializePipeline(ctx, terminatingPipelines.size)
                    } else {
                        input
                    }
                outputPipeline.emitPipeline(ctx, stateVar, inputTable, stageNumber, timerInfo)
            } else {
                null
            }
        deleteStateFn(ctx, stateVar)
        // If we are sink we must terminate the final pipeline.
        if (lastPipeline is TerminatingPipelineEmission) {
            val streamingFrame = builder.endCurrentStreamingPipeline()
            builder.add(Op.StreamingPipeline(streamingFrame))
        }
        return outputTable
    }
}
