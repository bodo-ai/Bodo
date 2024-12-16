package com.bodosql.calcite.codeGeneration

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import org.apache.calcite.rel.RelNode
import java.util.concurrent.atomic.AtomicInteger

abstract class PipelineEmission(
    private val outputStages: List<OutputtingStageEmission>,
    private val terminatingStage: TerminatingStageEmission?,
    private val startPipeline: Boolean,
    private val child: RelNode?,
) {
    init {
        if (startPipeline && child != null) {
            throw IllegalArgumentException("Cannot start pipeline with child")
        } else if (!startPipeline && child == null) {
            throw IllegalArgumentException("A node without a child must start a pipeline")
        }
    }

    fun initializePipeline(
        ctx: BodoPhysicalRel.BuildContext,
        pipelineIdx: Int,
    ): BodoEngineTable? =
        if (startPipeline) {
            ctx.startPipeline()
            null
        } else {
            ctx.visitChild(child!!, pipelineIdx)
        }

    fun terminatePipeline(ctx: BodoPhysicalRel.BuildContext) {
        if (terminatingStage != null) {
            ctx.endPipeline()
        }
    }

    fun emitPipeline(
        ctx: BodoPhysicalRel.BuildContext,
        stateVariable: StateVariable,
        input: BodoEngineTable?,
        stageNumber: AtomicInteger,
        timerInfo: StreamingRelNodeTimer,
    ): BodoEngineTable? {
        val emittedOutput = emitOutputStages(ctx, stateVariable, input, stageNumber, timerInfo)
        return if (terminatingStage == null) {
            emittedOutput
        } else {
            terminatingStage.emitStage(ctx, stateVariable, emittedOutput, stageNumber, timerInfo)
            return null
        }
    }

    private fun emitOutputStages(
        ctx: BodoPhysicalRel.BuildContext,
        stateVariable: StateVariable,
        input: BodoEngineTable?,
        stageNumber: AtomicInteger,
        timerInfo: StreamingRelNodeTimer,
    ): BodoEngineTable? {
        var result = input
        for (stage in outputStages) {
            result = stage.emitStage(ctx, stateVariable, result, stageNumber, timerInfo)
        }
        return result
    }
}
