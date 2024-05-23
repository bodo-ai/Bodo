package com.bodosql.calcite.codeGeneration

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import java.util.concurrent.atomic.AtomicInteger

abstract class StageEmission(
    val bodyFn: (BodoPhysicalRel.BuildContext, StateVariable, BodoEngineTable?) -> BodoEngineTable?,
    val reportOutTableSize: Boolean,
) {
    fun emitStage(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
        table: BodoEngineTable?,
        stageNumber: AtomicInteger,
        timerInfo: StreamingRelNodeTimer,
    ): BodoEngineTable? {
        val stage = stageNumber.getAndIncrement()
        timerInfo.insertLoopOperationStartTimer(stage)
        val result = bodyFn(ctx, stateVar, table)
        timerInfo.insertLoopOperationEndTimer(stage)
        if (reportOutTableSize) {
            timerInfo.updateRowCount(stage, result!!)
        }
        return result
    }
}
