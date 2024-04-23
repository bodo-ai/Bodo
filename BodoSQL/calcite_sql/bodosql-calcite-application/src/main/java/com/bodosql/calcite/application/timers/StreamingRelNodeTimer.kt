package com.bodosql.calcite.application.timers

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Op.Assign
import com.bodosql.calcite.ir.Op.Stmt
import com.bodosql.calcite.ir.StreamingPipelineFrame
import com.bodosql.calcite.ir.Variable

private val noOpVar = Variable("NOOP")

private const val IO_TIMING_VERBOSE_LEVEL = 1
private const val REL_NODE_TIMING_VERBOSE_LEVEL = 2

/**
 * Class that builds the framework for implementing runtime timers around the various components in a streaming
 * operator. This class tracks state between operations but the code generation is still responsible for calling
 * the functions at the appropriate times. This class duplicates some code in SingleBatchRelNodeTimer
 * because that class is scheduled to be removed when everything is ported to streaming.
 */
class StreamingRelNodeTimer(
    private val opID: Int,
    private val builder: Module.Builder,
    private val isVerbose: Boolean,
    private val tracingLevel: Int,
    private val operationDescriptor: String,
    private val loggingTitle: String,
    private val nodeDetails: String,
) {
    // TODO(aneesh) Rename this class to reflect that it isn't just a timer, but an interface to the profiler in general

    // Only generate timers if BODO_TRACING_LEVEL >= 1
    private val isNoOp = tracingLevel == 0

    /**
     * Insert the initial time.time() before the state for a streaming operator.
     * This should be used by operators where the state is potentially non-trivial
     * (e.g. Snowflake Read). This must be called before the state code is generated.
     */
    fun insertStateStartTimer(stage: Int) {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val timeCall = Expr.Call("time.time")
        val stmt = Assign(builder.symbolTable.genOperatorStageTimerStartVar(opID, stage), timeCall)
        frame.addInitialization(stmt)
    }

    /**
     * Insert the time.time() after the state for a streaming operator and compute
     * the difference in time. This should be used by operators where the state is
     * potentially non-trivial (e.g. Snowflake Read). This requires insertStateStartTimer()
     * to have previously been called and must be called after the state code is generated.
     */
    fun insertStateEndTimer(stage: Int) {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val stateTimerEndVar = builder.symbolTable.genOperatorStageTimerEndVar(opID, stage)
        val timeCall = Expr.Call("time.time")
        val endTimer = Assign(stateTimerEndVar, timeCall)
        frame.addInitialization(endTimer)
        // Compute the difference - no need to use the elapsed variable, just write directly to the timer variable since this isn't in a loop
        val totalTime = builder.symbolTable.genOperatorStageTimerVar(opID, stage)
        val subCall = Expr.Binary("-", stateTimerEndVar, builder.symbolTable.genOperatorStageTimerStartVar(opID, stage))
        val subAssign = Assign(totalTime, subCall)
        frame.addInitialization(subAssign)
        frame.addInitialization(
            Stmt(
                Expr.Call(
                    "bodo.libs.query_profile_collector.submit_operator_stage_time",
                    Expr.IntegerLiteral(opID),
                    Expr.IntegerLiteral(stage),
                    totalTime,
                ),
            ),
        )
    }

    /**
     * Insert the initial time.time() before the code in the body of a streaming operator.
     * This must be called before the code is generated that occurs on each batch
     * of a streaming operator.
     */
    fun insertLoopOperationStartTimer(stage: Int) {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()

        // set up accumulator for stage
        val accumulator = builder.symbolTable.genOperatorStageTimerVar(opID, stage)
        frame.addInitialization(Assign(accumulator, Expr.Zero))

        val timeCall = Expr.Call("time.time")
        val stmt = Assign(builder.symbolTable.genOperatorStageTimerStartVar(opID, stage), timeCall)
        frame.add(stmt)
    }

    fun updateRowCount(
        stage: Int,
        outTable: Variable,
    ) {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        val frame = builder.getCurrentStreamingPipeline()

        // Before the loop, set up an accumulator to store the total output rows produced by this stage
        val accumulator = builder.symbolTable.genOperatorStageRowCountVar(opID, stage)
        frame.addInitialization(Assign(accumulator, Expr.Zero))

        // In the loop, update the accumulator with the local_len of the output produced in this iteration
        val outputLength = Expr.Call("bodo.hiframes.table.local_len", outTable)
        val newLength = Expr.Binary("+", accumulator, outputLength)
        frame.add(Assign(accumulator, newLength))

        // After the loop, submit the row count to the profiler
        frame.addTermination(
            Stmt(
                Expr.Call(
                    "bodo.libs.query_profile_collector.submit_operator_stage_row_counts",
                    Expr.IntegerLiteral(opID),
                    Expr.IntegerLiteral(stage),
                    accumulator,
                ),
            ),
        )
    }

    /**
     * Insert the time.time() after the code in the body of a streaming operator and compute
     * the difference in time. This must be called after the code is generated that occurs on
     * each batch of a streaming operator and requires insertLoopOperationStartTimer() to
     * have previously been called.
     */
    fun insertLoopOperationEndTimer(stage: Int) {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val loopEndTimerVar = builder.symbolTable.genOperatorStageTimerEndVar(opID, stage)
        val timeCall = Expr.Call("time.time")
        val endTimer = Assign(loopEndTimerVar, timeCall)
        frame.add(endTimer)
        // Compute the difference
        val subVar = builder.symbolTable.genOperatorStageTimerElapsedVar(opID, stage)
        val subCall = Expr.Binary("-", loopEndTimerVar, builder.symbolTable.genOperatorStageTimerStartVar(opID, stage))
        val subAssign = Assign(subVar, subCall)
        frame.add(subAssign)
        // Update the accumulator
        val accumulator = builder.symbolTable.genOperatorStageTimerVar(opID, stage)
        val addCall = Expr.Binary("+", accumulator, subVar)
        val addAssign = Assign(accumulator, addCall)
        frame.add(addAssign)

        val updateProfiler =
            Expr.Call(
                "bodo.libs.query_profile_collector.submit_operator_stage_time",
                Expr.IntegerLiteral(opID),
                Expr.IntegerLiteral(stage),
                accumulator,
            )
        frame.addTermination(Stmt(updateProfiler))
    }

    /**
     * Insert the final print state to display the total time taken
     * by a streaming operator. This should be called after all code
     * for the operator has already been generated and requires initializeTimer()
     * have already been called.
     *
     * This function does not do any computation and merely prints the final result
     * of the timing from the other calls using a description of the given node.
     */
    fun terminateTimer() {
        if (isNoOp || !builder.isStreamingFrame()) {
            return
        }
        if (isVerbose) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()

        // Generate a variable with the node details to print
        val nodeDetailsVariable = builder.symbolTable.genGenericTempVar()
        frame.addTermination(Assign(nodeDetailsVariable, Expr.StringLiteral(nodeDetails)))

        val totalDurationCall = Expr.Call("bodo.libs.query_profile_collector.get_operator_duration", Expr.IntegerLiteral(opID))
        val printMessage =
            String.format(
                "f'''Execution time for %s {%s}: {%s}'''",
                operationDescriptor,
                nodeDetailsVariable.emit(),
                totalDurationCall.emit(),
            )
        val logMessageCall: Op =
            Stmt(
                Expr.Call(
                    "bodo.user_logging.log_message",
                    // TODO: Add a format string op?
                    Expr.StringLiteral(loggingTitle),
                    Expr.Raw(printMessage),
                ),
            )
        frame.addTermination(logMessageCall)
    }

    companion object {
        @JvmStatic
        fun createStreamingTimer(
            opID: Int,
            builder: Module.Builder,
            verboseLevel: Int,
            tracingLevel: Int,
            operationDescriptor: String,
            loggingTitle: String,
            nodeDetails: String,
            type: SingleBatchRelNodeTimer.OperationType,
        ): StreamingRelNodeTimer {
            val verboseThreshold =
                if (type == SingleBatchRelNodeTimer.OperationType.BATCH) {
                    REL_NODE_TIMING_VERBOSE_LEVEL
                } else {
                    IO_TIMING_VERBOSE_LEVEL
                }
            return StreamingRelNodeTimer(
                opID,
                builder,
                verboseLevel < verboseThreshold,
                tracingLevel,
                operationDescriptor,
                loggingTitle,
                nodeDetails,
            )
        }
    }
}
