package com.bodosql.calcite.application.timers

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Expr.DoubleLiteral
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
    private val builder: Module.Builder,
    private val isNoOp: Boolean,
    private val operationDescriptor: String,
    private val loggingTitle: String,
    private val nodeDetails: String,
) {
    // Avoid impacting tests when timers are disabled.
    private var accumulatorVar =
        if (isNoOp) {
            noOpVar
        } else {
            builder.symbolTable.genGenericTempVar()
        }
    private var stateStartTimerVar =
        if (isNoOp) {
            noOpVar
        } else {
            builder.symbolTable.genGenericTempVar()
        }
    private var loopStartTimerVar =
        if (isNoOp) {
            noOpVar
        } else {
            builder.symbolTable.genGenericTempVar()
        }

    /**
     * Initialize the accumulator state for a streaming operation. This must be
     * called before any other operation.
     */
    fun initializeTimer() {
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        frame.addInitialization(Assign(accumulatorVar, DoubleLiteral(0.0)))
    }

    /**
     * Insert the initial time.time() before the state for a streaming operator.
     * This should be used by operators where the state is potentially non-trivial
     * (e.g. Snowflake Read). This must be called before the state code is generated.
     */
    fun insertStateStartTimer() {
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val timeCall = Expr.Call("time.time")
        val stmt = Assign(stateStartTimerVar, timeCall)
        frame.addInitialization(stmt)
    }

    /**
     * Insert the time.time() after the state for a streaming operator and compute
     * the difference in time. This should be used by operators where the state is
     * potentially non-trivial (e.g. Snowflake Read). This requires insertStateStartTimer()
     * to have previously been called and must be called after the state code is generated.
     */
    fun insertStateEndTimer() {
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val stateTimerEndVar = builder.symbolTable.genGenericTempVar()
        val timeCall = Expr.Call("time.time")
        val endTimer = Assign(stateTimerEndVar, timeCall)
        frame.addInitialization(endTimer)
        // Compute the difference
        val subVar = builder.symbolTable.genGenericTempVar()
        val subCall = Expr.Binary("-", stateTimerEndVar, stateStartTimerVar)
        val subAssign = Assign(subVar, subCall)
        frame.addInitialization(subAssign)
        // Update the accumulator
        val addCall = Expr.Binary("+", accumulatorVar, subVar)
        val addAssign = Assign(accumulatorVar, addCall)
        frame.addInitialization(addAssign)
    }

    /**
     * Insert the initial time.time() before the code in the body of a streaming operator.
     * This must be called before the code is generated that occurs on each batch
     * of a streaming operator.
     */
    fun insertLoopOperationStartTimer() {
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val timeCall = Expr.Call("time.time")
        val stmt = Assign(loopStartTimerVar, timeCall)
        frame.add(stmt)
    }

    /**
     * Insert the time.time() after the code in the body of a streaming operator and compute
     * the difference in time. This must be called after the code is generated that occurs on
     * each batch of a streaming operator and requires insertLoopOperationStartTimer() to
     * have previously been called.
     */
    fun insertLoopOperationEndTimer() {
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
        val loopEndTimerVar = builder.symbolTable.genGenericTempVar()
        val timeCall = Expr.Call("time.time")
        val endTimer = Assign(loopEndTimerVar, timeCall)
        frame.add(endTimer)
        // Compute the difference
        val subVar = builder.symbolTable.genGenericTempVar()
        val subCall = Expr.Binary("-", loopEndTimerVar, loopStartTimerVar)
        val subAssign = Assign(subVar, subCall)
        frame.add(subAssign)
        // Update the accumulator
        val addCall = Expr.Binary("+", accumulatorVar, subVar)
        val addAssign = Assign(accumulatorVar, addCall)
        frame.add(addAssign)
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
        if (isNoOp) {
            return
        }
        val frame: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()

        // Generate a variable with the node details to print
        val nodeDetailsVariable = builder.symbolTable.genGenericTempVar()
        frame.addTermination(Assign(nodeDetailsVariable, Expr.StringLiteral(nodeDetails)))

        val printMessage =
            String.format(
                "f'''Execution time for %s {%s}: {%s}'''",
                operationDescriptor,
                nodeDetailsVariable.emit(),
                accumulatorVar.emit(),
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
            builder: Module.Builder,
            verboseLevel: Int,
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
                builder,
                verboseLevel < verboseThreshold,
                operationDescriptor,
                loggingTitle,
                nodeDetails,
            )
        }
    }
}
