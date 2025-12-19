package com.bodosql.calcite.ir

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer.Companion.createSingleBatchTimer
import com.bodosql.calcite.application.utils.JoinStateCache
import org.apache.calcite.rel.RelNode
import java.util.Stack

/**
 * Module is the top level compilation unit for code generation.
 * @param frame The main function frame for this module.
 */
class Module(
    private val frame: Frame,
) {
    /**
     * Emits the code for the module.
     * @return Emitted code for the full module.
     */
    fun emit(level: Int = 0): String {
        val doc = Doc(level = level)
        frame.emit(doc)
        return doc.toString()
    }

    /**
     * Builder is used to construct a new module.
     */
    class Builder(
        val symbolTable: SymbolTable,
        private val functionFrame: Frame,
        private var hideOperatorIDs: Boolean,
        val verboseLevel: Int,
    ) {
        constructor(
            verboseLevel: Int,
        ) : this(symbolTable = SymbolTable(), functionFrame = CodegenFrame(), hideOperatorIDs = false, verboseLevel)
        constructor() : this(symbolTable = SymbolTable(), functionFrame = CodegenFrame(), hideOperatorIDs = false, verboseLevel = 0)

        private val scope: StreamingStateScope = StreamingStateScope()

        // Info about Snowflake-managed Iceberg tables to generate a prefetch call for
        // TODO: Is there a better place to put this? We need this info to insert a call
        //       at the beginning of codegen
        private var sfConnStr: String? = null
        private var sfIcebergTablePaths = mutableSetOf<String>()

        private var activeFrame: Frame = functionFrame
        private var parentFrames: Stack<Frame> = Stack()
        private var assignedVariables: Set<Variable> = emptySet()

        private var currentPipeline: Int = 0
        private var nodeToOperatorCounters: MutableMap<Int, Int> = mutableMapOf()

        private var idMapping: Map<Int, Int> = mapOf()

        /**
         * Add a Snowflake-managed Iceberg table to track for prefetching generation
         */
        fun addSfIcebergTablePath(
            connExpr: Expr,
            tablePath: String,
        ) {
            if (connExpr !is Expr.StringLiteral) {
                throw Exception("Internal Error: Snowflake connection strings are not constant")
            }
            if (sfConnStr != null && sfConnStr != connExpr.arg) {
                throw Exception("Internal Error: Multiple Snowflake connection strings found")
            }
            sfConnStr = connExpr.arg
            sfIcebergTablePaths.add(tablePath)
        }

        fun setIDMapping(minId: Map<Int, Int>) {
            idMapping = minId
        }

        fun setHideOperatorIDs(flag: Boolean) {
            this.hideOperatorIDs = flag
        }

        /**
         * Generate a new operator ID based on the relnode ID. This makes it possible to determine which relnodes correspond to operators by reading the generated code later.
         */
        fun newOperatorID(n: RelNode): OperatorID {
            val opID = idMapping[n.id]!!
            if (!nodeToOperatorCounters.contains(opID)) {
                nodeToOperatorCounters[opID] = 0
            }

            val newId = nodeToOperatorCounters[opID]!! + 1
            nodeToOperatorCounters[opID] = newId

            val multiplier = 1000
            // If this assert ever fails in practice, we can increase the multiplier by factors of 10 here.
            assert(newId < multiplier)

            return OperatorID(opID * multiplier + newId, hideOperatorIDs)
        }

        private val joinStateCache = JoinStateCache()

        /**
         * Helper function called by add/addall. Checks that no variable is assigned too twice.
         * This is needed due to a bug when inlining BodoSQL code into python. Throws
         * an error if a variable is assigned too twice.
         */
        private fun checkNoVariableShadowing(op: Op) {
            if (op is Op.Assign) {
                val targetVar: Variable = op.target
                if (assignedVariables.contains(targetVar)) {
                    throw Exception("Internal error in Assign.emit(): Attempted to perform an invalid variable shadow.")
                }
                assignedVariables.plus(targetVar)
            } else if (op is Op.TupleAssign) {
                for (targetVar: Variable in op.targets) {
                    if (assignedVariables.contains(targetVar)) {
                        throw Exception("Internal error in Assign.emit(): Attempted to perform an invalid variable shadow.")
                    }
                    assignedVariables.plus(targetVar)
                }
            }
        }

        /**
         * getter for the JoinStateCache
         */
        fun getJoinStateCache(): JoinStateCache = joinStateCache

        /**
         * Adds the operation to the end of the active Frame.
         * @param op Operation to add to the active Frame.
         */
        fun add(op: Op) {
            checkNoVariableShadowing(op)
            activeFrame.add(op)
        }

        /**
         * Adds an assignment that can be safely hoisted from
         * a loop (is both pure and scalar). This is intended
         * as a generic API that will hoist a statement out
         * of the loop for streaming but act as normal for
         * non-streaming.
         *
         * TODO(njriasan): Consider adding type requirements/make this
         * less manual.
         *
         * @param op Operation to add to the active Frame. If
         * we are streaming this should be hoisted from the loop.
         */
        fun addPureScalarAssign(op: Op.Assign) {
            checkNoVariableShadowing(op)
            if (isStreamingFrame()) {
                getCurrentStreamingPipeline().addInitialization(op)
            } else {
                activeFrame.add(op)
            }
        }

        /**
         * Adds the list of operations to the end of the module.
         * @param ops Operations to add to the module.
         */
        fun addAll(ops: List<Op>) {
            ops.forEach { it: Op -> checkNoVariableShadowing(it) }
            activeFrame.addAll(ops)
        }

        /**
         * This simulates appending code directly to a StringBuilder.
         * It is primarily meant as a way to support older code
         * and shouldn't be used anymore.
         *
         * Add operations directly instead.
         *
         * @param
         */
        fun append(code: String): Builder {
            activeFrame.append(code)
            return this
        }

        fun append(code: StringBuilder): Builder = append(code.toString())

        /**
         * Generate the Snowflake-managed Iceberg prefetch call
         */
        private fun genSfIcebergPrefetch() {
            if (sfIcebergTablePaths.size == 0) {
                return
            }

            val timerInfo =
                createSingleBatchTimer(
                    this,
                    this.verboseLevel,
                    "prefetching SF-managed Iceberg metadata",
                    "PREFETCH TIMING",
                    sfIcebergTablePaths.sorted().joinToString(", "),
                    SingleBatchRelNodeTimer.OperationType.IO_BATCH,
                )

            val ops = mutableListOf<Op>()
            timerInfo.genStartTimer()?.let { ops.add(it) }
            ops.add(
                Op.Stmt(
                    Expr.Call(
                        "bodo.io.iceberg.sf_prefetch.prefetch_sf_tables_njit",
                        Expr.StringLiteral(sfConnStr!!),
                        Expr.List(sfIcebergTablePaths.map { Expr.StringLiteral(it) }),
                    ),
                ),
            )
            ops.addAll(timerInfo.genEndTimer())

            functionFrame.prependAll(ops)
        }

        /**
         * Construct a module from the built code.
         * @return The built module.
         */
        fun build(): Module {
            require(parentFrames.empty()) { "Internal Error in module.build: parentFrames stack is not empty" }
            if (scope.hasOperators()) {
                scope.addToFrame(functionFrame)

                // Generate Snowflake-managed Iceberg prefetch call
                this.genSfIcebergPrefetch()
            }

            return Module(functionFrame)
        }

        fun buildFunction(
            name: String,
            args: List<Variable>,
        ): Op.Function = Op.Function(name, args, functionFrame)

        /**
         * Updates a builder to create a new activeFrame
         * as a CodegenFrame.
         */
        fun startCodegenFrame() {
            parentFrames.add(activeFrame)
            activeFrame = CodegenFrame()
        }

        /**
         * Updates a builder to create a new activeFrame
         * as a StreamingPipelineFrame.
         */
        fun startStreamingPipelineFrame(
            exitCond: Variable,
            iterVar: Variable,
        ) {
            parentFrames.add(activeFrame)
            activeFrame = StreamingPipelineFrame(exitCond, iterVar, scope, currentPipeline)
            currentPipeline++
        }

        /**
         * Terminates the current active frame and returns it.
         */
        fun endFrame(): Frame {
            if (parentFrames.empty()) {
                throw BodoSQLCodegenException("Attempting to end a Frame when there are 0 remaining parent frames.")
            }
            val res = activeFrame
            activeFrame = parentFrames.pop()
            return res
        }

        /**
         * Terminates the current streaming pipeline and returns it.
         * If the current frame is not a streaming pipeline it raises
         * an exception.
         */
        fun endCurrentStreamingPipeline(): StreamingPipelineFrame {
            if (isStreamingFrame()) {
                return endFrame() as StreamingPipelineFrame
            }
            throw BodoSQLCodegenException("Attempting to end the current streaming pipeline from outside a streaming context.")
        }

        /**
         * Returns the current frame if it is a streaming pipeline.
         * Otherwise, this raises an exception.
         */
        fun getCurrentStreamingPipeline(): StreamingPipelineFrame {
            if (isStreamingFrame()) {
                return activeFrame as StreamingPipelineFrame
            }
            throw BodoSQLCodegenException("Attempting to fetch the current streaming pipeline from outside a streaming context.")
        }

        /**
         * Helper function that force registers the current
         * pipeline as the end pipeline of an operator.
         * This is currently only used for CombineStreamExchange
         * as it doesn't call currentStreamingPipeline.deleteState
         */
        fun forceEndOperatorAtCurPipeline(
            opID: OperatorID,
            pipeline: StreamingPipelineFrame,
        ) {
            scope.endOperator(opID, pipeline.pipelineID)
        }

        /**
         * Returns if we are in a streaming frame.
         */
        fun isStreamingFrame(): Boolean = activeFrame is StreamingPipelineFrame
    }
}
