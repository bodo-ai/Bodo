package com.bodosql.calcite.ir

import com.bodosql.calcite.application.BodoSQLCodegenException
import org.apache.calcite.rel.RelNode
import java.lang.Exception
import java.util.*

/**
 * Module is the top level compilation unit for code generation.
 * @param main The main function for this module.
 */
class Module(private val frame: Frame) {

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
    class Builder(val symbolTable: SymbolTable, private val functionFrame: Frame) {
        constructor() : this(symbolTable = SymbolTable(), functionFrame = CodegenFrame())

        private var activeFrame: Frame = functionFrame
        private var parentFrames: Stack<Frame> = Stack()
        private var assignedVariables: Set<Variable> = emptySet();

        /**
         * Helper function called by add/addall. Checks that no variable is assigned too twice.
         * This is needed due to a bug when inlining BodoSQL code into python. Throws
         * an error if a variable is assigned too twice.
         */
        private fun checkNoVariableShadowing(op: Op) {
            if (op is Op.Assign) {
                var targetVar: Variable = op.target
                if (assignedVariables.contains(targetVar)) {
                    throw Exception("Internal error in Assign.emit(): Attempted to perform an invalid variable shadow.")
                }
                assignedVariables.plus(targetVar)
            } else if (op is Op.TupleAssign) {
                for (targetVar: Variable in op.targets){
                    if (assignedVariables.contains(targetVar)) {
                        throw Exception("Internal error in Assign.emit(): Attempted to perform an invalid variable shadow.")
                    }
                    assignedVariables.plus(targetVar)
                }
            }
        }

        /**
         * Adds the operation to the end of the active Frame.
         * @param op Operation to add to the active Frame.
         */
        fun add(op: Op) {
            checkNoVariableShadowing(op)
            activeFrame.add(op)
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

        fun append(code: StringBuilder): Builder {
            return append(code.toString())
        }

        /**
         * Adds the operation to the end of the main Function.
         * This is used when we need to create state in the main function body,
         * either in streaming or case statements.
         * @param op Operation to add to the main Function's Frame.
         */
        fun addToMainFunction(op: Op) {
            functionFrame.add(op)
        }

        /**
         * This simulates appending directly to the main functions. This is used when we
         * need to create state in the main function body, either in streaming or
         * case statements.
         */

        fun appendToMainFunction(code: String): Builder {
            functionFrame.append(code)
            return this
        }

        fun appendToMainFunction(code: StringBuilder): Builder {
            return appendToMainFunction(code.toString())
        }


        fun genDataframe(rel: RelNode): Dataframe {
            val v = symbolTable.genDfVar()
            return Dataframe(v.name, rel)
        }


        /**
         * Construct a module from the built code.
         * @return The built module.
         */
        fun build(): Module {
            require(parentFrames.empty()) { "Internal Error in module.build: parentFrames stack is not empty" }
            return Module(functionFrame)
        }

        fun buildFunction(name: String, args: List<Variable>): Op.Function {
            return Op.Function(name, args, functionFrame)
        }

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
        fun startStreamingPipelineFrame(exitCond: Variable) {
            parentFrames.add(activeFrame)
            activeFrame = StreamingPipelineFrame(exitCond)
        }

        /**
         * Terminates the current active frame and returns it.
         */
        fun endFrame() : Frame {
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
        fun endCurrentStreamingPipeline() : StreamingPipelineFrame {
            if (activeFrame is StreamingPipelineFrame) {
                return endFrame() as StreamingPipelineFrame
            }
            throw BodoSQLCodegenException("Attempting to end the current streaming pipeline from outside a streaming context.")
        }

        /**
         * Returns the current frame if it is a streaming pipeline.
         * Otherwise, this raises an exception.
         */
        fun getCurrentStreamingPipeline(): StreamingPipelineFrame {
            if (activeFrame is StreamingPipelineFrame) {
                return activeFrame as StreamingPipelineFrame
            }
            throw BodoSQLCodegenException("Attempting to fetch the current streaming pipeline from outside a streaming context.")
        }
    }
}
