package com.bodosql.calcite.ir

/** A streaming pipeline frame is a representation of a single
 * Pipeline. All code in the body of this frame should execute
 * within the same while loop.
 *
 * exitCond: The variable that controls when the pipeline loop is exited.
 * **/
class StreamingPipelineFrame(var exitCond: Variable): Frame {

    /** Values to initialize before the loop generation. **/
    private var initializations: MutableList<Op.Assign> = mutableListOf()
    /** Statements to execute after the loop termination. **/
    private var terminations: MutableList<Op> = mutableListOf()
    /** Segment of code that will be updated for each operation. **/
    private var code: CodegenFrame = CodegenFrame()

    /** Is the exit variable currently synchronized **/
    private var synchronized: Boolean = false

    init {
        initExitCond()
    }

    /**
     * Emits the code for this Pipeline.
     */
    override fun emit(doc: Doc) {
        /** emit the initialized values **/
        for (initVal in initializations) {
            initVal.emit(doc)
        }
        ensureExitCondSynchronized()
        /** Generate the condition. **/
        val cond = Expr.Unary("not", exitCond)
        /** Emit the while loop. **/
        Op.While(cond, code).emit(doc)
        /** Emit any code at the end. **/
        for (term in terminations) {
            term.emit(doc)
        }

    }

    /**
     * Adds the operation to the end of the active Frame.
     * @param op Operation to add to the active Frame.
     */
    override fun add(op: Op) {
        code.add(op)
    }

    /**
     * Adds the list of operations to the end of the active Frame.
     * @param ops Operation to add to the active Frame.
     */
    override fun addAll(ops: List<Op>) {
        code.addAll(ops)
    }

    /**
     * This simulates appending code directly to a StringBuilder.
     * It is primarily meant as a way to support older code
     * and shouldn't be used anymore.
     *
     * Add operations directly instead.
     *
     * @param code: The code to append.
     */
    override fun append(code: String) {
        this.code.append(code)
    }

    /**
     * Adds a new assignment to be initialized before the loop.
     * @param assign Assignment to add.
     */
    fun addInitialization(assign: Op.Assign) {
        initializations.add(assign)
    }

    /**
     * Adds a new Op to be executed after the loop
     * This is intended if we need to "clean up" state.
     * @param assign Assignment to add.
     */
    fun addTermination(term: Op) {
        terminations.add(term)
    }


    /**
     * "Ends" the current section of the pipeline that is
     * controlled by the current exitCond and sets a new exitCond.
     *
     * This is used when an operation will output potentially more batches
     * than the previous "pipeline driver" (e.g. Join Probe). This function
     * has a side effect of synchronizing the old exitCond because the new
     * section may depend on it. This function should be called BEFORE adding
     * any code that would depend on a consistent value for the old exitCond
     * across all ranks.
     *
     * @param newExitCond The new exit condition.
     */
    fun endSection(newExitCond: Variable) {
        // Synchronize the old value.
        ensureExitCondSynchronized()
        // Setup the new condition.
        this.exitCond = newExitCond
        initExitCond()
    }

    /**
     * Generate to ensure the result of the exit condition is synchronized across all ranks.
     */
    fun ensureExitCondSynchronized() {
        if (!synchronized) {
            // TODO: Move Logical_And.value from Expr.Raw
            val andOp =
                Expr.Call("np.int32", listOf(Expr.Raw("bodo.libs.distributed_api.Reduce_Type.Logical_And.value")))
            // Generate the MPI call
            val syncCall = Expr.Call("bodo.libs.distributed_api.dist_reduce", listOf(exitCond, andOp))
            code.add(Op.Assign(exitCond, syncCall))
        }
        synchronized = true
    }

    /**
     * Initialize the exit condition for the generated code.
     */
    private fun initExitCond() {
        synchronized = false
        initializations.add(Op.Assign(exitCond, Expr.BooleanLiteral(false)))
    }
}
