package com.bodosql.calcite.ir

/** A streaming pipeline frame is a representation of a single
 * Pipeline. All code in the body of this frame should execute
 * within the same while loop.
 *
 * exitCond: The variable that controls when the pipeline loop is exited.
 * iterVar: The variable that tracks the count of iterations of the pipeline loop.
 * **/
class StreamingPipelineFrame(
    private var exitCond: Variable,
    private var iterVar: Variable,
    val scope: StreamingStateScope,
    val pipelineID: Int,
) : Frame {
    /** Values to initialize before the loop generation. **/
    private var initializations: MutableList<Op> = mutableListOf()

    /** Statements to execute after the loop termination. **/
    private var terminations: MutableList<Op> = mutableListOf()

    /** Variables to control whether operators output data and the index of input request to use for their values **/
    private var outputControls: MutableList<Pair<Variable, Int>> = mutableListOf()

    /** Variables for operators to request input through outputControls **/
    private var inputRequests: MutableList<Variable> = mutableListOf()

    /** Segment of code that will be updated for each operation. **/
    private var code: CodegenFrame = CodegenFrame()

    /** Is the exit variable currently synchronized **/
    private var synchronized: Boolean = false

    init {
        initExitCond()
        initIterVar()
    }

    /**
     * Emits the code for this Pipeline.
     */
    override fun emit(doc: Doc) {
        /** emit the initialized values **/
        for (initVal in initializations) {
            initVal.emit(doc)
        }
        /** Add variable tracking iteration number **/
        code.add(Op.Assign(iterVar, Expr.Binary("+", iterVar, Expr.IntegerLiteral(1))))
        /** Add operator IO control logic **/
        code.addAll(this.generateIOFlags())
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
     * Adds the list of operations to the beginning of the active Frame.
     * @param ops Operations to add to the active Frame.
     */
    override fun prependAll(ops: List<Op>) {
        code.prependAll(ops)
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
     * @param ops Operations to add to the active Frame.
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
    fun addInitialization(assign: Op) {
        initializations.add(assign)
    }

    fun initializeStreamingState(
        opID: Int,
        assign: Op.Assign,
        type: OperatorType,
        memoryEstimate: Int = -1,
    ) {
        scope.startOperator(opID, pipelineID, type, memoryEstimate)
        addInitialization(assign)
    }

    /**
     * Adds a new Op to be executed after the loop
     * This is intended if we need to "clean up" state.
     * @param term Operation to add.
     */
    fun addTermination(term: Op) {
        terminations.add(term)
    }

    fun deleteStreamingState(
        opID: Int,
        operation: Op,
    ) {
        scope.endOperator(opID, pipelineID)
        addTermination(operation)
    }

    /**
     * Adds a new Variable for controlling whether operators output data
     * @param control Variable to add
     */
    fun addOutputControl(control: Variable) {
        addInitialization(Op.Assign(control, Expr.True))
        outputControls.add(Pair(control, inputRequests.size))
    }

    /**
     * Adds a new variable for requesting input data
     * @param request Variable to add
     */
    fun addInputRequest(request: Variable) {
        inputRequests.add(request)
    }

    /**
     * "Ends" the current section of the pipeline that is
     * controlled by the current exitCond and sets a new exitCond.
     *
     * @param newExitCond The new exit condition.
     */
    fun endSection(newExitCond: Variable) {
        // Setup the new condition.
        this.exitCond = newExitCond
        initExitCond()
    }

    /**
     * Get the loop's exit condition
     * @return the loop's exit condition
     */
    fun getExitCond(): Variable {
        return exitCond
    }

    /**
     * Get the loop's variable tracking iterations
     * @return the loop's variable tracking iterations
     */
    fun getIterVar(): Variable {
        return iterVar
    }

    /**
     * Initialize the exit condition for the generated code.
     */
    private fun initExitCond() {
        synchronized = false
        initializations.add(Op.Assign(exitCond, Expr.BooleanLiteral(false)))
    }

    /**
     * Initialize the variable tracking iterations for the generated code.
     */
    private fun initIterVar() {
        initializations.add(Op.Assign(iterVar, Expr.IntegerLiteral(0)))
    }

    /**
     * Generates the code for the IO flags.
     * @return The code for the IO flags.
     */
    private fun generateIOFlags(): List<Op> {
        if (inputRequests.isEmpty()) {
            return listOf()
        }

        val ops = mutableListOf<Op>()
        for ((outputControl, i) in outputControls) {
            // If there's no input requests for this output control, skip it.
            if (i >= inputRequests.size) {
                continue
            }

            var base: Expr = inputRequests[i]
            val inputRequestSlice = inputRequests.subList(i + 1, inputRequests.size)
            for (inputRequest in inputRequestSlice) {
                base = Expr.Binary("and", inputRequest, base)
            }
            ops.add(Op.Assign(outputControl, base))
        }
        return ops
    }
}
