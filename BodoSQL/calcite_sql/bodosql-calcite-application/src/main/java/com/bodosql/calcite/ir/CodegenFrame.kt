package com.bodosql.calcite.ir

/**
 * CodegenFrame is roughly equivalent to a Python Frame, consisting of
 * operations found within the same scope.
 */
class CodegenFrame : Frame {
    private var code: MutableList<Op> = mutableListOf()

    /**
     * Emits the code for this Frame.
     */
    override fun emit(doc: Doc) {
        for (item in code) {
            item.emit(doc)
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
     * @param
     */
    override fun append(code: String) {
        // If the last operation is an Op.Code,
        // append directly to that.
        val op = this.code.lastOrNull()
        if (op != null && op is Op.Code) {
            op.append(code)
        } else {
            // Otherwise, create a new Op.Code.
            this.code.add(Op.Code(code))
        }
    }
}
