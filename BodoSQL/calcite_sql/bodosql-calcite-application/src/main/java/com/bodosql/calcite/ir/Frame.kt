package com.bodosql.calcite.ir

/**
 * Frame is equivalent to a Python Frame, consisting of
 * operations found within the same scope.
 */
interface Frame {
    /**
     * Emits the code for this Frame.
     */
    fun emit(doc: Doc)

    /**
     * Adds the list of operations to the beginning of the active Frame.
     * @param ops Operations to add to the active Frame.
     */
    fun prependAll(ops: List<Op>)

    /**
     * Adds the operation to the end of the active Frame.
     * @param op Operation to add to the active Frame.
     */
    fun add(op: Op)

    /**
     * Adds the list of operations to the end of the active Frame.
     * @param ops Operations to add to the active Frame.
     */
    fun addAll(ops: List<Op>)

    /**
     * This simulates appending code directly to a StringBuilder.
     * It is primarily meant as a way to support older code
     * and shouldn't be used anymore.
     *
     * Add operations directly instead.
     *
     * @param code: The code to append.
     */
    fun append(code: String)
}
