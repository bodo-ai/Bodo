package com.bodosql.calcite.ir

abstract class Expr {
    /**
     * Emits code for this expression.
     * @return The python code that represents this expression.
     */
    abstract fun emit(): String

    /**
     * Represents raw python code to insert directly into the document.
     * @param code The code to insert directly into the document.
     */
    data class Raw(val code: String) : Expr() {
        override fun emit(): String = code
    }

    /**
     * Represents a call to a python function with arguments.
     * While python allows call expressions to be on arbitrary expressions,
     * we choose to only allow calls on function names.
     *
     * TODO(jsternberg): If needed in the future, we might want to consider
     * creating a type for function definitions and making the argument to
     * callee one of the function definitions. This could help with type safety.
     * If someone in the future is reading this and it hasn't been done, it
     * probably wasn't important.
     *
     * @param callee the function this expression will invoke.
     * @param args a list of expressions to be used as arguments.
     */
    data class Call(val callee: String, val args: List<Expr> = listOf()) : Expr() {
        constructor(callee: String, vararg args: Expr) : this(callee, args.toList())

        override fun emit(): String {
            val args = this.args.joinToString(separator = ", ") { it.emit() }
            return "${callee}(${args})"
        }
    }

    /**
     * Represents a tuple creation.
     * @param args The inputs to the tuple.
     */
    data class Tuple(val args: List<Expr>) : Expr() {
        override fun emit(): String {
            if (args.isEmpty()) {
                return "()"
            }
            // Note we use postfix to ensure tuples of length 1 work.
            val tupleArgs = args.joinToString(separator = ", ", postfix = ",") { it.emit() }
            return "(${tupleArgs})"
        }
    }
    /**
     * Represents a triple quoted String.
     * @param arg The body of the string.
     */
    data class TripleQuotedString(val arg: Expr) : Expr() {
        override fun emit(): String = "\"\"\"${arg.emit()}\"\"\""
    }
}
