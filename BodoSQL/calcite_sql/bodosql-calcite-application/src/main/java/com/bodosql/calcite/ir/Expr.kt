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
    data class Call(val callee: String, val args: kotlin.collections.List<Expr> = listOf()) : Expr() {
        constructor(callee: String, vararg args: Expr) : this(callee, args.toList())

        override fun emit(): String {
            val args = this.args.joinToString(separator = ", ") { it.emit() }
            return "${callee}(${args})"
        }
    }

    /**
     * Represents a Python method call. This should be used as a generic call for certain methods that are unlikely to repeat.
     * For common operations (e.g. Groupby) we may want to implement an optimized implementation.
     *
     * @param inputVar: The input value whose method is being invoked.
     * @param methodName: The name of the method being called.
     * @param args: The arguments to be passed as regular arguments.
     * @param namedArgs: The arguments to be passed as keyword arguments. Note we use a pair instead of a map
     * to ensure the order is deterministic.
     */
    data class Method(val inputVar: Expr, val methodName: String, val args: kotlin.collections.List<Expr> = listOf(), val namedArgs: kotlin.collections.List<Pair<String, Expr>> = listOf()) : Expr() {


        override fun emit(): String {
            var regularArgs = ""
            var keywordArgs = ""
            if (this.args.isNotEmpty()) {
                regularArgs = this.args.joinToString(separator = ", ", postfix = ", ") { it.emit() }
            }
            if (namedArgs.isNotEmpty()) {
                keywordArgs = this.namedArgs.map{ Raw(it.first + " = " + it.second.emit())}.joinToString(separator = ", ") { it.emit() }
            }
            return "${inputVar.emit()}.$methodName(${regularArgs}${keywordArgs})"
        }
    }

    /**
     * Represents a call to groupby with the arguments that could change depending on the call.
     *
     * @param inputVar: The input value whose method is being invoked.
     * @param keys: The groupby keys.
     * @param asIndex: Should keys be returned as an index.
     * @param dropna: Should NA values be dropped.
     *
     */
    data class Groupby(val inputVar: Expr, val keys: Expr.List, val asIndex: Boolean, val dropna: Boolean) : Expr() {


        override fun emit(): String {
            // Generate the keyword args
            val keywordArgs = listOf(Pair("as_index", BooleanLiteral(asIndex)), Pair("dropna", BooleanLiteral(dropna)), Pair("_is_bodosql", BooleanLiteral(true)));
            return Method(inputVar, "groupby", listOf(keys), keywordArgs).emit()
        }
    }

    /**
     * Represents a tuple creation.
     * @param args The inputs to the tuple.
     */
    data class Tuple(val args: kotlin.collections.List<Expr>) : Expr() {
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
     * Represents a List creation.
     * @param args The inputs to the list.
     */
    data class List(val args: kotlin.collections.List<Expr>) : Expr() {
        override fun emit(): String {
            val listArgs = args.joinToString(separator = ", ") { it.emit() }
            return "[${listArgs}]"
        }
    }

    /**
     * Represents a triple quoted String.
     * @param arg The body of the string.
     */
    data class TripleQuotedString(val arg: Expr) : Expr() {
        override fun emit(): String = "\"\"\"${arg.emit()}\"\"\""
    }

    /**
     * Represents a Single literal wrapped in regular
     * double quotes.
     * @param arg The body of the string.
     */
    data class StringLiteral(val arg: Expr) : Expr() {
        override fun emit(): String = "\"${arg.emit()}\""
    }

    /**
     * Represents a Boolean Literal.
     * @param arg The value of the literal.
     */
    data class BooleanLiteral(val arg: Boolean) : Expr() {
        override fun emit(): String {
            if (arg) {
                return "True"
            } else {
                return "False"
            }
        }
    }
}
