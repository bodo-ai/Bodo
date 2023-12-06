package com.bodosql.calcite.ir

import java.math.BigDecimal

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
    class Raw(val code: String) : Expr() {
        override fun emit(): String = code
    }

    open class DelegateExpr(private val delegate: Expr) : Expr() {
        override fun emit(): String = delegate.emit()
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
    class Call(val callee: Expr, val args: kotlin.collections.List<Expr> = listOf(), val namedArgs: kotlin.collections.List<Pair<String, Expr>> = listOf()) : Expr() {
        constructor(callee: Expr, args: kotlin.collections.List<Expr>) : this(callee, args, listOf())
        constructor(callee: Expr, vararg args: Expr) : this(callee, args.toList(), listOf())
        constructor(callee: String, args: kotlin.collections.List<Expr>, namedArgs: kotlin.collections.List<Pair<String, Expr>>) :
            this(Raw(callee), args, namedArgs)
        constructor(callee: String, args: kotlin.collections.List<Expr>) : this(callee, args, listOf())
        constructor(callee: String, vararg args: Expr) : this(callee, args.toList())

        override fun emit(): String {
            val args = sequenceOf(
                this.args.asSequence().map { it.emit() },
                this.namedArgs.asSequence().map { (name, value) -> "$name=${value.emit()}" },
            ).flatten().joinToString(separator = ", ")
            return "${callee.emit()}($args)"
        }
    }

    class Len(expr: Expr) : DelegateExpr(
        Call("len", expr),
    )

    /**
     * Controls access to fields for an input. This can be used to access member variables
     * or function definitions.
     *
     * @param inputVar the expression to access a field for.
     * @param attributeName the field to access.
     */
    class Attribute(val inputVar: Expr, val attributeName: String) : Expr() {
        override fun emit(): String = "${inputVar.emit()}.$attributeName"
    }

    /**
     * Represents a taking a slice of the rows of a dataframe.
     *
     * @param inputDf: The dataframe to be sliced
     * @param lowerBound: The integer lower bound of the slice
     * @param upperBound: The integer upper bound of the slice
     */
    class DataFrameSlice(inputDf: Expr, lowerBound: Expr, upperBound: Expr) : DelegateExpr(
        Index(
            Attribute(inputDf, "iloc"),
            Slice(lowerBound, upperBound),
        ),
    )

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
    class Method(
        inputVar: Expr,
        methodName: String,
        args: kotlin.collections.List<Expr> = listOf(),
        namedArgs: kotlin.collections.List<Pair<String, Expr>> = listOf(),
    ) :
        DelegateExpr(
            Call(Attribute(inputVar, methodName), args, namedArgs),
        )

    /**
     * Represents a call to groupby with the arguments that could change depending on the call.
     *
     * @param inputVar: The input value whose method is being invoked.
     * @param keys: The groupby keys.
     * @param asIndex: Should keys be returned as an index.
     * @param dropna: Should NA values be dropped.
     *
     */
    class Groupby(inputVar: Expr, keys: List, asIndex: Boolean, dropna: Boolean) :
        DelegateExpr(
            Method(
                inputVar,
                "groupby",
                listOf(keys),
                listOf(
                    "as_index" to BooleanLiteral(asIndex),
                    "dropna" to BooleanLiteral(dropna),
                    "_is_bodosql" to BooleanLiteral(true),
                ),
            ),
        )

    /**
     * Represents an index operation into an expression. This it the equivalent
     * of getitem most of the time.
     */
    class Index(val input: Expr, val args: kotlin.collections.List<Expr>) : Expr() {
        constructor(input: Expr, vararg args: Expr) : this(input, args.toList())

        override fun emit(): String {
            val index = args.joinToString(separator = ", ") { it.emit() }
            return "${input.emit()}[$index]"
        }
    }

    /**
     * Represents a call to sort_values with the arguments that could change depending on the call.
     *
     * @param inputVar: The input value whose method is being invoked.
     * @param ascending: Sort ascending vs. descending
     * @param naPosition: Puts NaNs at the beginning if 'first'; 'last' puts NaNs at the end.
     *
     */
    class SortValues(val inputVar: Expr, val by: List, val ascending: List, val naPosition: List) : Expr() {

        override fun emit(): String {
            // Generate the keyword args
            val keywordArgs = listOf(
                Pair("by", by),
                Pair("ascending", ascending),
                Pair("na_position", naPosition),
            )

            return if (by.args.isEmpty() && ascending.args.isEmpty() && naPosition.args.isEmpty()) {
                inputVar.emit()
            } else {
                Method(inputVar, "sort_values", listOf(), keywordArgs).emit()
            }
        }
    }

    /**
     * Represents an array or DataFrame getitem call with the given
     * index.
     *
     * @param inputExpr: The input array/DataFrame.
     * @param index: The index into the array/DataFrame
     */
    class GetItem(inputExpr: Expr, index: Expr) : DelegateExpr(
        Index(inputExpr, index),
    )

    /**
     * Represents the unary operator. This should only
     * be called on scalars because it doesn't match SQL NULL
     * semantics
     *
     * @param opString: The string for the input unary op. This is a symbol
     * not a name.
     * @param inputExpr: The input expression to apply the op to.
     */
    class Unary(private val opString: String, private val inputExpr: Expr) : Expr() {

        override fun emit(): String {
            return "$opString(${inputExpr.emit()})"
        }
    }

    /**
     * Represents the binary operator. This should only
     * be called on scalars because it doesn't match SQL NULL
     * semantics
     *
     * @param opString: The string for the input binop. This is a symbol
     * not a name.
     * @param input1: The first input to the op.
     * @param input2: The second input to the op.
     */
    class Binary(private val opString: String, private val input1: Expr, private val input2: Expr) : Expr() {

        override fun emit(): String {
            // TODO(jsternberg): As things get more complex in here, we'll have to mark
            // operations with their precedence.
            return "(${input1.emit()} $opString ${input2.emit()})"
        }
    }

    /**
     * Represents a range creation.
     * @param start The start Expr.
     * @param stop The stop Expr.
     * @param step The step Expr. Null if there is no step.
     */
    class Range(start: Expr, stop: Expr, step: Expr? = null) : DelegateExpr(
        Call("range", sequenceOf(start, stop, step).filterNotNull().toList()),
    )

    /**
     * Represents a tuple creation.
     * @param args The inputs to the tuple.
     */
    class Tuple(val args: kotlin.collections.List<Expr>) : Expr() {
        constructor(vararg args: Expr) : this(args.toList())

        override fun emit(): String = if (args.size == 1) {
            // Special handling for length 1, so they aren't treated like
            // parenthesis.
            "(${args[0].emit()},)"
        } else {
            args.joinToString(separator = ", ", prefix = "(", postfix = ")") { it.emit() }
        }
    }

    /**
     * Represents the slice syntax of start:stop.
     *
     * This syntax DOES NOT perform the slice operation but is mostly meant to
     * be a parameter to the index operation.
     *
     * @param start start index of the slice or null if empty
     * @param stop stop index of the slice or null if empty
     */
    class Slice(val start: Expr? = null, val stop: Expr? = null, val step: Expr? = null) : Expr() {
        /**
         * Constructor without step mostly for Java.
         */
        constructor(start: Expr?, stop: Expr?) : this(start, stop, null)

        override fun emit(): String {
            val s = "${start?.emit() ?: ""}:${stop?.emit() ?: ""}"
            return if (step != null) {
                "$s:${step.emit()}"
            } else {
                s
            }
        }
    }

    /**
     * Represents a List creation.
     * @param args The inputs to the list.
     */
    class List(val args: kotlin.collections.List<Expr>) : Expr() {
        constructor(vararg args: Expr) : this(args.toList())

        override fun emit(): String =
            args.joinToString(separator = ", ", prefix = "[", postfix = "]") { it.emit() }
    }

    /**
     * Represents a Dictionary creation. Keys and values must
     * be the same length.
     * @param keys The key inputs to the dictionary.
     * @param values The value inputs to the dictionary
     */
    class Dict(val items: kotlin.collections.List<Pair<Expr, Expr>>) : Expr() {
        constructor(items: Iterator<Map.Entry<Expr, Expr>>) : this(
            items.asSequence()
                .map { (k, v) -> k to v }
                .toList(),
        )

        constructor(vararg items: Pair<Expr, Expr>) : this(items.toList())

        override fun emit(): String =
            items.joinToString(separator = ", ", prefix = "{", postfix = "}") { (k, v) ->
                "${k.emit()}: ${v.emit()}"
            }
    }

    /**
     * Represents a triple quoted String.
     * @param arg The body of the string.
     */
    class TripleQuotedString(arg: String) : Expr() {
        private val s = arg.replace("\"\"\"", """\"\"\"""")

        override fun emit(): String = "\"\"\"$s\"\"\""
    }

    /**
     * Represents a triple quoted String built from a Frame.
     * This is used when several expressions need to be passed in a global variable (e.g. Case).
     * @param arg The body of the string as a Frame.
     * @param indentLevel The indent level used for emitting the Frame.
     */
    class FrameTripleQuotedString(val arg: Frame, private val indentLevel: Int) : Expr() {

        override fun emit(): String {
            // Generate a doc with indent level provided + emit the frame
            val doc = Doc(level = indentLevel)
            arg.emit(doc)
            // Generate the body
            val body = doc.toString().replace("\"\"\"", """\"\"\"""")
            return "\"\"\"$body\"\"\""
        }
    }

    /**
     * Represents a Single literal wrapped in regular
     * double quotes.
     * @param arg The body of the string.
     */
    class StringLiteral(val arg: String) : Expr() {
        override fun emit(): String {
            val literal = arg
                .replace("""["\\\n\r\t\u0008\f]""".toRegex()) { m ->
                    when (val v = m.value) {
                        "\n" -> """\n"""
                        "\r" -> """\r"""
                        "\t" -> """\t"""
                        "\b" -> """\b"""
                        "\u000c" -> """\f"""
                        else -> "\\$v"
                    }
                }
            return "\"$literal\""
        }
    }

    /**
     * Represents a Binary literal wrapped in double quotes for binary strings.
     * @param arg The body of the binary.
     */
    class BinaryLiteral(val arg: String) : Expr() {
        override fun emit(): String = "b\"$arg\""
    }

    /**
     * Represents a Boolean Literal.
     * @param arg The value of the literal.
     */
    class BooleanLiteral(val arg: Boolean) : Expr() {
        override fun emit(): String =
            if (arg) {
                "True"
            } else {
                "False"
            }
    }

    object True : DelegateExpr(BooleanLiteral(true))
    object False : DelegateExpr(BooleanLiteral(false))

    /**
     * Represents an Integer Literal.
     * @param arg The value of the literal.
     */
    class IntegerLiteral(val arg: Int) : Expr() {
        override fun emit(): String = arg.toString()
    }

    /**
     * Represents a Decimal Literal.
     * @param arg The value of the literal.
     */
    class DecimalLiteral(val arg: BigDecimal) : Expr() {
        override fun emit(): String = arg.toString()
    }

    /**
     * Represents a Double Literal.
     * @param arg The value of the literal.
     */
    class DoubleLiteral(val arg: Double) : Expr() {
        override fun emit(): String = arg.toString()
    }

    /**
     * Represents a Python None value.
     */
    object None : Expr() {
        override fun emit(): String = "None"
    }

    companion object {
        val Zero = IntegerLiteral(0)
        val One = IntegerLiteral(1)
    }
}

fun BodoSQLKernel(callee: String, args: List<Expr> = listOf(), namedArgs: List<Pair<String, Expr>> = listOf()): Expr {
    return Expr.Call("bodo.libs.bodosql_array_kernels.$callee", args, namedArgs)
}

fun initRangeIndex(args: List<Expr> = listOf(), namedArgs: List<Pair<String, Expr>> = listOf()): Expr {
    return Expr.Call("bodo.hiframes.pd_index_ext.init_range_index", args, namedArgs)
}
