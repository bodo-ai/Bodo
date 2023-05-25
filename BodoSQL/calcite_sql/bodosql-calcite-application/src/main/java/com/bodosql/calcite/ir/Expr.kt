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
    data class Call(val callee: String, val args: kotlin.collections.List<Expr> = listOf(), val namedArgs: kotlin.collections.List<Pair<String, Expr>> = listOf()) : Expr() {
        constructor(callee: String, args: kotlin.collections.List<Expr>) : this(callee, args, listOf())
        constructor(callee: String, vararg args: Expr) : this(callee, args.toList(), listOf())
        
        override fun emit(): String {
            val posArgs = args.asSequence().map { it.emit() }
            val namedArgs = namedArgs.asSequence().map { (name, value) -> "${name}=${value.emit()}" }
            val args = (posArgs + namedArgs).joinToString(separator = ", ")
            return "${callee}(${args})"
        }
    }

    data class Len(val expr: Expr) : Expr() {
        override fun emit(): String {
            return "len(${expr.emit()})"
        }
    }

    data class Attribute(val inputVar: Expr, val attributeName: String): Expr() {
        override fun emit(): String {
            return "${inputVar.emit()}.${attributeName}"
        }
    }


    /**
     * Represents a taking a slice of the rows of a dataframe.
     *
     * @param inputDf: The dataframe to be sliced
     * @param lowerBound: The integer lower bound of the slice
     * @param upperBound: The integer upper bound of the slice
     */
    data class DataFrameSlice(val inputDf: Expr, val lowerBound: Expr, val upperBound: Expr) : Expr() {

        override fun emit(): String {
            return "(${inputDf.emit()}).iloc[${lowerBound.emit()} : ${upperBound.emit()}]"
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
            val keywordArgs = listOf(Pair("as_index", BooleanLiteral(asIndex)), Pair("dropna", BooleanLiteral(dropna)), Pair("_is_bodosql", BooleanLiteral(true)))
            return Method(inputVar, "groupby", listOf(keys), keywordArgs).emit()
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
    data class SortValues(val inputVar: Expr, val by: Expr.List, val ascending: Expr.List, val naPosition: Expr.List) : Expr() {

        override fun emit(): String {
            // Generate the keyword args
            val keywordArgs = listOf(Pair("by", by), Pair("ascending", ascending),
                                     Pair("na_position", naPosition))

            return if (by.args.isEmpty() && ascending.args.isEmpty() && naPosition.args.isEmpty()) {
                inputVar.emit();
            } else {
                Method(inputVar, "sort_values", listOf(), keywordArgs).emit();
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
    data class GetItem(val inputExpr: Expr, val index: Expr) : Expr() {

        override fun emit(): String {
            return "${inputExpr.emit()}[${index.emit()}]"
        }

    }


    data class Slice(val args: kotlin.collections.List<Expr> = listOf()) : Expr() {

        override fun emit(): String {
            val slice = args.joinToString(separator = ":") { it.emit() }
            if (slice.isEmpty()) {
                return ":"
            }
            return slice
        }
    }


    /**
     * Represents the unary operator. This should only
     * be called on scalars because it doesn't match SQL NULL
     * semantics
     *
     * @param opString: The string for the input unary op. This is a symbol
     * not a name.
     * @param inputExpr: The input expression to apply the op to.
     */
    data class Unary(val opString: String, val inputExpr: Expr) : Expr() {

        override fun emit(): String {
            return "${opString}(${inputExpr.emit()})"
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
    data class Binary(val opString: String, val input1: Expr, val input2: Expr) : Expr() {

        override fun emit(): String {
            return "(${input1.emit()} $opString ${input2.emit()})"
        }

    }

    /**
     * Represents a range creation.
     * @param start The start Expr.
     * @param stop The stop Expr.
     * @param step The step Expr. Null if there is no step.
     */
    data class Range(val start: Expr, val stop: Expr, val step: Expr? = null) : Expr() {
        override fun emit(): String {
            if (step == null) {
                return "range(${start.emit()}, ${stop.emit()})"
            } else {
                return "range(${start.emit()}, ${stop.emit()}, ${step.emit()})"
            }
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
     * Represents a Dictionary creation. Keys and values must
     * be the same length.
     * @param keys The key inputs to the dictionary.
     * @param values The value inputs to the dictionary
     */
    data class Dict(val keys: kotlin.collections.List<StringLiteral>, val values: kotlin.collections.List<Expr>) : Expr() {
        override fun emit(): String {
            val mergedValues = keys zip values
            val dictArgs = mergedValues.joinToString(separator = ", ") { it.first.emit() + " : " + it.second.emit() }
            return "{${dictArgs}}"
        }
    }

    data class PandasDataFrame(val data: Expr, val index: kotlin.collections.List<Expr>): Expr() {
        /*
        New code shouldn't be using this IR, as we should be using init_dataframe
        instead of constructing the pd.DataFrame() directly
         */
        override fun emit(): String {
            return when (index.size) {
                0 -> "pd.DataFrame(${data.emit()})"
                1 -> "pd.DataFrame(${data.emit()}, index=${index[0].emit()})"
                else -> {
                    val indexStr = index.joinToString(separator = ", " ) { it.emit()}
                    "pd.DataFrame(${data.emit()}, index=[${indexStr}])"
                }
            }
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
    data class StringLiteral(val arg: String) : Expr() {
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

    /**
     * Represents an Integer Literal.
     * @param arg The value of the literal.
     */
    data class IntegerLiteral(val arg: Int) : Expr() {
        override fun emit(): String = arg.toString()

    }

    /**
     * Represents a Decimal Literal.
     * @param arg The value of the literal.
     */
    data class DecimalLiteral(val arg: BigDecimal) : Expr() {
        override fun emit(): String = arg.toString()
    }

    /**
     * Represents a Double Literal.
     * @param arg The value of the literal.
     */
    data class DoubleLiteral(val arg: Double) : Expr() {
        override fun emit(): String = arg.toString()

    }

    /**
     * Represents a Python None value.
     */
    object None : Expr() {
        override fun emit(): String = "None"

    }
}


fun BodoSQLKernel(callee: String, args: List<Expr> = listOf(), namedArgs: List<Pair<String, Expr>> = listOf()) : Expr {
    return Expr.Call("bodo.libs.bodosql_array_kernels.${callee}", args, namedArgs);
}

fun initRangeIndex(args: List<Expr> = listOf(), namedArgs: List<Pair<String, Expr>> = listOf()) : Expr {
    return Expr.Call("bodo.hiframes.pd_index_ext.init_range_index", args, namedArgs);
}
