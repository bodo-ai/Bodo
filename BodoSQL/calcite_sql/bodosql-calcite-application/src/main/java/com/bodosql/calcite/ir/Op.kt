package com.bodosql.calcite.ir

/**
 * Represents a single operation in the emitted code.
*/
interface Op {
    /**
     * Emits the code for this operation.
     * @param doc The document to write the code into.
     */
    fun emit(doc: Doc)

    /**
     * Represents an assignment of an expression to a target
     * variable.
     * @param target Target variable.
     * @param expr Expression to evaluate.
     */
    data class Assign(val target: Variable, val expr: Expr) : Op {
        override fun emit(doc: Doc) = doc.write("${target.name} = ${expr.emit()}")
    }

    /**
     * Represents a return operation.
     * @param value Variable to return.
     */
    data class Return(val value: Variable) : Op {
        override fun emit(doc: Doc) = doc.write("return ${value.name}")
    }

    /**
     * A fallthrough to insert text directly into the document.
     * @param line Raw text to insert into the document.
     */
    class Code private constructor(private val code: StringBuilder) : Op {
        constructor(code: String) : this(code = StringBuilder(code))

        fun append(code: String): Code {
            this.code.append(code)
            return this
        }

        fun append(code: StringBuilder): Code {
            return append(code.toString())
        }

        override fun emit(doc: Doc) {
            // Trim indentation and then write non-blank lines.
            val output = code.toString()
            output.trimIndent().lineSequence().forEach {
                if (it.isNotBlank()) {
                    doc.write(it)
                }
            }
        }
    }
}
