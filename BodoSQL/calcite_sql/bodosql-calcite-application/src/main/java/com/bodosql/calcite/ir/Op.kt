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

        override fun emit(doc: Doc) {
            //Assertion to check that we're not shadowing variables
            //by assigning to them multiple times

            doc.write("${target.name} = ${expr.emit()}")
        }

    }

    /**
     * Represents an expression without a return value.
     * @param expr Expression to evaluate.
     */
    data class Stmt(val expr: Expr) : Op {
        override fun emit(doc: Doc) = doc.write(expr.emit())
    }

    /**
     * Represents an if operation with an optional else case
     * for bodies with a single line.
     */
    data class If(val cond: Expr, val ifFrame: Frame, val elseFrame: Frame) : Op {

        override fun emit(doc: Doc) {
            doc.write("if ${cond.emit()}:")
            ifFrame.emit(doc.indent())
            doc.write("else:")
            elseFrame.emit(doc.indent())
        }

    }

    /**
     * Represents a return statement
     */
    data class ReturnStatement( val retVal: Variable?) : Op {

        override fun emit(doc: Doc) {
            if (retVal != null) {
                doc.write("return ${this.retVal.emit()}")
            } else {
                doc.write("return")
            }
        }

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
