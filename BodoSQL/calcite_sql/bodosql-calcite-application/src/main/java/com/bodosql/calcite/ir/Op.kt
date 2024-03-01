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
            // Assertion to check that we're not shadowing variables
            // by assigning to them multiple times

            doc.write("${target.name} = ${expr.emit()}")
        }
    }

    /**
     * Represents an assignment of an expression to a tuple of variables
     * variable.
     * @param targets Target variables.
     * @param expr Expression to evaluate.
     */
    data class TupleAssign(val targets: List<Variable>, val expr: Expr) : Op {
        override fun emit(doc: Doc) {
            // If we have no variables to assign to, this is a no-op
            if (targets.isEmpty()) {
                return
            }
            var tupleBuilder: StringBuilder = StringBuilder("")
            for (target: Variable in targets) {
                tupleBuilder.append(target.emit())
                tupleBuilder.append(", ")
            }
            doc.write("($tupleBuilder) = ${expr.emit()}")
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
     * Represents an if operation with an optional else case.
     */
    data class If(val cond: Expr, val ifFrame: Frame, val elseFrame: Frame? = null) : Op {
        override fun emit(doc: Doc) {
            doc.write("if ${cond.emit()}:")
            ifFrame.emit(doc.indent())
            if (elseFrame != null) {
                doc.write("else:")
                elseFrame.emit(doc.indent())
            }
        }
    }

    /**
     * Represents a return statement
     */
    data class ReturnStatement(val retVal: Variable?) : Op {
        override fun emit(doc: Doc) {
            if (retVal != null) {
                doc.write("return ${this.retVal.emit()}")
            } else {
                doc.write("return")
            }
        }
    }

    /**
     * Represents a while loop in python.
     */
    data class While(val cond: Expr, val body: Frame) : Op {
        override fun emit(doc: Doc) {
            doc.write("while ${cond.emit()}:")
            body.emit(doc.indent())
        }
    }

    /**
     * Represents a for loop in python.
     */
    data class For(val identifier: String, val collection: Expr, val body: List<Op>) : Op {
        constructor(identifier: String, collection: Expr, body: (Variable, MutableList<Op>) -> Unit) :
            this(identifier, collection, buildList<Op> { body(Variable(identifier), this) })

        override fun emit(doc: Doc) {
            doc.write("for $identifier in ${collection.emit()}:")
            doc.indent().also {
                body.forEach { stmt -> stmt.emit(it) }
            }
        }
    }

    /**
     * Represents a streaming pipeline. Used to provide a layer of
     * abstraction with the actual pipeline details.
     */
    data class StreamingPipeline(val frame: StreamingPipelineFrame) : Op {
        override fun emit(doc: Doc) {
            frame.emit(doc)
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

    class Function(val name: String, val args: List<Variable>, val body: Frame) : Op {
        override fun emit(doc: Doc) {
            val argList = args.joinToString(separator = ", ") { it.name }
            doc.write("def $name($argList):")
            body.emit(doc.indent())
        }
    }

    /**
     * Represents an array setitem call with the given index
     * and value
     *
     * @param inputExpr: The input array.
     * @param index: The index into the array.
     * @param value: The Expr being Set
     */
    class SetItem(private val inputExpr: Expr, private val index: Expr, private val value: Expr) : Op {
        override fun emit(doc: Doc) {
            val line = "${inputExpr.emit()}[${index.emit()}] = ${value.emit()}"
            doc.write(line)
        }
    }

    object Continue : Op {
        override fun emit(doc: Doc) {
            doc.write("continue")
        }
    }

    /**
     * Implementation of OP that does nothing including code generation.
     * This is not appropriate for a section where an OP logically belongs,
     * but can be used to represent an "optional" OP.
     */
    object NoOp : Op {
        override fun emit(doc: Doc) {}
    }
}
