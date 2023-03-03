package com.bodosql.calcite.ir

/**
 * Module is the top level compilation unit for code generation.
 * @param main The main function for this module.
 */
class Module(private val main: List<Op>) {

    /**
     * Emits the code for the module.
     * @return Emitted code for the full module.
     */
    fun emit(level: Int = 0): String {
        val doc = Doc(level = level)
        for (item in main) {
            item.emit(doc)
        }
        return doc.toString()
    }

    /**
     * Builder is used to construct a new module.
     */
    class Builder(val symbolTable: SymbolTable) {
        constructor() : this(symbolTable = SymbolTable())

        private val code: MutableList<Op> = mutableListOf()

        /**
         * Adds the operation to the end of the module.
         * @param op Operation to add to the module.
         */
        fun add(op: Op) {
            code.add(op)
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
        fun append(code: String): Builder {
            // If the last operation is an Op.Code,
            // append directly to that.
            val op = this.code.lastOrNull()
            if (op != null && op is Op.Code) {
                op.append(code)
                return this
            }

            // Otherwise, create a new Op.Code.
            this.code.add(Op.Code(code))
            return this
        }

        fun append(code: StringBuilder): Builder {
            return append(code.toString())
        }

        /**
         * Construct a module from the built code.
         * @return The built module.
         */
        fun build(): Module {
            return Module(code.toList())
        }
    }
}
