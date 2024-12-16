package com.bodosql.calcite.ir

/**
 * This class abstracts away the process of writing to a document.
 * It's primarily used to keep track of indentation levels,
 * but it may be extended for more functionality as more is needed
 * from the document writer.
 *
 * @param code The string builder to write code to.
 * @param indent The indent string to use for each indentation level.
 * @param indent Indentation level for this section of the document.
 */
class Doc private constructor(
    private val code: StringBuilder,
    private val indent: String,
    private val level: Int,
) {
    /**
     * Constructs a new document for code writing.
     * @param indent Indentation string.
     * @param level Starting indentation level.
     */
    constructor(indent: String = "  ", level: Int = 0) :
        this(code = StringBuilder(), indent = indent, level = level)

    /**
     * Write a single line to the document.
     * @param line Line to write to the document.
     */
    fun write(line: String) {
        val prefix = indent.repeat(level)

        // Lines shouldn't really have any newlines, but
        // we're going to be extra safe and just handle this
        // correctly in case those slip through.
        // This just checks for the newline, splits it,
        // maps the prefix onto each line, then joins
        // them back together.
        val block =
            if (line.contains("\n")) {
                line
                    .split("\n")
                    .joinToString { "${prefix}${it.trim()}" }
            } else {
                "${prefix}$line"
            }
        code.append(block)
        code.append("\n")
    }

    /**
     * Indent the document one level.
     * @return New document with the same parameters at
     *         an extra level of indentation.
     */
    fun indent(): Doc = Doc(code = this.code, indent = this.indent, level = level + 1)

    /**
     * Returns the code that was written to this document.
     * @return Code written to the document.
     */
    override fun toString(): String = code.toString()
}
