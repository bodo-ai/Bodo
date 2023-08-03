package com.bodosql.calcite.sql.parser

import org.apache.calcite.sql.parser.SqlParserUtil

class SqlBodoParserUtil {
    companion object {
        // https://docs.snowflake.com/en/sql-reference/data-types-text#escape-sequences-in-single-quoted-string-constants
        // Triple quoted strings in Kotlin are raw strings.
        // The escape sequence in the regex is for the regex and not Java/Kotlin.
        private val ESCAPE_SEQUENCES = """\\(['"\\bfnrt])""".toRegex()

        /**
         * Parses a string with the additional work of resolving escape
         * sequences from Snowflake.
         *
         * This parsing method does not correctly resolve the escape
         * sequences for ASCII NUL, octal escape sequences, or hex escape
         * sequences, or unicode escape sequences.
         */
        @JvmStatic
        fun parseString(s: String): String =
            SqlParserUtil.parseString(s)
                .replace(ESCAPE_SEQUENCES) { res ->
                    when (val ch = res.groupValues[1]) {
                        "b" -> "\b"
                        // unicode escape for formfeed. Kotlin doesn't
                        // allow \f natively.
                        "f" -> "\u000c"
                        "n" -> "\n"
                        "r" -> "\r"
                        "t" -> "\t"
                        else -> ch
                    }
                }

    }
}
