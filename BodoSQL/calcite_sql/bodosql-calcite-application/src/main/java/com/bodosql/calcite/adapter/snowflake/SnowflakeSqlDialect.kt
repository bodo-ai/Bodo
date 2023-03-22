package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.sql.SqlIntervalLiteral
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.dialect.SnowflakeSqlDialect

class SnowflakeSqlDialect(context: Context) : SnowflakeSqlDialect(context) {
    override fun unparseSqlIntervalLiteral(
        writer: SqlWriter,
        literal: SqlIntervalLiteral,
        leftPrec: Int,
        rightPrec: Int
    ) {
        // Upstream doesn't seem to implement this correctly so we will need to.
        // At the moment, I don't know yet how to implement this so leaving it set here
        // so it's easier to come back in the future.
        super.unparseSqlIntervalLiteral(writer, literal, leftPrec, rightPrec)
    }

    companion object {
        val DEFAULT_CONTEXT: Context = org.apache.calcite.sql.dialect.SnowflakeSqlDialect.DEFAULT_CONTEXT
            .withLiteralQuoteString("$$")
            .withLiteralEscapedQuoteString("\\$\\$")

        val DEFAULT = SnowflakeSqlDialect(DEFAULT_CONTEXT)
    }
}
