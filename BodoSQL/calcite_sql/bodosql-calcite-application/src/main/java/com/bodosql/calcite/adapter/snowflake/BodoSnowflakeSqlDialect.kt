package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.sql.SqlIntervalLiteral
import org.apache.calcite.sql.SqlIntervalLiteral.IntervalValue
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.dialect.SnowflakeSqlDialect

class BodoSnowflakeSqlDialect(context: Context) : SnowflakeSqlDialect(context) {
    override fun unparseSqlIntervalLiteral(
        writer: SqlWriter,
        literal: SqlIntervalLiteral,
        leftPrec: Int,
        rightPrec: Int
    ) {
        // SnowflakeSqlDialect isn't implemented properly in upstream. We
        // reimplement it here.
        val interval = literal.getValueAs(IntervalValue::class.java)
        writer.keyword("INTERVAL")
        if (interval.sign == -1) {
            writer.print("-")
        }
        // Snowflake requires both the interval
        // literal and the interval qualifier be
        // wrapped in quotes.
        writer.print("'")
        writer.literal(interval.intervalLiteral)
        unparseSqlIntervalQualifier(
            writer, interval.intervalQualifier,
            RelDataTypeSystem.DEFAULT
        )
        writer.print("'")
    }

    companion object {
        @JvmField
        val DEFAULT_CONTEXT: Context = org.apache.calcite.sql.dialect.SnowflakeSqlDialect.DEFAULT_CONTEXT
            .withLiteralQuoteString("$$")
            .withLiteralEscapedQuoteString("\\$\\$")

        @JvmField
        val DEFAULT = BodoSnowflakeSqlDialect(DEFAULT_CONTEXT)
    }
}
