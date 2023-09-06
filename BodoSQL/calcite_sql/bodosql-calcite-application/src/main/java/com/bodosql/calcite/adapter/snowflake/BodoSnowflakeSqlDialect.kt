package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.sql.SqlIntervalLiteral
import org.apache.calcite.sql.SqlIntervalLiteral.IntervalValue
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.dialect.SnowflakeSqlDialect
import org.apache.calcite.sql.type.AbstractSqlType
import org.apache.calcite.sql.type.BodoSqlTypeUtil
import org.apache.calcite.sql.type.SqlTypeName

class BodoSnowflakeSqlDialect(context: Context) : SnowflakeSqlDialect(context) {
    override fun unparseSqlIntervalLiteral(
        writer: SqlWriter,
        literal: SqlIntervalLiteral,
        leftPrec: Int,
        rightPrec: Int,
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
            writer,
            interval.intervalQualifier,
            RelDataTypeSystem.DEFAULT,
        )
        writer.print("'")
    }

    // This is directly copied from SqlDialect
    private val HEXITS = charArrayOf(
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    )

    /**
     * Converts a string into a unicode string literal that is interpretable by python.
     */
    override fun quoteStringLiteralUnicode(buf: StringBuilder, unicode_str: String) {
        buf.append("'")
        for (i in 0 until unicode_str.length) {
            val c = unicode_str[i]
            if (c.code < 32 || c.code >= 128) {
                buf.append('\\')
                buf.append('u')
                buf.append(HEXITS[c.code shr 12 and 0xf])
                buf.append(HEXITS[c.code shr 8 and 0xf])
                buf.append(HEXITS[c.code shr 4 and 0xf])
                buf.append(HEXITS[c.code and 0xf])
            } else if (c == '\'' || c == '\\') {
                buf.append(c)
                buf.append(c)
            } else {
                buf.append(c)
            }
        }
        buf.append("'")
    }

    /** Returns SqlNode for type in "cast(column as type)", which might be
     * different between databases by type name, precision etc.
     *
     *
     * If this method returns null, the cast will be omitted. In the default
     * implementation, this is the case for the NULL type, and therefore
     * `CAST(NULL AS <nulltype>)` is rendered as `NULL`.
     *
     * We implement this to ensure all casting goes through our own
     * BodoSqlTypeUtil.convertTypeToSpec casting rules. All other details
     * are copied from Calcite.
     * */
    override fun getCastSpec(type: RelDataType): SqlNode? {
        // Note: This implementation is borrowed from Calcite just
        // replacing SqlTypeUtil.convertTypeToSpec with
        // BodoSqlTypeUtil.convertTypeToSpec.
        var maxPrecision = -1
        var maxScale = -1
        if (type is AbstractSqlType) {
            when (type.getSqlTypeName()) {
                SqlTypeName.NULL -> return null
                SqlTypeName.DECIMAL -> {
                    maxScale = typeSystem.getMaxScale(type.getSqlTypeName())
                    // if needed, adjust varchar length to max length supported by the system
                    maxPrecision = typeSystem.getMaxPrecision(type.getSqlTypeName())
                }

                SqlTypeName.CHAR, SqlTypeName.VARCHAR ->
                    maxPrecision =
                        typeSystem.getMaxPrecision(type.getSqlTypeName())

                else -> {}
            }
            val charSet = if (supportsCharSet()) type.getCharset()?.name() else null
            return BodoSqlTypeUtil.convertTypeToSpec(type, charSet, maxPrecision, maxScale)
        }
        return BodoSqlTypeUtil.convertTypeToSpec(type)
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
