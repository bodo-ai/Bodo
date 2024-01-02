package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem
import org.apache.calcite.avatica.util.Casing
import org.apache.calcite.avatica.util.TimeUnit
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlIntervalLiteral
import org.apache.calcite.sql.SqlIntervalLiteral.IntervalValue
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.dialect.SnowflakeSqlDialect
import org.apache.calcite.sql.`fun`.SqlTrimFunction
import org.apache.calcite.sql.type.AbstractSqlType
import org.apache.calcite.sql.type.BodoSqlTypeUtil
import org.apache.calcite.sql.type.SqlTypeName
import java.math.BigDecimal

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
        if (interval.intervalQualifier.startUnit != TimeUnit.SECOND) {
            // Truncate the interval amount to an integer if the time unit isn't a second.
            // TODO(aneesh) we should see if it is commonly supported for other time units to also support fractional amounts.
            writer.literal(BigDecimal(interval.intervalLiteral).setScale(0).toString())
        } else {
            writer.literal(interval.intervalLiteral)
        }
        unparseSqlIntervalQualifier(
            writer,
            interval.intervalQualifier,
            BodoSQLRelDataTypeSystem(),
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
        var maxPrecision = RelDataType.PRECISION_NOT_SPECIFIED
        var maxScale = RelDataType.SCALE_NOT_SPECIFIED
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
            // Snowflake doesn't support character set
            val charSet = null
            return BodoSqlTypeUtil.convertTypeToSpec(type, charSet, maxPrecision, maxScale)
        }
        return BodoSqlTypeUtil.convertTypeToSpec(type)
    }

    /**
     * UnParses a function that has the typical syntax, IE:
     * FN_NAME(ARG0, ARG1, ARG2..)
     *
     * copied heavily from
     * org.apache.calcite.sql.SqlUtil.unparseFunctionSyntax
     */
    private fun genericFunctionUnParse(writer: SqlWriter, fnName: String, operandList: List<SqlNode>) {
        writer.print(fnName)
        val frame: SqlWriter.Frame = writer.startList(SqlWriter.FrameTypeEnum.FUN_CALL, "(", ")")

        for (operand in operandList) {
            writer.sep(",")
            operand.unparse(writer, 0, 0)
        }

        writer.endList(frame)
    }

    private fun unParseTrim(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        assert(call.operandCount() == 3) { "Trim has incorrect number of operands" }
        assert(call.operandList.get(0) is SqlLiteral && (call.operandList.get(0) as SqlLiteral).value is SqlTrimFunction.Flag) { "Trim operand 0 is not a flag" }
        val flag: SqlTrimFunction.Flag = (call.operandList.get(0) as SqlLiteral).value as SqlTrimFunction.Flag

        // Snowflake syntax is: "TRIM(baseExpr, charsToTrim)"
        // Calcite syntax is: "TRIM(BOTH charsToTrim FROM baseExpr)"
        val argsList = listOf(call.operandList.get(2), call.operandList.get(1))
        when (flag) {
            SqlTrimFunction.Flag.BOTH -> genericFunctionUnParse(writer, "TRIM", argsList)
            SqlTrimFunction.Flag.LEADING -> genericFunctionUnParse(writer, "LTRIM", argsList)
            SqlTrimFunction.Flag.TRAILING -> genericFunctionUnParse(writer, "RTRIM", argsList)
        }
    }

    private fun unParseSubstring(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        // Calcite syntax is SUBSTRING(`STRING_COLUMN` FROM 2 FOR 10)
        // Snowflake syntax is SUBSTRING(`STRING_COLUMN`, 2, 10)

        genericFunctionUnParse(writer, "SUBSTRING", call.operandList)
    }

    private fun unParseLenAlias(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        genericFunctionUnParse(writer, "LENGTH", call.operandList)
    }

    private fun unParseNanosecond(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        // Calcite doesn't have a NanoSecond function, so need to convert
        // back into DATE_PART.
        writer.print("DATE_PART")
        val frame: SqlWriter.Frame = writer.startList(SqlWriter.FrameTypeEnum.FUN_CALL, "(", ")")
        writer.print("NANOSECOND, ")
        call.operandList[0].unparse(writer, 0, 0)
        writer.endList(frame)
    }

    override fun unparseCall(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        when (call.kind) {
            SqlKind.TRIM -> unParseTrim(writer, call, leftPrec, rightPrec)
            SqlKind.OTHER, SqlKind.OTHER_FUNCTION -> {
                when (call.operator.name) {
                    "SUBSTR" -> unParseSubstring(writer, call, leftPrec, rightPrec)
                    "SUBSTRING" -> unParseSubstring(writer, call, leftPrec, rightPrec)
                    "CHAR_LENGTH" -> unParseLenAlias(writer, call, leftPrec, rightPrec)
                    "NANOSECOND" -> unParseNanosecond(writer, call, leftPrec, rightPrec)
                    else -> super.unparseCall(writer, call, leftPrec, rightPrec)
                }
            }
            else -> super.unparseCall(writer, call, leftPrec, rightPrec)
        }
    }

    companion object {
        @JvmField
        val DEFAULT_CONTEXT: Context = org.apache.calcite.sql.dialect.SnowflakeSqlDialect.DEFAULT_CONTEXT
            .withLiteralQuoteString("$$")
            .withLiteralEscapedQuoteString("\\$\\$")
            // TODO: Switch to True
            // https://bodo.atlassian.net/browse/BSE-2348
            .withCaseSensitive(false)
            .withQuotedCasing(Casing.UNCHANGED)
            .withUnquotedCasing(Casing.TO_UPPER)

        @JvmField
        val DEFAULT = BodoSnowflakeSqlDialect(DEFAULT_CONTEXT)

        // Default implementation to use before we have support for $ strings in the parser.
        // TODO: Add $ string suppport
        @JvmField
        val NO_DOLLAR_ESCAPE = BodoSnowflakeSqlDialect(org.apache.calcite.sql.dialect.SnowflakeSqlDialect.DEFAULT_CONTEXT)
    }
}
