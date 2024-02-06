package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem
import org.apache.calcite.avatica.util.Casing
import org.apache.calcite.avatica.util.TimeUnit
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlAbstractDateTimeLiteral
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlIntervalLiteral
import org.apache.calcite.sql.SqlIntervalLiteral.IntervalValue
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.dialect.SnowflakeSqlDialect
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.`fun`.SqlTrimFunction
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.type.AbstractSqlType
import org.apache.calcite.sql.type.BodoSqlTypeUtil
import org.apache.calcite.sql.type.SqlTypeName
import java.math.BigDecimal
import java.util.*
import kotlin.collections.ArrayList

class BodoSnowflakeSqlDialect(context: Context) : SnowflakeSqlDialect(context) {

    /**
     * Helper method for outputting just the duration part of an Interval literal
     */
    private fun unparseSqlIntervalLiteralInner(
        writer: SqlWriter,
        literal: SqlIntervalLiteral,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        val interval = literal.getValueAs(IntervalValue::class.java)
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
    }
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
        unparseSqlIntervalLiteralInner(writer, literal, leftPrec, rightPrec)
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

    private fun unParseCombineIntervals(writer: SqlWriter, call: SqlCall, leftPrec: Int, rightPrec: Int) {
        // COMBINE_INTERVAL calls are inserted by the parser as an abstraction to deal with INTERVAL literals with commas (multiple values),
        // which isn't well-supported by calcite. However, interval addition isn't allowed in snowflake for all interval types,
        // so we need to unparse COMBINE_INTERVAL calls back into their original form.
        // We know that user code should never create COMBINE_INTERVALS calls, so we can make assumptions about the arguments.
        // The format of COMBINE_INTERVALS calls are:
        // INTERVAL 'interval0, interval1, interval2' -> COMBINE_INTERVALS(COMBINE_INTERVALS(interval0, interval1), interval2)
        writer.print("INTERVAL ")
        val arg1 = call.operandList[1] as SqlIntervalLiteral
        if (arg1.signum() == -1) {
            writer.print("-")
        }
        // Snowflake requires both the interval
        // literal and the interval qualifier be
        // wrapped in quotes.
        writer.print("'")

        // Collect all the intervals in reverse order
        val intervals = ArrayList(listOf(arg1))
        var node = call.operandList[0]
        while (node is SqlCall) {
            intervals.add(node.operandList[1] as SqlIntervalLiteral)
            node = node.operandList[0]
        }
        intervals.add(node as SqlIntervalLiteral)

        // Write out all the intervals in the correct order and add commas between each
        unparseSqlIntervalLiteralInner(writer, intervals[intervals.size - 1], leftPrec, rightPrec)
        for (i in (intervals.size - 2) downTo 0) {
            writer.print(", ")
            unparseSqlIntervalLiteralInner(writer, intervals[i], leftPrec, rightPrec)
        }
        writer.print("'")
    }

    private fun unParseIsBoolean(writer: SqlWriter, call: SqlCall, leftPrec: Int, rightPrec: Int) {
        val targetValue = when (call.kind) {
            SqlKind.IS_FALSE, SqlKind.IS_NOT_FALSE -> false
            else -> true
        }

        val isInequality = when (call.kind) {
            SqlKind.IS_NOT_FALSE, SqlKind.IS_NOT_TRUE -> true
            else -> false
        }

        val newOp = if (isInequality) {
            SqlStdOperatorTable.IS_DISTINCT_FROM
        } else {
            SqlStdOperatorTable.IS_NOT_DISTINCT_FROM
        }

        val compValue = SqlLiteral.createBoolean(targetValue, SqlParserPos.ZERO)
        // The alternate syntax: newOp.createCall(call.parserPosition, listOf(call.operand(0), compValue))
        // raises a strange error for certain SQL files (ex RadialWash):
        // java.lang.ArrayStoreException: org.apache.calcite.sql.SqlIdentifier
        // I'm not sure why this is, there's a followup issue to investigate:
        // https://bodo.atlassian.net/browse/BSE-2608
        val newCall = newOp.createCall(call.parserPosition, call.operand(0), compValue)
        newCall.unparse(writer, leftPrec, rightPrec)
    }

    override fun unparseCall(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        when (call.kind) {
            SqlKind.IS_FALSE, SqlKind.IS_NOT_FALSE, SqlKind.IS_TRUE, SqlKind.IS_NOT_TRUE -> unParseIsBoolean(writer, call, leftPrec, rightPrec)
            SqlKind.TRIM -> unParseTrim(writer, call, leftPrec, rightPrec)
            SqlKind.OTHER, SqlKind.OTHER_FUNCTION -> {
                when (call.operator.name) {
                    "SUBSTR" -> unParseSubstring(writer, call, leftPrec, rightPrec)
                    "SUBSTRING" -> unParseSubstring(writer, call, leftPrec, rightPrec)
                    "CHAR_LENGTH" -> unParseLenAlias(writer, call, leftPrec, rightPrec)
                    "NANOSECOND" -> unParseNanosecond(writer, call, leftPrec, rightPrec)
                    "COMBINE_INTERVALS" -> unParseCombineIntervals(writer, call, leftPrec, rightPrec)
                    else -> super.unparseCall(writer, call, leftPrec, rightPrec)
                }
            }
            else -> super.unparseCall(writer, call, leftPrec, rightPrec)
        }
    }

    override fun unparseDateTimeLiteral(
        writer: SqlWriter,
        literal: SqlAbstractDateTimeLiteral,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        if (literal.typeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
            // Snowflake doesn't recognize TIMESTAMP_WITH_LOCAL_TIME_ZONE as a way to create,
            // a literal, so we need to remap it to a TO_TIMESTAMP_LTZ call.
            writer.print("TO_TIMESTAMP_LTZ")
            val frame: SqlWriter.Frame = writer.startList(SqlWriter.FrameTypeEnum.FUN_CALL, "(", ")")
            writer.print(String.format(Locale.ROOT, "'%s'", literal.toFormattedString()))
            writer.endList(frame)
        } else {
            super.unparseDateTimeLiteral(writer, literal, leftPrec, rightPrec)
        }
    }

    companion object {
        @JvmField
        val DEFAULT_CONTEXT: Context = SnowflakeSqlDialect.DEFAULT_CONTEXT
            .withLiteralQuoteString("$$")
            .withLiteralEscapedQuoteString("\\$\\$")
            .withCaseSensitive(true)
            .withQuotedCasing(Casing.UNCHANGED)
            .withUnquotedCasing(Casing.TO_UPPER)

        @JvmField
        val DEFAULT = BodoSnowflakeSqlDialect(DEFAULT_CONTEXT)
    }
}
