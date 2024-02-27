package com.bodosql.calcite.sql.parser

import com.bodosql.calcite.application.operatorTables.DatetimeFnUtils.DateTimePart
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import org.apache.calcite.avatica.util.TimeUnit
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlIntervalQualifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlSelect
import org.apache.calcite.sql.SqlUtil
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.parser.SqlParserUtil
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE
import org.apache.calcite.util.Static.RESOURCE

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

        /**
         * Returns a call to a builtin Table Function.
         */
        @JvmStatic
        fun createBuiltinTableFunction(
            name: SqlIdentifier,
            pos: SqlParserPos,
            args: List<SqlNode>,
        ): SqlNode {
            if (name.simple == "FLATTEN") {
                // In Snowflake FLATTEN supports calling functions using a mix of
                // position and named arguments, which is not generally supported in SQL.
                // Calcite has the backed in assumption that it must be either all positional
                // or all named (which Snowflake docs support for UDFs), so we add support
                // for this mix by inserting the parameter names for any initial arguments
                // preceding the first named argument.
                val argNames = listOf("INPUT", "PATH", "OUTER", "RECURSIVE", "MODE")
                val updatedArgs =
                    if (args.any { x -> x.kind == SqlKind.ARGUMENT_ASSIGNMENT } && args.size <= argNames.size) {
                        var seenNamed = false
                        args.mapIndexed { i, arg ->
                            if (!seenNamed && arg.kind != SqlKind.ARGUMENT_ASSIGNMENT) {
                                SqlStdOperatorTable.ARGUMENT_ASSIGNMENT.createCall(
                                    arg.parserPosition,
                                    arg,
                                    SqlIdentifier(argNames[i], arg.parserPosition),
                                )
                            } else {
                                seenNamed = true
                                arg
                            }
                        }
                    } else {
                        args
                    }
                return TableFunctionOperatorTable.FLATTEN.createCall(pos, updatedArgs)
            } else {
                throw RuntimeException("Internal Error: Unexpected builtin table")
            }
        }

        /**
         * Convert a call to SPLIT_TO_TABLE to a call to FLATTEN(SPLIT(...)) with a pruning projection
         * that only selects the columns of FLATTEN that can be newArgs by SPLIT_TO_TABLE.
         */
        @JvmStatic
        fun createSplitToTable(
            start: SqlParserPos,
            end: SqlParserPos,
            args: List<SqlNode>,
        ): SqlNode? {
            val splitCall: SqlNode = StringOperatorTable.SPLIT.createCall(end, args)
            val flattenCall = TableFunctionOperatorTable.FLATTEN.createCall(end, listOf(splitCall))
            val tableCall = SqlStdOperatorTable.COLLECTION_TABLE.createCall(end, flattenCall)
            val prunedCols = SqlNodeList(listOf(SqlIdentifier("SEQ", end), SqlIdentifier("INDEX", end), SqlIdentifier("VALUE", end)), end)
            val pruningSelect = SqlSelect(end, null, prunedCols, tableCall, null, null, null, null, null, null, null, null, null)
            return pruningSelect
        }

        /**
         * Dispatch a DATE_PART or Extract call to the appropriate individual function.
         */
        @JvmStatic
        fun createDatePartFunction(
            funcName: String,
            pos: SqlParserPos,
            intervalName: SqlLiteral,
            args: List<SqlNode>,
        ): SqlNode {
            // Convert the enum to the correct function call.
            assert(intervalName.typeName == SqlTypeName.SYMBOL) { "Internal Error in createDatePartFunction: intervalName is not a symbol" }
            assert(intervalName.value is DateTimePart) { "Internal Error in createDatePartFunction: intervalName is not a symbol" }
            when (val intervalEnum: DateTimePart = intervalName.value as DateTimePart) {
                DateTimePart.YEAR -> return SqlStdOperatorTable.YEAR.createCall(pos, args)
                DateTimePart.MONTH -> return SqlStdOperatorTable.MONTH.createCall(pos, args)
                DateTimePart.DAY -> return SqlStdOperatorTable.DAYOFMONTH.createCall(pos, args)
                DateTimePart.DAYOFWEEK -> return SqlStdOperatorTable.DAYOFWEEK.createCall(pos, args)
                DateTimePart.DAYOFWEEKISO -> return DatetimeOperatorTable.DAYOFWEEKISO.createCall(pos, args)
                DateTimePart.DAYOFYEAR -> return SqlStdOperatorTable.DAYOFYEAR.createCall(pos, args)
                DateTimePart.WEEK -> return SqlStdOperatorTable.WEEK.createCall(pos, args)
                DateTimePart.WEEKISO -> return DatetimeOperatorTable.WEEKISO.createCall(pos, args)
                DateTimePart.QUARTER -> return SqlStdOperatorTable.QUARTER.createCall(pos, args)
                DateTimePart.YEAROFWEEK -> return DatetimeOperatorTable.YEAROFWEEK.createCall(pos, args)
                DateTimePart.YEAROFWEEKISO -> return DatetimeOperatorTable.YEAROFWEEKISO.createCall(pos, args)
                DateTimePart.HOUR -> return SqlStdOperatorTable.HOUR.createCall(pos, args)
                DateTimePart.MINUTE -> return SqlStdOperatorTable.MINUTE.createCall(pos, args)
                DateTimePart.SECOND -> return SqlStdOperatorTable.SECOND.createCall(pos, args)
                DateTimePart.NANOSECOND -> return DatetimeOperatorTable.NANOSECOND.createCall(pos, args)
                DateTimePart.EPOCH_SECOND -> return DatetimeOperatorTable.EPOCH_SECOND.createCall(pos, args)
                DateTimePart.EPOCH_MILLISECOND -> return DatetimeOperatorTable.EPOCH_MILLISECOND.createCall(pos, args)
                DateTimePart.EPOCH_MICROSECOND -> return DatetimeOperatorTable.EPOCH_MICROSECOND.createCall(pos, args)
                DateTimePart.EPOCH_NANOSECOND -> return DatetimeOperatorTable.EPOCH_NANOSECOND.createCall(pos, args)
                DateTimePart.TIMEZONE_HOUR -> return DatetimeOperatorTable.TIMEZONE_HOUR.createCall(pos, args)
                DateTimePart.TIMEZONE_MINUTE -> return DatetimeOperatorTable.TIMEZONE_MINUTE.createCall(pos, args)
                else -> throw SqlUtil.newContextException(
                    pos,
                    BODO_SQL_RESOURCE.illegalDatePartTimeUnit(funcName, intervalEnum.name),
                )
            }
        }

        @JvmStatic
        fun createLastDayFunction(
            pos: SqlParserPos,
            intervalName: SqlLiteral,
            args: List<SqlNode>,
        ): SqlNode {
            assert(intervalName.typeName == SqlTypeName.SYMBOL) { "Internal Error in createDatePartFunction: intervalName is not a symbol" }
            assert(intervalName.value is DateTimePart) { "Internal Error in createDatePartFunction: intervalName is not a symbol" }
            when (val intervalEnum: DateTimePart = intervalName.value as DateTimePart) {
                DateTimePart.YEAR,
                DateTimePart.MONTH,
                DateTimePart.WEEK,
                DateTimePart.QUARTER,
                -> {
                    val newArgs = args.plus(intervalName)
                    return SqlBodoOperatorTable.LAST_DAY.createCall(pos, newArgs)
                }
                else -> throw SqlUtil.newContextException(
                    pos,
                    BODO_SQL_RESOURCE.illegalDatePartTimeUnit("LAST_DAY", intervalEnum.name),
                )
            }
        }

        /**
         * Parses an interval literal that is accepted by Snowflake:
         * https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#interval-constants
         *
         * The parser currently removes the quotes.
         *
         * TODO: Add proper comma support
         */
        @JvmStatic
        fun parseSnowflakeIntervalLiteral(
            pos: SqlParserPos,
            sign: Int,
            s: String,
        ): SqlNode {
            // Collect all the interval strings/qualifiers found in the string
            val intervalStrings: MutableList<String> = ArrayList()
            val intervalQualifiers: MutableList<SqlIntervalQualifier> = ArrayList()
            // Parse the Snowflake interval literal string. This is a comma separated
            // series of <integer> [ <date_time_part> ]. If any date_time_part is omitted this
            // defaults to seconds.
            val splitStrings = s.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
            for (intervalInfo in splitStrings) {
                val trimmedStr = intervalInfo.trim { it <= ' ' }
                val intervalParts =
                    trimmedStr.split("\\s+".toRegex()).dropLastWhile { it.isEmpty() }
                        .toTypedArray()
                if (intervalParts.size == 1 || intervalParts.size == 2) {
                    val intervalAmountStr: String
                    val timeUnitStr: String
                    if (intervalParts.size == 1) {
                        // If we have only 1 part then it may not be space separated (rare but possible).
                        // For example: Interval '30d'.
                        // If we so we look for the first non-numeric character and split on that.
                        var endIdx = -1
                        val baseStr = intervalParts[0]
                        for (i in 0 until baseStr.length) {
                            if (!Character.isDigit(baseStr[i])) {
                                endIdx = i
                                break
                            }
                        }
                        if (endIdx == -1) {
                            // If we only have 1 part the default Interval is seconds.
                            intervalAmountStr = baseStr
                            timeUnitStr = "second"
                        } else {
                            intervalAmountStr = baseStr.substring(0, endIdx)
                            timeUnitStr = baseStr.substring(endIdx).lowercase()
                        }
                    } else {
                        intervalAmountStr = intervalParts[0]
                        timeUnitStr = intervalParts[1].lowercase()
                    }
                    intervalStrings.add(intervalAmountStr)
                    // Parse the second string into the valid time units.
                    // Here we support the time units supported by the other interval syntax only.
                    // TODO: Support all interval values supported by Snowflake in both interval paths.
                    val unit: TimeUnit
                    unit =
                        when (timeUnitStr) {
                            "year", "years", "y", "yy", "yyy", "yyyy", "yr", "yrs" -> TimeUnit.YEAR
                            "quarter", "quarters", "qtr", "qtrs", "q" -> TimeUnit.QUARTER
                            "month", "months", "mm", "mon", "mons" -> TimeUnit.MONTH
                            "week", "weeks", "w", "wk", "weekofyear", "woy", "wy" -> TimeUnit.WEEK
                            "day", "days", "d", "dd", "dayofmonth" -> TimeUnit.DAY
                            "hour", "hours", "h", "hh", "hr", "hrs" -> TimeUnit.HOUR
                            "minute", "minutes", "m", "mi", "min", "mins" -> TimeUnit.MINUTE
                            "second", "seconds", "s", "sec", "secs" -> TimeUnit.SECOND
                            "millisecond", "milliseconds", "ms", "msec" -> TimeUnit.MILLISECOND
                            "microsecond", "microseconds", "us", "usec" -> TimeUnit.MICROSECOND
                            "nanosecond", "nanoseconds", "ns", "nsec", "nanosec", "nanosecs", "nsecond", "nseconds" -> TimeUnit.NANOSECOND
                            else -> throw SqlUtil.newContextException(
                                pos,
                                RESOURCE.illegalIntervalLiteral(s, pos.toString()),
                            )
                        }
                    intervalQualifiers.add(SqlIntervalQualifier(unit, null, pos))
                } else {
                    throw SqlUtil.newContextException(
                        pos,
                        RESOURCE.illegalIntervalLiteral(s, pos.toString()),
                    )
                }
            }
            if (intervalStrings.size == 0) {
                throw SqlUtil.newContextException(
                    pos,
                    RESOURCE.illegalIntervalLiteral(s, pos.toString()),
                )
            }

            var res: SqlNode = SqlLiteral.createInterval(sign, intervalStrings[0], intervalQualifiers[0], pos)
            for (i in 1 until intervalStrings.size) {
                val interval = SqlLiteral.createInterval(sign, intervalStrings[i], intervalQualifiers[i], pos)
                // we use COMBINE_INTERVALS instead of operator+ because adding day/time intervals to year/month intervals is not allowed.
                res = SqlBodoOperatorTable.COMBINE_INTERVALS.createCall(pos, listOf(res, interval))
            }
            return res
        }
    }
}
