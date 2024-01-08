package com.bodosql.calcite.sql.parser

import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import org.apache.calcite.avatica.util.TimeUnit
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlIntervalQualifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlSelect
import org.apache.calcite.sql.SqlUtil
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.parser.SqlParserUtil
import org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE
import org.apache.calcite.util.Static.RESOURCE
import java.util.*
import kotlin.collections.ArrayList

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
        fun createBuiltinTableFunction(name: SqlIdentifier, pos: SqlParserPos, args: List<SqlNode>): SqlNode {
            if (name.simple == "FLATTEN") {
                return TableFunctionOperatorTable.FLATTEN.createCall(pos, args)
            } else {
                throw RuntimeException("Internal Error: Unexpected builtin table")
            }
        }

        /**
         * Convert a call to SPLIT_TO_TABLE to a call to FLATTEN(SPLIT(...)) with a pruning projection
         * that only selects the columns of FLATTEN that can be produced by SPLIT_TO_TABLE.
         */
        @JvmStatic
        fun createSplitToTable(start: SqlParserPos, end: SqlParserPos, args: List<SqlNode>): SqlNode? {
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
        fun createDatePartFunction(funcName: String, pos: SqlParserPos, intervalName: String, args: List<SqlNode>): SqlNode {
            // Normalize to uppercase
            // Convert the name to the correct function call.
            when (intervalName.uppercase(Locale.ROOT)) {
                "YEAR", "YEARS", "Y", "YY", "YYY", "YYYY", "YR", "YRS" -> return SqlStdOperatorTable.YEAR.createCall(pos, args)
                "MONTH", "MONTHS", "MM", "MON", "MONS" -> return SqlStdOperatorTable.MONTH.createCall(pos, args)
                "DAY", "DAYS", "D", "DD", "DAYOFMONTH" -> return SqlStdOperatorTable.DAYOFMONTH.createCall(pos, args)
                "DAYOFWEEK", "WEEKDAY", "DOW", "DW" -> return SqlStdOperatorTable.DAYOFWEEK.createCall(pos, args)
                "DAYOFWEEKISO", "WEEKDAY_ISO", "DOW_ISO", "DW_ISO" -> return DatetimeOperatorTable.DAYOFWEEKISO.createCall(pos, args)
                "DAYOFYEAR", "YEARDAY", "DOY", "DY" -> return SqlStdOperatorTable.DAYOFYEAR.createCall(pos, args)
                "WEEK", "WEEKS", "W", "WK", "WEEKOFYEAR", "WOY", "WY" -> return SqlStdOperatorTable.WEEK.createCall(pos, args)
                "WEEKISO", "WEEK_ISO", "WEEKOFYEARISO", "WEEKOFYEAR_ISO" -> return DatetimeOperatorTable.WEEKISO.createCall(pos, args)
                "QUARTER", "Q", "QTR", "QTRS", "QUARTERS" -> return SqlStdOperatorTable.QUARTER.createCall(pos, args)
                "YEAROFWEEK" -> return DatetimeOperatorTable.YEAROFWEEK.createCall(pos, args)
                "YEAROFWEEKISO" -> return DatetimeOperatorTable.YEAROFWEEKISO.createCall(pos, args)
                "HOUR", "H", "HH", "HR", "HOURS", "HRS" -> return SqlStdOperatorTable.HOUR.createCall(pos, args)
                "MINUTE", "M", "MI", "MIN", "MINUTES", "MINS" -> return SqlStdOperatorTable.MINUTE.createCall(pos, args)
                "SECOND", "S", "SEC", "SECONDS", "SECS" -> return SqlStdOperatorTable.SECOND.createCall(pos, args)
                "NANOSECOND", "NS", "NSEC", "NANOSEC", "NSECOND", "NANOSECONDS", "NANOSECS", "NSECONDS" -> return DatetimeOperatorTable.NANOSECOND.createCall(pos, args)
                "EPOCH_SECOND", "EPOCH", "EPOCH_SECONDS" -> return DatetimeOperatorTable.EPOCH_SECOND.createCall(pos, args)
                "EPOCH_MILLISECOND", "EPOCH_MILLISECONDS" -> return DatetimeOperatorTable.EPOCH_MILLISECOND.createCall(pos, args)
                "EPOCH_MICROSECOND", "EPOCH_MICROSECONDS" -> return DatetimeOperatorTable.EPOCH_MICROSECOND.createCall(pos, args)
                "EPOCH_NANOSECOND", "EPOCH_NANOSECONDS" -> return DatetimeOperatorTable.EPOCH_NANOSECOND.createCall(pos, args)
                "TIMEZONE_HOUR", "TZH" -> return DatetimeOperatorTable.TIMEZONE_HOUR.createCall(pos, args)
                "TIMEZONE_MINUTE", "TZM" -> return DatetimeOperatorTable.TIMEZONE_MINUTE.createCall(pos, args)
                else -> throw SqlUtil.newContextException(
                    pos,
                    BODO_SQL_RESOURCE.illegalDatePartTimeUnit(funcName, intervalName),
                )
            }
        }

        @JvmStatic
        fun createLastDayFunction(pos: SqlParserPos, intervalName: String, args: List<SqlNode>): SqlNode {
            when (intervalName.uppercase(Locale.ROOT)) {
                "YEAR", "YEARS", "Y", "YY", "YYY", "YYYY", "YR", "YRS",
                "MONTH", "MONTHS", "MM", "MON", "MONS",
                "WEEK", "WEEKS", "W", "WK", "WEEKOFYEAR", "WOY", "WY",
                "QUARTER", "Q", "QTR", "QTRS", "QUARTERS",
                -> {
                    val timeArg = SqlLiteral.createCharString(intervalName, pos)
                    val new_args = args.plus(timeArg)
                    return SqlBodoOperatorTable.LAST_DAY.createCall(pos, new_args)
                }
                else -> throw SqlUtil.newContextException(
                    pos,
                    BODO_SQL_RESOURCE.illegalLastDayTimeUnit(intervalName),
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
                val intervalParts = trimmedStr.split("\\s+".toRegex()).dropLastWhile { it.isEmpty() }
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
                    unit = when (timeUnitStr) {
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
            for (i in 1..(intervalStrings.size - 1)) {
                val interval = SqlLiteral.createInterval(sign, intervalStrings[i], intervalQualifiers[i], pos)
                // we use COMBINE_INTERVALS instead of operator+ because adding day/time intervals to year/month intervals is not allowed.
                res = SqlBodoOperatorTable.COMBINE_INTERVALS.createCall(pos, listOf(res, interval))
            }
            return res
        }
    }
}
