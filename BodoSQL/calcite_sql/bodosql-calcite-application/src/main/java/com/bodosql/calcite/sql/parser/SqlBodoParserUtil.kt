package com.bodosql.calcite.sql.parser

import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlUtil
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.parser.SqlParserUtil
import org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE
import java.util.*

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
    }
}
