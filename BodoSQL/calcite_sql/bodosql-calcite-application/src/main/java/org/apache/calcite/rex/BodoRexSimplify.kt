package org.apache.calcite.rex

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeFnUtils
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.rex.JsonPecUtil
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.avatica.util.ByteString
import org.apache.calcite.avatica.util.TimeUnitRange
import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlAggFunction
import org.apache.calcite.sql.SqlBinaryOperator
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.BasicSqlType
import org.apache.calcite.sql.type.BodoTZInfo
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.SqlTypeUtil.isTimestamp
import org.apache.calcite.util.Bug
import org.apache.calcite.util.DateString
import org.apache.calcite.util.TimeString
import org.apache.calcite.util.TimestampString
import org.apache.calcite.util.TimestampWithTimeZoneString
import java.lang.IllegalArgumentException
import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode
import java.util.*
import java.util.regex.Pattern
import kotlin.math.absoluteValue
import kotlin.math.sign

/**
 * Bodo's implementation of RexSimplifier
 * with extended support for Bodo specific functions. This must be placed
 * in the same package, so we can override package private methods.
 */
class BodoRexSimplify(
    rexBuilder: RexBuilder,
    predicates: RelOptPredicateList,
    defaultUnknownAs: RexUnknownAs,
    predicateElimination: Boolean,
    paranoid: Boolean,
    executor: RexExecutor,
) : RexSimplify(rexBuilder, predicates, defaultUnknownAs, predicateElimination, paranoid, executor) {

    constructor(rexBuilder: RexBuilder, predicates: RelOptPredicateList, executor: RexExecutor) : this(rexBuilder, predicates, RexUnknownAs.UNKNOWN, true, false, executor)

    /** Returns a RexSimplify the same as this but with a specified
     * [.predicates] value.  */
    override fun withPredicates(predicates: RelOptPredicateList): RexSimplify {
        return if (predicates === this.predicates) {
            this
        } else {
            BodoRexSimplify(
                rexBuilder,
                predicates,
                defaultUnknownAs,
                predicateElimination,
                paranoid,
                executor,
            )
        }
    }

    /** Returns a RexSimplify the same as this but which verifies that
     * the expression before and after simplification are equivalent.
     *
     * @see .verify
     */
    override fun withParanoid(paranoid: Boolean): RexSimplify {
        return if (paranoid == this.paranoid) {
            this
        } else {
            BodoRexSimplify(
                rexBuilder,
                predicates,
                defaultUnknownAs,
                predicateElimination,
                paranoid,
                executor,
            )
        }
    }

    /** Returns a RexSimplify the same as this but with a specified
     * [.predicateElimination] value.
     *
     *
     * This is introduced temporarily, until
     * [[CALCITE-2401] is fixed][Bug.CALCITE_2401_FIXED].
     */
    override fun withPredicateElimination(predicateElimination: Boolean): RexSimplify {
        return if (predicateElimination == this.predicateElimination) {
            this
        } else {
            BodoRexSimplify(
                rexBuilder,
                predicates,
                defaultUnknownAs,
                predicateElimination,
                paranoid,
                executor,
            )
        }
    }

    /**
     * Converts a Snowflake string literal that represents an offset from the
     * start of UNIX Time and returns that value normalized to milliseconds.
     *
     * A precondition for this function is that the entire string is an
     * integer literal.
     *
     * Note Snowflake interprets the epoch value differently depending
     * on the value of the integer, which we duplicate here:
     * https://docs.snowflake.com/en/sql-reference/functions/to_timestamp#usage-notes
     */
    private fun integerStringLiteralToEpoch(literalString: String): BigDecimal {
        val offset = BigDecimal(literalString)
        return if (offset < BigDecimal(SECOND_EPOCH_THRESHOLD)) {
            // Value is in seconds.
            offset * BigDecimal(NANOSEC_PER_SECOND)
        } else if (offset < BigDecimal(MILLISECOND_EPOCH_THRESHOLD)) {
            // Value is in milliseconds
            offset * BigDecimal(NANOSEC_PER_MILLISECOND)
        } else if (offset < BigDecimal(MICROSECOND_EPOCH_THRESHOLD)) {
            // Value is in microseconds
            offset * BigDecimal(NANOSEC_PER_MICROSECOND)
        } else {
            // Value is in nanoseconds
            offset
        }
    }

    /**
     * Convert epoch in nanoseconds to the corresponding date string.
     */
    private fun epochToDateString(epochNanoseconds: BigDecimal): DateString {
        // Only keep the date portion
        val dateOffset = epochNanoseconds.divideToIntegralValue(BigDecimal(NANOSEC_PER_DAY)).toInt()
        return DateString.fromDaysSinceEpoch(dateOffset)
    }

    /**
     * Convert epoch in nanoseconds to the corresponding time string.
     */
    private fun epochToTimeString(epochNanoseconds: BigDecimal): TimeString {
        // Only keep the time portion
        val timeOffset = epochNanoseconds.remainder(BigDecimal(NANOSEC_PER_DAY))
        val nanos = timeOffset.remainder(BigDecimal(NANOSEC_PER_SECOND)).toInt()
        // We can now move from decimal to integer
        val secondHigher = timeOffset.divideToIntegralValue(BigDecimal(NANOSEC_PER_SECOND)).toInt()
        val second = secondHigher.mod(SECOND_PER_MINUTE)
        val minute = secondHigher.div(SECOND_PER_MINUTE).mod(MINUTE_PER_HOUR)
        val hour = secondHigher.div(SECOND_PER_HOUR)
        return TimeString(hour, minute, second).withNanos(nanos)
    }

    private fun stringLiteralToDateString(literalString: String): Pair<DateString?, String> {
        // Just support the main Snowflake patterns without whitespace. Note that Snowflake
        // appends leading 0s if they are absent
        val pattern1 = Pattern.compile("^(\\d{1,4})-(\\d{1,2})-(\\d{1,2})")
        val matcher1 = pattern1.matcher(literalString)
        return if (matcher1.find()) {
            val year = matcher1.group(1)
            val month = matcher1.group(2)
            val day = matcher1.group(3)
            val remainingString = literalString.substring(matcher1.end())
            Pair(DateString(Integer.valueOf(year), Integer.valueOf(month), Integer.valueOf(day)), remainingString)
        } else {
            val pattern2 = Pattern.compile("^(\\d{1,2})/(\\d{1,2})/(\\d{1,4})")
            val matcher2 = pattern2.matcher(literalString)
            if (matcher2.find()) {
                val month = matcher2.group(1)
                val day = matcher2.group(2)
                val year = matcher2.group(3)
                val remainingString = literalString.substring(matcher2.end())
                Pair(DateString(Integer.valueOf(year), Integer.valueOf(month), Integer.valueOf(day)), remainingString)
            } else {
                val pattern3 = Pattern.compile("^(\\d{1,2})-(\\p{L}{3})-(\\d{1,4})$")
                val matcher3 = pattern3.matcher(literalString)
                if (matcher3.find()) {
                    val day = matcher3.group(1)
                    val monthName = matcher3.group(2).uppercase()
                    val year = matcher3.group(3)
                    // Month is 1-indexed
                    val month = monthCodeList.indexOf(monthName) + 1
                    val remainingString = literalString.substring(matcher3.end())
                    // This format cannot be used at the start of a timestamp string
                    if (remainingString.isEmpty()) {
                        Pair(
                            DateString(Integer.valueOf(year), Integer.valueOf(month), Integer.valueOf(day)),
                            remainingString,
                        )
                    } else {
                        Pair(null, literalString)
                    }
                } else {
                    Pair(null, literalString)
                }
            }
        }
    }

    private fun stringLiteralToDate(call: RexCall, literal: RexLiteral): RexNode {
        val literalString = literal.value2.toString()
        // Snowflake accepts timestamp inputs for date casts
        return if (literalString.all { char -> char.isDigit() }) {
            val nanoseconds = integerStringLiteralToEpoch(literalString)
            rexBuilder.makeDateLiteral(epochToDateString(nanoseconds))
        } else {
            val decomposed = stringLiteralToTimestampString(literalString, true, false)
            if (decomposed == null) {
                call
            } else {
                // Extract the date component
                val (tsString, _) = decomposed
                val dateComponent = tsString.toString().split(" ")[0]
                val dateString = DateString(dateComponent)
                rexBuilder.makeDateLiteral(dateString)
            }
        }
    }

    /**
     * Converts a date literal to a TIMESTAMP literal. This supports both
     * TIMESTAMP_NTZ, TIMESTAMP_LTZ and TIMESTAMP_TZ.
     */
    private fun dateLiteralToTimestamp(literal: RexLiteral, precision: Int, isTzAware: Boolean, isOffset: Boolean): RexNode {
        val calendar = literal.getValueAs(Calendar::class.java)!!
        val tsString = TimestampString(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH) + 1, calendar.get(Calendar.DAY_OF_MONTH), 0, 0, 0)
        return if (isTzAware) {
            rexBuilder.makeTimestampWithLocalTimeZoneLiteral(tsString, precision)
        } else if (isOffset) {
            val defaultZone = TimeZone.getTimeZone(BodoTZInfo.getDefaultTZInfo(this.rexBuilder.typeFactory.typeSystem).zone)
            val zoneOffset = getOffsetOfTimestamp(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH) + 1, calendar.get(Calendar.DAY_OF_MONTH), 0, 0, defaultZone)
            rexBuilder.makeTimestampTzLiteral(TimestampWithTimeZoneString(tsString, zoneOffset), precision)
        } else {
            rexBuilder.makeTimestampLiteral(tsString, precision)
        }
    }

    /**
     * Converts a TIMESTAMP_LTZ or TIMESTAMP_NTZ literal to a date literal.
     */
    private fun timestampLiteralToDate(literal: RexLiteral): RexNode {
        val string = literal.getValueAs(TimestampString::class.java)!!.toString()
        val dateString = DateString(string.split(" ")[0])
        return rexBuilder.makeDateLiteral(dateString)
    }

    /**
     * Converts a timestamp literal to a date literal. This supports both
     * TIMESTAMP_LTZ and TIMESTAMP_NTZ.
     */
    private fun timestampLiteralToTime(literal: RexLiteral, precision: Int): RexNode {
        val tsString = literal.getValueAs(TimestampString::class.java)
        return if (tsString == null) {
            // This code path should never be reachable.
            literal
        } else {
            // Convert to a string and extract the time value.
            val strValue = tsString.toString()
            val parts = strValue.split(" ")
            if (parts.size != 2) {
                // This code path should never be reachable
                literal
            } else {
                val timeStr = TimeString(parts[1])
                rexBuilder.makeTimeLiteral(timeStr, precision)
            }
        }
    }

    /**
     * Simplify ::date and to_date functions for supported literals.
     */
    private fun simplifyDateCast(call: RexCall, operand: RexNode): RexNode {
        return if (operand is RexLiteral) {
            // Resolve constant casts for literals
            if (operand.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                timestampLiteralToDate(operand)
            } else {
                when (operand.typeName) {
                    SqlTypeName.DATE -> operand
                    SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                    SqlTypeName.VARCHAR, SqlTypeName.CHAR -> stringLiteralToDate(call, operand)
                    SqlTypeName.TIMESTAMP -> timestampLiteralToDate(operand)
                    else -> call
                }
            }
        } else if (operand is RexCall) {
            val innerCall = call.operands[0] as RexCall
            if (innerCall.operator.name == DatetimeOperatorTable.GETDATE.name) {
                // Replace GETDATE::DATE with CURRENT_DATE
                rexBuilder.makeCall(SqlStdOperatorTable.CURRENT_DATE)
            } else {
                call
            }
        } else {
            call
        }
    }

    /**
     * Simplify a Numeric cast via the infix cast ::.
     * Right now we only add functionality where we omit intermediate casts that
     * are always safe.
     *
     * This is motivated by a gap in our Snowflake typing where we can type NUMBER(38,0)
     * as BIGINT, so when inlining a view we will generate a cast of X::DECIMAL(38,0)::BIGINT
     * where X has an original type of BIGINT.
     */
    private fun simplifyIntegerCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexCall && (operand.kind == SqlKind.CAST || operand.kind == SqlKind.SAFE_CAST)) {
            val outerCastType = call.type
            val innerCastType = operand.type
            val innerMostNode = operand.operands[0]
            val innerMostNodeType = innerMostNode.type
            val allNumeric = SqlTypeFamily.EXACT_NUMERIC.contains(outerCastType) && SqlTypeFamily.EXACT_NUMERIC.contains(innerCastType) && SqlTypeFamily.EXACT_NUMERIC.contains(innerMostNodeType)
            val allInteger = outerCastType.scale == 0 && innerCastType.scale == 0 && innerMostNodeType.scale == 0
            val innerIsUpcast = innerCastType.precision >= innerMostNodeType.precision
            // We can remove at least 1 cast if it's all integers and there is an upcast for the first cast.
            if (allNumeric && allInteger && innerIsUpcast) {
                // Determine if we can return the innermost node or we need a new cast
                rexBuilder.makeCast(outerCastType, innerMostNode)
            } else {
                call
            }
        } else if (operand is RexLiteral) {
            // If the cast is determined to be invalid, return NULL for TRY_ casts, and the original
            // call for non-TRY_ casts (since they need to raise an error at runtime).
            val nullInteger = rexBuilder.makeNullLiteral(call.type)
            // TODO: Add a bit width check as a non-error case. We don't want
            // to confuse the TINYINT definitions in SF and hit errors.
            // TODO: Separate Decimal Literals from Double literals in the plan.
            // TODO: Be more conservative with the error code path to avoid false positives
            // on try cast.
            val errorValue = if (isTryCast) { nullInteger } else { call }
            when (operand.type.sqlTypeName) {
                SqlTypeName.TINYINT,
                SqlTypeName.SMALLINT,
                SqlTypeName.INTEGER,
                SqlTypeName.BIGINT,
                SqlTypeName.DECIMAL,
                // Note FLOAT and DOUBLE are internally stored as BigDecimal in Calcite
                SqlTypeName.DOUBLE,
                SqlTypeName.FLOAT,
                -> {
                    val asDecimal = operand.getValueAs(BigDecimal::class.java)
                    if (asDecimal == null) {
                        errorValue
                    } else {
                        val storedValue = asDecimal.setScale(0, RoundingMode.HALF_UP).toBigInteger()
                        val isSafe = when (call.type.sqlTypeName) {
                            SqlTypeName.TINYINT -> (storedValue >= BigInteger.valueOf(Byte.MIN_VALUE.toLong())) && (
                                storedValue <= BigInteger.valueOf(
                                    Byte.MAX_VALUE.toLong(),
                                )
                                )

                            SqlTypeName.SMALLINT -> (storedValue >= BigInteger.valueOf(Short.MIN_VALUE.toLong())) && (
                                storedValue <= BigInteger.valueOf(
                                    Short.MAX_VALUE.toLong(),
                                )
                                )

                            SqlTypeName.INTEGER -> (storedValue >= BigInteger.valueOf(Int.MIN_VALUE.toLong())) && (
                                storedValue <= BigInteger.valueOf(
                                    Int.MAX_VALUE.toLong(),
                                )
                                )

                            SqlTypeName.BIGINT -> (storedValue >= BigInteger.valueOf(Long.MIN_VALUE)) && (
                                storedValue <= BigInteger.valueOf(
                                    Long.MAX_VALUE,
                                )
                                )

                            else -> true
                        }
                        if (isSafe) {
                            rexBuilder.makeLiteral(storedValue.longValueExact(), call.type)
                        } else {
                            call
                        }
                    }
                }
                SqlTypeName.CHAR,
                SqlTypeName.VARCHAR,
                -> {
                    val asString = operand.getValueAs(String::class.java)?.uppercase()?.trim()
                    val asDecimal = if (asString.equals("E") || asString.equals("-E")) {
                        BigDecimal(0)
                    } else {
                        asString?.toBigDecimalOrNull()
                    }
                    if (asDecimal == null) {
                        errorValue
                    } else {
                        val storedValue = asDecimal.setScale(0, RoundingMode.HALF_UP).toBigInteger()
                        val isSafe = when (call.type.sqlTypeName) {
                            SqlTypeName.TINYINT -> (storedValue >= BigInteger.valueOf(Byte.MIN_VALUE.toLong())) && (storedValue <= BigInteger.valueOf(Byte.MAX_VALUE.toLong()))
                            SqlTypeName.SMALLINT -> (storedValue >= BigInteger.valueOf(Short.MIN_VALUE.toLong())) && (storedValue <= BigInteger.valueOf(Short.MAX_VALUE.toLong()))
                            SqlTypeName.INTEGER -> (storedValue >= BigInteger.valueOf(Int.MIN_VALUE.toLong())) && (storedValue <= BigInteger.valueOf(Int.MAX_VALUE.toLong()))
                            SqlTypeName.BIGINT -> (storedValue >= BigInteger.valueOf(Long.MIN_VALUE)) && (storedValue <= BigInteger.valueOf(Long.MAX_VALUE))
                            else -> true
                        }
                        if (isSafe) {
                            rexBuilder.makeLiteral(storedValue.longValueExact(), call.type)
                        } else {
                            call
                        }
                    }
                }
                SqlTypeName.BOOLEAN -> {
                    val isTrue = operand.isAlwaysTrue
                    val isFalse = operand.isAlwaysFalse
                    if (isTrue) {
                        rexBuilder.makeLiteral(1, call.getType())
                    } else if (isFalse) {
                        rexBuilder.makeLiteral(0, call.getType())
                    } else {
                        // This shouldn't be reachable.
                        call
                    }
                }
                SqlTypeName.NULL -> nullInteger
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Simplifies a cast to a decimal type.
     *
     * @param call The original cast call.
     * @param operand The argument being casted.
     * @param isTryCast If true, returns null on invalid inputs.
     */
    private fun simplifyDecimalCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexLiteral) {
            // If the cast is determined to be invalid, return NULL for TRY_ casts, and the original
            // call for non-TRY_ casts (since they need to raise an error at runtime).
            val nullDecimal = rexBuilder.makeNullLiteral(call.type)
            val errorValue = if (isTryCast) { nullDecimal } else { call }
            val precision = call.getType().precision
            val scale = call.getType().scale
            // Compute bounds for the decimal value
            val numericSection = BigDecimal(10, MathContext.UNLIMITED).pow((precision - scale))
            // Subtract the smallest positive value
            val smallestPositive = BigDecimal(BigInteger.ONE, scale)
            val domain = numericSection.subtract(smallestPositive)
            // TODO: Be more conservative with the error code paths.
            when (operand.type.sqlTypeName) {
                SqlTypeName.TINYINT,
                SqlTypeName.SMALLINT,
                SqlTypeName.INTEGER,
                SqlTypeName.BIGINT,
                SqlTypeName.DECIMAL,
                -> {
                    val asDecimal = operand.getValueAs(BigDecimal::class.java)
                    if (asDecimal == null) { errorValue } else {
                        if (asDecimal <= -domain || asDecimal >= domain) {
                            errorValue
                        } else {
                            rexBuilder.makeLiteral(asDecimal.setScale(scale, RoundingMode.HALF_UP), call.type)
                        }
                    }
                }
                SqlTypeName.DOUBLE,
                SqlTypeName.FLOAT,
                -> {
                    // Float values round so they don't need to fit.
                    val asDecimal = operand.getValueAs(BigDecimal::class.java)
                    if (asDecimal == null) {
                        errorValue
                    } else {
                        rexBuilder.makeLiteral(asDecimal.setScale(scale, RoundingMode.HALF_UP), call.type)
                    }
                }
                SqlTypeName.CHAR,
                SqlTypeName.VARCHAR,
                -> {
                    val asString = operand.getValueAs(String::class.java)?.uppercase()?.trim()
                    val asDecimal = if (asString.equals("E") || asString.equals("-E")) {
                        BigDecimal(0)
                    } else {
                        asString?.toBigDecimalOrNull()
                    }
                    if (asDecimal == null) {
                        errorValue
                    } else {
                        if (asDecimal <= -domain || asDecimal >= domain) {
                            errorValue
                        } else {
                            rexBuilder.makeLiteral(asDecimal.setScale(scale, RoundingMode.HALF_UP), call.type)
                        }
                    }
                }
                SqlTypeName.BOOLEAN -> {
                    val isTrue = operand.isAlwaysTrue
                    val isFalse = operand.isAlwaysFalse
                    if (isTrue) {
                        if (scale == precision) {
                            // If scale == precision 1 cannot be represented.
                            // This is allowed in SF but clearly a bug as it has value 1.
                            call
                        } else {
                            rexBuilder.makeLiteral(BigDecimal.ONE, call.getType())
                        }
                    } else if (isFalse) {
                        if (scale == precision) {
                            // If scale == precision calcite won't allow making
                            // the literal unless we make the precision 0.
                            rexBuilder.makeLiteral(BigDecimal.ZERO.setScale(1), call.getType())
                        } else {
                            rexBuilder.makeLiteral(BigDecimal.ZERO, call.getType())
                        }
                    } else {
                        // This shouldn't be reachable.
                        call
                    }
                }
                SqlTypeName.NULL -> nullDecimal
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Simplifies a cast to a double type.
     *
     * @param call The original cast call.
     * @param operand The argument being casted.
     * @param isTryCast If true, returns null on invalid inputs.
     */
    private fun simplifyDoubleCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexLiteral) {
            // If the cast is determined to be invalid, return NULL for TRY_ casts, and the original
            // call for non-TRY_ casts (since they need to raise an error at runtime).
            val nullDouble = rexBuilder.makeNullLiteral(call.type)
            val errorValue = if (isTryCast) { nullDouble } else { call }
            when (operand.type.sqlTypeName) {
                SqlTypeName.TINYINT,
                SqlTypeName.SMALLINT,
                SqlTypeName.INTEGER,
                SqlTypeName.BIGINT,
                SqlTypeName.DECIMAL,
                // Note: Double and float internally are represented as BigDecimal inside Calcite, so
                // we use the same code path as decimal/integer.
                SqlTypeName.DOUBLE,
                SqlTypeName.FLOAT,
                -> {
                    val asDecimal = operand.getValueAs(BigDecimal::class.java)
                    if (asDecimal == null) { errorValue } else { rexBuilder.makeLiteral(asDecimal, call.type) }
                }
                SqlTypeName.CHAR,
                SqlTypeName.VARCHAR,
                -> {
                    val asString = operand.getValueAs(String::class.java)?.uppercase()?.trim()
                    val asDecimal = asString?.toBigDecimalOrNull()
                    if (asDecimal == null) {
                        // Manually check for NAN or strings containing INF.
                        // Don't simplify these because they aren't invalid.
                        // Note: We are conservative in case whitespace matters.
                        if (asString.equals("NAN") || asString.equals("INF") || asString.equals("-INF")) {
                            call
                        } else {
                            errorValue
                        }
                    } else {
                        rexBuilder.makeLiteral(asDecimal, call.type)
                    }
                }
                SqlTypeName.BOOLEAN -> {
                    val isTrue = operand.isAlwaysTrue
                    val isFalse = operand.isAlwaysFalse
                    if (isTrue) {
                        rexBuilder.makeLiteral(1.0, call.getType())
                    } else if (isFalse) {
                        rexBuilder.makeLiteral(0.0, call.getType())
                    } else {
                        // This shouldn't be reachable.
                        call
                    }
                }
                SqlTypeName.NULL -> nullDouble
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Parse the time component from a Snowflake string. This returns null for the time string
     * if there is no time component as this is shared by both Time and Timestamp support.
     *
     * This also returns any leftover string components for further validation
     */
    private fun parseTimeComponents(literalString: String, isTimestamp: Boolean): Pair<TimeString?, String> {
        val prefix = if (isTimestamp) {
            "[ T]"
        } else {
            ""
        }
        // This supports HH24:MI:SS.FF and HH24:MI:SS ISO time formats. Note Snowflake
        // supports when leading 0s are missing.
        val timeStringPattern1 = Pattern.compile("^$prefix(\\d{1,2}):(\\d{1,2}):(\\d{1,2})")
        val timeStringMatcher1 = timeStringPattern1.matcher(literalString)
        var remainingString = literalString
        return if (timeStringMatcher1.find()) {
            val hour = Integer.valueOf(timeStringMatcher1.group(1))
            val minute = Integer.valueOf(timeStringMatcher1.group(2))
            val second = Integer.valueOf(timeStringMatcher1.group(3))
            var subsecond = ""
            remainingString = remainingString.substring(timeStringMatcher1.end())

            // The pattern for the sub-second components of a timestamp string.
            val subsecondStringPattern = Pattern.compile("^\\.(\\d{1,9})")
            val subsecondStringMatcher = subsecondStringPattern.matcher(remainingString)
            if (subsecondStringMatcher.find()) {
                subsecond = subsecondStringMatcher.group(1)
                remainingString = remainingString.substring(subsecondStringMatcher.end())
            }
            var timeStr = TimeString(hour, minute, second)
            if (subsecond.isNotEmpty()) {
                timeStr = timeStr.withFraction(subsecond)
            }
            Pair(timeStr, remainingString)
        } else {
            // We also support HH24:MI. This requires no fractional component
            val timeStringPattern2 = Pattern.compile("^$prefix(\\d{1,2}):(\\d{1,2})")
            val timeStringMatcher2 = timeStringPattern2.matcher(literalString)
            if (timeStringMatcher2.find()) {
                val hour = Integer.valueOf(timeStringMatcher2.group(1))
                val minute = Integer.valueOf(timeStringMatcher2.group(2))
                val second = 0
                val suffix = remainingString.substring(timeStringMatcher2.end())
                // We cannot have anything after the time string
                if (suffix.isNotEmpty()) {
                    Pair(null, remainingString)
                } else {
                    var timeStr = TimeString(hour, minute, second)
                    Pair(timeStr, suffix)
                }
            } else if (isTimestamp) {
                // Only Timestamp support just hours. There must be no fractional component.
                val timeStringPattern3 = Pattern.compile("^$prefix(\\d{1,2})")
                val timeStringMatcher3 = timeStringPattern3.matcher(literalString)
                if (timeStringMatcher3.find()) {
                    val hour = Integer.valueOf(timeStringMatcher3.group(1))
                    val minute = 0
                    val second = 0
                    val suffix = remainingString.substring(timeStringMatcher3.end())
                    // We cannot have anything after the time string
                    if (suffix.isNotEmpty()) {
                        Pair(null, remainingString)
                    } else {
                        var timeStr = TimeString(hour, minute, second)
                        Pair(timeStr, suffix)
                    }
                } else {
                    Pair(null, remainingString)
                }
            } else {
                Pair(null, remainingString)
            }
        }
    }

    /**
     * Convert a String literal that represents a datetime value to a TimestampString
     */
    private fun stringLiteralToTimestampString(literalString: String, isNaive: Boolean, isOffset: Boolean): Pair<TimestampString, TimeZone>? {
        // Convert a SQL literal being cast to a timestamp with the default Snowflake
        // parsing to a timestamp literal. If the parsed date doesn't match our supported
        // formats then we return the original cast call.
        val (dateString, nonDateString) = stringLiteralToDateString(literalString)
        if (dateString == null) {
            return null
        }
        val parts = dateString.toString().split("-")
        val year = Integer.valueOf(parts[0])
        val month = Integer.valueOf(parts[1])
        val day = Integer.valueOf(parts[2])

        var (timeString, remainingString) = parseTimeComponents(nonDateString, true)
        val (timeParts, subSecond) = if (timeString == null) {
            Pair(Triple(0, 0, 0), "")
        } else {
            // Parse the time string.
            val strValue = timeString.toString()
            val clockParts = strValue.split(".")
            val clockTime = clockParts[0]
            val subClockTime = if (clockParts.size == 2) {
                clockParts[1]
            } else {
                ""
            }
            val units = clockTime.split(":")
            val integerUnits = Triple(Integer.valueOf(units[0]), Integer.valueOf(units[1]), Integer.valueOf(units[2]))
            Pair(integerUnits, subClockTime)
        }
        val (hour, minute, second) = timeParts
        var timeZone = TimeZone.getTimeZone(BodoTZInfo.getDefaultTZInfo(this.rexBuilder.typeFactory.typeSystem).zone)
        // Identify the UTC offset component of the string
        val offsetPattern = Pattern.compile("^[ ]{0,1}[+-]((\\d{1,2}[:]\\d{1,2})|(\\d{4}))$")
        val offsetMatcher = offsetPattern.matcher(remainingString)
        if (offsetMatcher.find()) {
            val start = offsetMatcher.start()
            val stop = offsetMatcher.end()
            timeZone = TimeZone.getTimeZone("GMT" + remainingString.substring(start, stop).trim(' '))
            remainingString = remainingString.substring(stop)
        }
        // Verify that the remainder of the string is either empty or "+00:00"
        return if (remainingString.isEmpty()) {
            val yearInt = Integer.valueOf(year)
            val monthInt = Integer.valueOf(month)
            val dayInt = Integer.valueOf(day)
            val hourInt = Integer.valueOf(hour)
            val minuteInt = Integer.valueOf(minute)
            val secondInt = Integer.valueOf(second)
            var tsString = TimestampString(
                yearInt,
                monthInt,
                dayInt,
                hourInt,
                minuteInt,
                secondInt,
            )
            if (subSecond.isNotEmpty()) {
                tsString = tsString.withFraction(subSecond)
            }
            val timeZoneAsOffset = getOffsetOfTimestamp(yearInt, monthInt, dayInt, hourInt, minuteInt, timeZone)
            Pair(tsString, timeZoneAsOffset)
        } else {
            null
        }
    }

    // Takes in a TimeZone and a specific day and returns another time zone object referencing the fixed offset of
    // that time zone in that time of year.
    private fun getOffsetOfTimestamp(year: Int, month: Int, day: Int, hour: Int, minute: Int, zone: TimeZone): TimeZone {
        // Get the raw offset in milliseconds from UTC of the current timezone in the specified time of year
        val hourMinuteAsMs = hour * 3_600_000 + minute * 60_000
        val msOffset = zone.getOffset(1, year, month - 1, day, 1, hourMinuteAsMs)
        // Convert the raw offset to the number of hours and minutes ahead/behind UTC
        val signChar = if (msOffset.sign >= 0) { "+" } else { "-" }
        val hourOffset = msOffset.absoluteValue / 3_600_000
        val minuteOffset = (msOffset.absoluteValue % 3_600_000) / 60_000
        // Convert the signed hours/minutes into another timezone using the offset instead of a named zone
        val minuteOffsetLengthTwo = minuteOffset.toString().padStart(2, '0')
        return TimeZone.getTimeZone("GMT${signChar}${hourOffset}$minuteOffsetLengthTwo")
    }

    /**
     * Convert a SQL literal being cast to a timestamp with the default Snowflake
     * parsing to a timestamp literal. If the parsed date doesn't match our supported
     * formats then we return the original cast.
     *
     * Note: This handles TIMESTAMP_NTZ, TIMESTAMP_LTZ and TIMESTAMP_TZ.
     *
     * @param call The original call to cast the string to a timestamp
     * @param literal The string literal being casted
     * @param isNaive If true, returns a TIMESTAMP_NTZ
     * @param isOffset If true, returns a TIMESTAMP_TZ
     * @return Either the simplified timestamp literal, or the original call
     */
    private fun stringLiteralToTimestamp(call: RexCall, literal: RexLiteral, isNaive: Boolean, isOffset: Boolean): RexNode {
        val literalString = literal.value2.toString()
        return if (literalString.all { char -> char.isDigit() }) {
            val nanoseconds = integerStringLiteralToEpoch(literalString)
            // Compute the date and timestamp string and convert it to a Timestamp
            val dateString = epochToDateString(nanoseconds)
            val timeString = epochToTimeString(nanoseconds)
            // Combine the string to build the timestamp string
            val stringValue = "$dateString $timeString"
            val tsString = TimestampString(stringValue)
            if (isNaive) {
                rexBuilder.makeTimestampLiteral(tsString, call.getType().precision)
            } else if (isOffset) {
                val tsTzString = TimestampWithTimeZoneString(tsString, TimeZone.getTimeZone("GMT+00:00"))
                rexBuilder.makeTimestampTzLiteral(tsTzString, call.getType().precision)
            } else {
                rexBuilder.makeTimestampWithLocalTimeZoneLiteral(tsString, call.getType().precision)
            }
        } else {
            val decomposed = stringLiteralToTimestampString(literalString, isNaive, isOffset)
            return if (decomposed == null) {
                call
            } else {
                val (tsString, timeZone) = decomposed
                if (isNaive) {
                    rexBuilder.makeTimestampLiteral(tsString, call.getType().precision)
                } else if (isOffset) {
                    val tsTzString = TimestampWithTimeZoneString(tsString, timeZone)
                    rexBuilder.makeTimestampTzLiteral(tsTzString, call.getType().precision)
                } else {
                    rexBuilder.makeTimestampWithLocalTimeZoneLiteral(tsString, call.getType().precision)
                }
            }
        }
    }

    /**
     * Convert a SQL literal being cast to a time with the default Snowflake
     * parsing to a time literal. If the parsed date doesn't match our supported
     * formats then we return the original cast.
     *
     * Note: This handles both TIMESTAMP_NTZ and TIMESTAMP_LTZ
     *
     * @param call The original call to cast the string to a timestamp
     * @param literal The string literal being casted
     * @return Either the simplified time literal, or the original call
     */
    private fun stringLiteralToTime(call: RexCall, literal: RexLiteral): RexNode {
        // Convert a SQL literal being cast to a timestamp with the default Snowflake
        // parsing to a timestamp literal. If the parsed date doesn't match our supported
        // formats then we return the original cast call.
        var literalString = literal.value2.toString()
        return if (literalString.all { char -> char.isDigit() }) {
            val nanoseconds = integerStringLiteralToEpoch(literalString)
            rexBuilder.makeTimeLiteral(epochToTimeString(nanoseconds), call.type.precision)
        } else {
            val (timeStr, remaining) = parseTimeComponents(literalString, false)
            if (timeStr == null || remaining.isNotEmpty()) {
                call
            } else {
                rexBuilder.makeTimeLiteral(timeStr, call.type.precision)
            }
        }
    }

    /**
     * Simplify TIMESTAMP_NTZ cast for literals.
     */
    private fun simplifyTimestampNtzCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexLiteral) {
            val nullValue = rexBuilder.makeNullLiteral(call.type)
            val errorValue = if (isTryCast) nullValue else { call }
            if (operand.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                val timestampString = operand.getValueAs(TimestampString::class.java)
                if (timestampString == null) {
                    errorValue
                } else {
                    rexBuilder.makeTimestampLiteral(timestampString, call.type.precision)
                }
            } else {
                when (operand.typeName) {
                    SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                    SqlTypeName.VARCHAR,
                    SqlTypeName.CHAR,
                    -> stringLiteralToTimestamp(call, operand, true, false)

                    SqlTypeName.DATE -> dateLiteralToTimestamp(operand, call.getType().precision, false, false)
                    SqlTypeName.TIMESTAMP -> {
                        // Support change of precision
                        val timestampString = operand.getValueAs(TimestampString::class.java)
                        if (timestampString == null) {
                            errorValue
                        } else {
                            rexBuilder.makeTimestampLiteral(timestampString, call.type.precision)
                        }
                    }
                    else -> call
                }
            }
        } else {
            call
        }
    }

    private fun timestampToTimestampTz(call: RexCall, tsString: TimestampString, zone: TimeZone, isNaive: Boolean): RexNode {
        val calendar = tsString.toCalendar()
        val zoneOffset = getOffsetOfTimestamp(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH) + 1, calendar.get(Calendar.DAY_OF_MONTH), calendar.get(Calendar.HOUR), calendar.get(Calendar.MINUTE), zone)
        val tsStringWithTz = TimestampWithTimeZoneString(tsString, zoneOffset)
        return rexBuilder.makeTimestampTzLiteral(tsStringWithTz, call.type.precision)
    }

    /**
     * Simplify TIMESTAMP_TZ cast for literals.
     */
    private fun simplifyTimestampTzCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexLiteral) {
            val nullValue = rexBuilder.makeNullLiteral(call.type)
            val errorValue = if (isTryCast) nullValue else { call }
            if (operand.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                val timestampString = operand.getValueAs(TimestampString::class.java)
                if (timestampString == null) {
                    errorValue
                } else {
                    val defaultZone = TimeZone.getTimeZone(BodoTZInfo.getDefaultTZInfo(this.rexBuilder.typeFactory.typeSystem).zone)
                    timestampToTimestampTz(call, timestampString, defaultZone, false)
                }
            } else {
                when (operand.typeName) {
                    SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                    SqlTypeName.VARCHAR,
                    SqlTypeName.CHAR,
                    -> {
                        stringLiteralToTimestamp(call, operand, false, true)
                    }

                    SqlTypeName.DATE -> {
                        dateLiteralToTimestamp(operand, call.getType().precision, false, true)
                    }
                    SqlTypeName.TIMESTAMP -> {
                        // Support change of precision
                        val timestampString = operand.getValueAs(TimestampString::class.java)
                        if (timestampString == null) {
                            errorValue
                        } else {
                            val defaultZone = TimeZone.getTimeZone(BodoTZInfo.getDefaultTZInfo(this.rexBuilder.typeFactory.typeSystem).zone)
                            timestampToTimestampTz(call, timestampString, defaultZone, true)
                        }
                    }
                    else -> call
                }
            }
        } else {
            call
        }
    }

    /**
     * Simplify TIMESTAMP_LTZ cast for literals.
     */
    private fun simplifyTimestampLtzCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        return if (operand is RexLiteral) {
            val nullValue = rexBuilder.makeNullLiteral(call.type)
            val errorValue = if (isTryCast) nullValue else { call }
            if (operand.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                // Support change of precision
                val timestampString = operand.getValueAs(TimestampString::class.java)
                if (timestampString == null) {
                    errorValue
                } else {
                    rexBuilder.makeTimestampWithLocalTimeZoneLiteral(timestampString, call.type.precision)
                }
            } else {
                when (operand.typeName) {
                    SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                    SqlTypeName.VARCHAR,
                    SqlTypeName.CHAR,
                    -> stringLiteralToTimestamp(call, operand, false, false)

                    SqlTypeName.DATE -> dateLiteralToTimestamp(operand, call.getType().precision, true, false)
                    SqlTypeName.TIMESTAMP -> {
                        val timestampString = operand.getValueAs(TimestampString::class.java)
                        if (timestampString == null) {
                            errorValue
                        } else {
                            rexBuilder.makeTimestampWithLocalTimeZoneLiteral(timestampString, call.type.precision)
                        }
                    }

                    else -> call
                }
            }
        } else {
            call
        }
    }

    /**
     * Simplify Varchar casts that are equivalent in our type system. These
     * are scalar casts to varchar or varchar casts that only change precision.
     */
    private fun simplifyVarcharCast(call: RexCall, operand: RexNode, unknownAs: RexUnknownAs): RexNode {
        val operandTypeName = operand.type.sqlTypeName
        return if (operand is RexLiteral) {
            when (operandTypeName) {
                SqlTypeName.CHAR, SqlTypeName.VARCHAR -> rexBuilder.makeLiteral(operand.getValueAs(String::class.java), call.getType())
                SqlTypeName.DATE -> rexBuilder.makeLiteral(
                    operand.getValueAs(DateString::class.java)!!.toString(),
                    call.getType(),
                )

                SqlTypeName.TINYINT, SqlTypeName.SMALLINT, SqlTypeName.INTEGER, SqlTypeName.BIGINT, SqlTypeName.DECIMAL -> {
                    val literalValue = operand.getValueAs(BigDecimal::class.java)!!
                    val newLiteral = literalValue.toPlainString()
                    rexBuilder.makeLiteral(newLiteral, call.getType())
                }

                SqlTypeName.BOOLEAN -> {
                    val isTrue = operand.isAlwaysTrue
                    val isFalse = operand.isAlwaysFalse
                    if (isTrue) {
                        rexBuilder.makeLiteral("true", call.getType())
                    } else if (isFalse) {
                        rexBuilder.makeLiteral("false", call.getType())
                    } else {
                        // This shouldn't be reachable.
                        call
                    }
                }
                SqlTypeName.BINARY, SqlTypeName.VARBINARY -> {
                    val byteString = operand.getValueAs(ByteString::class.java)!!
                    // Snowflake treats A-F as uppercase
                    rexBuilder.makeLiteral(byteString.toString().uppercase(), call.getType())
                }
                SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                else -> call
            }
        } else if (operandTypeName == SqlTypeName.VARCHAR || operandTypeName == SqlTypeName.CHAR) {
            // Ignore precision for intermediate casts
            simplify(operand, unknownAs)
        } else {
            call
        }
    }

    /**
     * Simplify Varbinary casts that are equivalent in our type system. These
     * are scalar casts to Varbinary/Binary or literal simplification.
     */
    private fun simplifyVarbinaryCast(call: RexCall, operand: RexNode, unknownAs: RexUnknownAs, isTryCast: Boolean): RexNode {
        val nullValue = rexBuilder.makeNullLiteral(call.type)
        val errorValue = if (isTryCast) nullValue else { call }
        val operandTypeName = operand.type.sqlTypeName
        // Ignore precision on casts (TODO: Do we need to fix precision in our type system for binary/varbinary)
        return if (operand is RexLiteral) {
            when (operandTypeName) {
                SqlTypeName.CHAR, SqlTypeName.VARCHAR -> {
                    val asString = operand.getValueAs(String::class.java)
                    if (asString == null) {
                        errorValue
                    } else {
                        // Cast to Binary converts to hex.
                        try {
                            val bytesString = ByteString.of(asString, 16)
                            rexBuilder.makeLiteral(bytesString, call.type)
                        } catch (e: IllegalArgumentException) {
                            // hex parsing is straightforward, so if we have an illicit cast convert
                            // to the error value.
                            errorValue
                        }
                    }
                }
                SqlTypeName.BINARY, SqlTypeName.VARBINARY -> rexBuilder.makeLiteral(operand.getValueAs(ByteString::class.java), call.type)
                SqlTypeName.NULL -> nullValue
                else -> call
            }
        } else if (operandTypeName == SqlTypeName.VARBINARY || operandTypeName == SqlTypeName.BINARY) {
            // Ignore precision for intermediate casts
            simplify(operand, unknownAs)
        } else {
            call
        }
    }

    /**
     * Simplifies a cast to a boolean type.
     *
     * @param call The original cast call.
     * @param operand The argument being casted.
     * @param isTryCast If true, returns null on invalid inputs.
     */
    private fun simplifyBooleanCast(call: RexCall, operand: RexNode, isTryCast: Boolean): RexNode {
        // If the cast is determined to be invalid, return NULL for TRY_ casts, and the original
        // call for non-TRY_ casts (since they need to raise an error at runtime).
        val nullBoolean = rexBuilder.makeNullLiteral(call.type)
        val errorValue = if (isTryCast) { nullBoolean } else { call }
        return if (operand is RexLiteral) {
            when (operand.type.sqlTypeName) {
                SqlTypeName.FLOAT,
                SqlTypeName.DOUBLE,
                -> {
                    val asNumber = operand.getValueAs(Double::class.java)
                    if (asNumber == null || !asNumber.isFinite()) {
                        errorValue
                    } else {
                        rexBuilder.makeLiteral(asNumber != 0.0)
                    }
                }
                SqlTypeName.TINYINT,
                SqlTypeName.SMALLINT,
                SqlTypeName.INTEGER,
                SqlTypeName.BIGINT,
                SqlTypeName.DECIMAL,
                -> {
                    val asNumber = operand.getValueAs(BigDecimal::class.java)
                    if (asNumber == null) {
                        errorValue
                    } else {
                        rexBuilder.makeLiteral(asNumber.signum() != 0)
                    }
                }
                SqlTypeName.CHAR,
                SqlTypeName.VARCHAR,
                -> {
                    val asString = operand.getValueAs(String::class.java)
                    if (asString == null) {
                        errorValue
                    } else {
                        when (asString.lowercase(Locale.ROOT)) {
                            "true", "t", "yes", "y", "on", "1" -> {
                                rexBuilder.makeLiteral(true)
                            }
                            "false", "f", "no", "n", "off", "0" -> {
                                rexBuilder.makeLiteral(false)
                            }
                            else -> {
                                errorValue
                            }
                        }
                    }
                }
                SqlTypeName.NULL -> nullBoolean
                else -> call
            }
        } else { call }
    }

    /**
     * Simplifies a cast to a time type.
     *
     * @param call The original cast call.
     * @param operand The argument being casted.
     */
    private fun simplifyTimeCast(call: RexCall, operand: RexNode): RexNode {
        // If the cast is determined to be invalid, return NULL for TRY_ casts, and the original
        // call for non-TRY_ casts (since they need to raise an error at runtime).
        // We are conservative with String parsing here.
        val nullTime = rexBuilder.makeNullLiteral(call.type)
        return if (operand is RexLiteral) {
            if (operand.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                timestampLiteralToTime(operand, call.getType().precision)
            } else {
                when (operand.type.sqlTypeName) {
                    SqlTypeName.TIMESTAMP -> timestampLiteralToTime(operand, call.getType().precision)
                    SqlTypeName.TIME -> {
                        // Handle precision changes.
                        val timeStr = operand.getValueAs(TimeString::class.java)
                        if (timeStr == null) {
                            // Note: This shouldn't be reachable.
                            call
                        } else {
                            rexBuilder.makeTimeLiteral(timeStr, call.getType().precision)
                        }
                    }
                    SqlTypeName.CHAR,
                    SqlTypeName.VARCHAR,
                    -> { stringLiteralToTime(call, operand) }
                    SqlTypeName.NULL -> nullTime
                    else -> call
                }
            }
        } else { call }
    }

    /**
     * Simplify Bodo call expressions that don't depend on handling unknown
     * values in custom way.
     */
    private fun simplifyBodoCast(call: RexCall, operand: RexNode, targetType: RelDataType, isTryCast: Boolean, unknownAs: RexUnknownAs): RexNode {
        return if (call.type.sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
            simplifyTimestampLtzCast(call, operand, isTryCast)
        } else {
            when (targetType.sqlTypeName) {
                SqlTypeName.INTEGER, SqlTypeName.SMALLINT, SqlTypeName.TINYINT, SqlTypeName.BIGINT -> simplifyIntegerCast(
                    call,
                    operand,
                    isTryCast,
                )

                SqlTypeName.FLOAT, SqlTypeName.DOUBLE -> simplifyDoubleCast(call, operand, isTryCast)
                SqlTypeName.DECIMAL -> simplifyDecimalCast(
                    call,
                    operand,
                    isTryCast,
                )

                SqlTypeName.DATE -> simplifyDateCast(call, operand)
                SqlTypeName.TIMESTAMP -> simplifyTimestampNtzCast(call, operand, isTryCast)
                SqlTypeName.TIMESTAMP_TZ -> simplifyTimestampTzCast(call, operand, isTryCast)
                SqlTypeName.CHAR, SqlTypeName.VARCHAR -> simplifyVarcharCast(call, operand, unknownAs)
                SqlTypeName.BOOLEAN -> simplifyBooleanCast(call, operand, isTryCast)
                SqlTypeName.TIME -> simplifyTimeCast(call, operand)
                SqlTypeName.BINARY, SqlTypeName.VARBINARY -> simplifyVarbinaryCast(call, operand, unknownAs, isTryCast)
                else -> call
            }
        }
    }

    /**
     * Takes in the arguments to an addition or subtraction where one argument is a date
     * and the other is an interval, then returns the arguments such that the date is
     * first & the interval is second, as well as the sign to multiply the offset by.
     *
     * @param lit1 The lhs operand of the operation
     * @param lit2 The rhs operand of the operaiton
     * @param isPlus Whether the operation was a +
     * @return
     */
    private fun getDateMathArguments(lit1: RexLiteral, lit2: RexLiteral, isPlus: Boolean): Triple<RexLiteral, RexLiteral, Int> {
        val multiplier = if (isPlus) { 1 } else { -1 }
        return if (lit1.type.sqlTypeName in listOf(SqlTypeName.DATE, SqlTypeName.TIMESTAMP)) {
            Triple(lit1, lit2, multiplier)
        } else {
            Triple(lit2, lit1, -multiplier)
        }
    }

    /**
     * Compute the PLUS/MINUS operator between two literals. This extends
     * Calcite's behavior, so we only support Timestamp +/- Interval
     */
    private fun bodoLiteralPlusMinus(call: RexCall, lit1: RexLiteral, lit2: RexLiteral, isPlus: Boolean): RexNode {
        val supportedIntervals = listOf(SqlTypeName.INTERVAL_DAY, SqlTypeName.INTERVAL_MONTH, SqlTypeName.INTERVAL_YEAR)
        if ((lit1.type.sqlTypeName in listOf(SqlTypeName.DATE, SqlTypeName.TIMESTAMP) && supportedIntervals.contains(lit2.type.sqlTypeName)) || (isPlus && supportedIntervals.contains(lit1.type.sqlTypeName) && lit2.type.sqlTypeName in listOf(SqlTypeName.DATE, SqlTypeName.TIMESTAMP))) {
            val (dateLiteral, intervalLiteral, signMultiplier) = getDateMathArguments(lit1, lit2, isPlus)
            val calendarVal = dateLiteral.getValueAs(Calendar::class.java)!!
            val intervalDecimal = intervalLiteral.getValueAs(BigDecimal::class.java)!!
            val (intervalInt, calendarUnit) = when (intervalLiteral.type.sqlTypeName) {
                // Years & Months have the integer value represented in months
                SqlTypeName.INTERVAL_YEAR,
                SqlTypeName.INTERVAL_MONTH,
                -> Pair(intervalDecimal.toInt(), Calendar.MONTH)
                // Days have their integer values represented in milliseconds
                SqlTypeName.INTERVAL_DAY -> Pair(intervalDecimal.divide(BigDecimal(24 * 60 * 60 * 1000)).toInt(), Calendar.DAY_OF_MONTH)
                else -> return call
            }
            calendarVal.add(calendarUnit, intervalInt * signMultiplier)
            if (dateLiteral.type.sqlTypeName == SqlTypeName.DATE) {
                return rexBuilder.makeDateLiteral(DateString.fromCalendarFields(calendarVal))
            } else {
                return rexBuilder.makeTimestampLiteral(TimestampString.fromCalendarFields(calendarVal), 9)
            }
        }
        return call
    }

    /**
     * Compute the TIMES operator between two literals. This extends
     * Calcite's behavior, so we only support Interval * Integer.
     */
    private fun bodoLiteralTimes(call: RexCall, lit1: RexLiteral, lit2: RexLiteral): RexNode {
        val type1 = lit1.getType()
        val type2 = lit2.getType()
        return if ((SqlTypeFamily.INTEGER.contains(type1) && SqlTypeFamily.INTERVAL_DAY_TIME.contains(type2)) || (SqlTypeFamily.INTERVAL_DAY_TIME.contains(type1) && SqlTypeFamily.INTEGER.contains(type2))) {
            val firstInteger = SqlTypeFamily.INTEGER.contains(type1)
            val intLiteral = if (firstInteger) {
                lit1
            } else {
                lit2
            }
            val intVal = intLiteral.getValueAs(BigDecimal::class.java)!!
            val intervalLiteral = if (firstInteger) {
                lit2
            } else {
                lit1
            }
            val intervalVal = intervalLiteral.getValueAs(BigDecimal::class.java)!!
            val newValue = intervalVal * intVal
            rexBuilder.makeIntervalLiteral(newValue, intervalLiteral.type.intervalQualifier)!!
        } else {
            call
        }
    }

    /**
     * Simplify Bodo calls that involve the + and - operators
     * and are not implemented in Calcite
     */
    private fun simplifyBodoPlusMinus(call: RexCall, isPlus: Boolean): RexNode {
        if (call.operands.size != 2) return call
        val firstArg = call.operands[0]
        val secondArg = call.operands[1]
        return if (firstArg is RexLiteral && secondArg is RexLiteral) {
            bodoLiteralPlusMinus(call, firstArg, secondArg, isPlus)
        } else {
            call
        }
    }

    /**
     * Simplify numeric literals being negated.
     */
    private fun simplifyBodoMinusPrefix(call: RexCall): RexNode {
        val operand = call.operands[0]
        return if (operand is RexLiteral) {
            when (operand.type.sqlTypeName) {
                SqlTypeName.TINYINT,
                SqlTypeName.SMALLINT,
                SqlTypeName.INTEGER,
                SqlTypeName.BIGINT,
                SqlTypeName.FLOAT,
                SqlTypeName.DOUBLE,
                SqlTypeName.DECIMAL,
                -> {
                    val decimalValue = operand.getValueAs(BigDecimal::class.java)!!
                    // Check for integer limits
                    val isSafe = when (operand.type.sqlTypeName) {
                        SqlTypeName.TINYINT -> decimalValue.toLong() != Byte.MIN_VALUE.toLong()
                        SqlTypeName.SMALLINT -> decimalValue.toLong() != Short.MIN_VALUE.toLong()
                        SqlTypeName.INTEGER -> decimalValue.toLong() != Int.MIN_VALUE.toLong()
                        SqlTypeName.BIGINT -> decimalValue.toLong() != Long.MIN_VALUE
                        else -> true
                    }
                    if (isSafe) {
                        val newLiteral = decimalValue.negate()
                        rexBuilder.makeLiteral(newLiteral, call.type)
                    } else {
                        call
                    }
                }
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Simplify Calls to the LIKE operator that can be simplified
     * based on a constant pattern. This looks for cases with:
     * 1. No special characters, just Equals.
     * 2. ONLY % at the start, which is StartsWith
     * 3. ONLY % at the end, which is EndsWith
     * 4. % at the start and end, but not the middle, which is Contains
     * 5. % only in the middle, which is AND(startswith, endswith, LENGTH(A) >= (startPattern + endPattern))
     *
     * We do not have any support for _ in simplification at this time.
     * @param call The original call to LIKE
     * @return Either the original call or a simplified call based on
     * the provided pattern.
     */
    private fun simplifyBodoLike(call: RexCall): RexNode {
        if (call.operator.name != SqlStdOperatorTable.LIKE.name || call.operands.size == 3) {
            return call
        }
        val operands = call.operands
        val pattern = call.operands[1]
        return if (pattern is RexLiteral) {
            val patternString = pattern.getValueAs(String::class.java)!!
            // We cannot rewrite if we see an _
            val unsafeSpecialCharacter = '_'
            if (patternString.contains(unsafeSpecialCharacter)) {
                call
            } else {
                val safeSpecialCharacter = '%'
                if (!patternString.contains(safeSpecialCharacter)) {
                    // If there is no special character this is just EQUALS
                    rexBuilder.makeCall(SqlStdOperatorTable.EQUALS, operands)
                } else {
                    // Determine the start index.
                    var startIndex = 0
                    var endIndex = patternString.length
                    while (startIndex < patternString.length && patternString[startIndex] == safeSpecialCharacter) {
                        startIndex++
                    }
                    while (endIndex > 0 && patternString[endIndex - 1] == safeSpecialCharacter) {
                        endIndex--
                    }
                    if (endIndex < startIndex) {
                        // If endIndex < startIndex then the string is entirely %, which always matches.
                        rexBuilder.makeLiteral(true)
                    } else {
                    val matchSection = patternString.substring(startIndex, endIndex)
                    if (matchSection.contains(safeSpecialCharacter)) {
                        // If we have a % in the middle, we can simplify if there is exactly "1" location.
                        // We search through the string to find the first and last %.
                        val canSimplify = if (startIndex == 0 && endIndex == patternString.length) {
                            while (startIndex < patternString.length && patternString[startIndex] != safeSpecialCharacter) {
                                startIndex++
                            }
                            while (endIndex > 0 && patternString[endIndex - 1] != safeSpecialCharacter) {
                                endIndex--
                            }
                            // We can simplify if all the %s are in the middle
                            patternString.substring(startIndex, endIndex).all { it == safeSpecialCharacter }
                        } else {
                            false
                        }
                        if (canSimplify) {
                            // Generate an AND between startswith and endswith
                            val startString = patternString.substring(0, startIndex)
                            val startLiteral = rexBuilder.makeLiteral(startString)
                            val startsWith = rexBuilder.makeCall(StringOperatorTable.STARTSWITH, operands[0], startLiteral)
                            val endString = patternString.substring(endIndex, patternString.length)
                            val endLiteral = rexBuilder.makeLiteral(endString)
                            val endsWith = rexBuilder.makeCall(StringOperatorTable.ENDSWITH, operands[0], endLiteral)
                            // Ensure that something like a%a doesn't match on 'a'. Do this by checking the length of the string.
                            val length = startString.length + endString.length
                            val lengthLiteral = rexBuilder.makeBigintLiteral(length.toBigDecimal())
                            val lengthCall = rexBuilder.makeCall(StringOperatorTable.LENGTH, operands[0])
                            val comparisonCall = rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN_OR_EQUAL, lengthCall, lengthLiteral)
                            rexBuilder.makeCall(SqlStdOperatorTable.AND, startsWith, endsWith, comparisonCall)
                        } else {
                            call
                        }
                    } else {
                        val matchLiteral = rexBuilder.makeLiteral(matchSection)
                        if (startIndex > 0 && endIndex < patternString.length) {
                            // Both sides have a %, this is a contains
                            rexBuilder.makeCall(StringOperatorTable.CONTAINS, operands[0], matchLiteral)
                        } else if (startIndex > 0) {
                            // Only the start has a %, this is endswith
                            rexBuilder.makeCall(StringOperatorTable.ENDSWITH, operands[0], matchLiteral)
                        } else if (endIndex < patternString.length) {
                            // Only the end has a %, this is startswith
                            rexBuilder.makeCall(StringOperatorTable.STARTSWITH, operands[0], matchLiteral)
                        } else {
                            // This should never be reached
                            call
                        }
                    }
                    }
                }
            }
        } else {
            call
        }
    }

    /**
     * Simplify a Bodo Row operation if it directly matches in the input.
     * Assume we have plan that looks like this:
     *
     * <code>
     *  Project(A=[ROW($0.a, $0.b), $1)
     *     INPUT TYPE - RecordType(INTEGER a, BIGINT b), BIGINT
     *     RELNODE(...)
     * </code>
     *
     * If we detect that the ROW operation is directly matching the input, we can
     * remove it entirely.
     * @param call The original call to ROW.
     * @return Either the ROW or an input ref.
     */
    private fun simplifyBodoRow(call: RexCall): RexNode {
        val operands = call.operands
        // Check that every element is a field access and that element i is field i for its reference inputRef.
        // This prevents possible reordering.
        return if (operands.withIndex().all { it.value is RexFieldAccess && (it.value as RexFieldAccess).referenceExpr is RexInputRef && (it.value as RexFieldAccess).field.index == it.index}) {
            val inputRefs = operands.map { (it as RexFieldAccess).referenceExpr as RexInputRef }.toSet()
            if (inputRefs.size != 1) {
                call
            } else {
                // Verify we don't prune any trailing fields.
                val inputRef = inputRefs.first()
                if (operands.size != inputRef.getType().fieldCount) {
                    call
                } else {
                    inputRef
                }
            }
        } else {
            call
        }
    }

    /**
     * Simplify Bodo calls that involve the * operator
     * and are not implemented in Calcite
     */
    private fun simplifyBodoTimes(call: RexCall): RexNode {
        return if (call.operands.size == 2 && call.operands[0] is RexLiteral && call.operands[1] is RexLiteral) {
            bodoLiteralTimes(call, call.operands[0] as RexLiteral, call.operands[1] as RexLiteral)
        } else {
            call
        }
    }

    /**
     * Simplify a comparison involving a coalesce and a literal
     */
    private fun simplifyCoalesceComparison(call: RexCall): RexNode {
        val firstArg = call.operands[0]
        val secondArg = call.operands[1]
        val callOp = call.op
        val (lit, coalesceCall, litFirst) = if (firstArg is RexLiteral) {
            Triple(firstArg, secondArg as RexCall, true)
        } else {
            Triple(secondArg as RexLiteral, firstArg as RexCall, false)
        }

        // Helper that wraps makeCall. Derives the return type and makes the call
        // We can probably re-deriving the return type in a few locations, but we've had
        // issues with returning the wrong type previously, so I'm being overly defensive.
        fun makeCallWrapper(op: SqlOperator, args: List<RexNode>): RexNode {
            val retType = rexBuilder.deriveReturnType(op, args)
            return rexBuilder.makeCall(retType, op, args)
        }

        // Helper function that breaks down the coalesce call.
        // creates a new comparison between the literal, and one of the two operands of the coalesce function,
        // with the proper ordering.
        fun makeComparison(coalesceArg: RexNode, literal: RexLiteral): RexNode {
            val args = if (litFirst) { listOf(literal, coalesceArg) } else { listOf(coalesceArg, literal) }
            return makeCallWrapper(callOp, args)
        }

        // Decompose COMP((COL, LIT2), LIT1) into OR(IS_TRUE(COMP(COL, LIT1)), AND(IS_NULL(COL), COMP(LIT2, LIT1)))
        val columnComparison: RexNode = makeComparison(coalesceCall.operands[0], lit)
        val columnNullCheck: RexNode = makeCallWrapper(SqlStdOperatorTable.IS_NULL, listOf(coalesceCall.operands[0]))
        val literalComparison = makeComparison(coalesceCall.operands[1], lit)
        // AND(IS_NULL(COL), COMP(LIT2, LIT1))
        val literalComparisonWithNullCheck = makeCallWrapper(SqlStdOperatorTable.AND, listOf<RexNode>(columnNullCheck, literalComparison))
        // IS_TRUE(COMP(COL, LIT1))
        val columnComparisonWithNullCheck = makeCallWrapper(SqlStdOperatorTable.IS_TRUE, listOf(columnComparison))

        // OR(IS_TRUE(COMP(COL, LIT1)), AND(IS_NULL(COL), COMP(LIT2, LIT1)))
        val outputExpression = makeCallWrapper(SqlStdOperatorTable.OR, listOf(columnComparisonWithNullCheck, literalComparisonWithNullCheck))

        // NOTE: If this expression doesn't have the correct nullability (or type in general), the simplifier will
        // automatically insert a cast to the correct type.
        return outputExpression
    }

    /**
     * Determine if an expression is a comparison containing coalesce that can be simplified
     * because a literal argument may be possible to remove.
     */
    private fun isCoalesceComparison(node: RexNode): Boolean {
        val supportedComparison = setOf(SqlKind.EQUALS, SqlKind.NOT_EQUALS, SqlKind.NULL_EQUALS, SqlKind.LESS_THAN, SqlKind.GREATER_THAN, SqlKind.LESS_THAN_OR_EQUAL, SqlKind.GREATER_THAN_OR_EQUAL)
        return if (node is RexCall && supportedComparison.contains(node.kind)) {
            // Note: Comparison operators shouldn't have more than two args,
            // but double check in case a future change doesn't chain equalities.
            val firstArg = node.operands[0]
            val secondArg = node.operands[1]
            // Note: Comparison operators shouldn't have more than two args,
            // but double check to ensure a future change doesn't chain equalities.
            val isValidCoalesceFunction = { r: RexNode -> r is RexCall && r.operator.name == SqlStdOperatorTable.COALESCE.name && r.operands.size == 2 && r.operands[1] is RexLiteral }
            return node.operands.size == 2 && (firstArg is RexLiteral && isValidCoalesceFunction(secondArg)) || (secondArg is RexLiteral && isValidCoalesceFunction(firstArg))
        } else {
            false
        }
    }

    /**
     * @param e The RexNode being checked
     * @return Whether e is a call to a concatenation operation without a separator
     */
    private fun isConcat(e: RexNode): Boolean {
        return e is RexCall && (e.operator.name == SqlStdOperatorTable.CONCAT.name || e.operator.name == StringOperatorTable.CONCAT.name)
    }

    /**
     * Recursively probes through the arguments of calls to concatenation operations
     * to append their arguments to a list, thus "flattening" the tree of concat calls.
     *
     * @param e The RexNode having its arguments extracted
     * @param concatArgs The list where the leaves of the concatenation tree are added to
     */
    private fun collectConcatArgs(e: RexNode, concatArgs: MutableList<RexNode>) {
        if (isConcat(e)) {
            (e as RexCall).operands.forEach { collectConcatArgs(it, concatArgs) }
        } else {
            concatArgs.add(simplify(e))
        }
    }

    /**
     * Performs two kinds of simplifications on calls to || or CONCAT:
     * 1. Flattens a tree of such calls into one call of concat
     *    For example: '%' || ' ' || T.A || '%' || 'ing' -> CONCAT('%', ' ', T.A, '%', 'ing')
     * 2. Combines any adjacent terms that are string literals
     *    For example: CONCAT('%', ' ', T.A, '%', 'ing') -> CONCAT('% ', T.A, '%ing')
     *
     * @param e The call to a concatenation operation that is being simplified.
     * @return A simplified version of the concatenation operation with adjacent
     * string constants combined.
     */
    private fun simplifyConcat(e: RexCall): RexNode {
        val concatArgs: MutableList<RexNode> = mutableListOf()
        e.operands.forEach { collectConcatArgs(it, concatArgs) }
        val compressedArgs: MutableList<RexNode> = mutableListOf()
        concatArgs.forEach {
            if (it is RexLiteral) {
                if (compressedArgs.isNotEmpty() && compressedArgs[compressedArgs.size - 1] is RexLiteral) {
                    val lhs = (compressedArgs.removeAt(compressedArgs.size - 1) as RexLiteral).getValueAs(String::class.java)
                    val rhs = it.getValueAs(String::class.java)
                    compressedArgs.add(rexBuilder.makeLiteral(lhs + rhs))
                } else {
                    compressedArgs.add(it)
                }
            } else {
                compressedArgs.add(it)
            }
        }
        // If the entire sequence was combined, there is no need for a CONCAT call
        if (compressedArgs.size == 1) {
            return compressedArgs[0]
        }
        // If no optimizations were performed, return the original call
        if (e.operands == compressedArgs) {
            return e
        }
        // Upgrade the operator from || to CONCAT if the number of operands changed
        val operator = if (compressedArgs.size == e.operands.size) { e.operator } else { StringOperatorTable.CONCAT }
        return rexBuilder.makeCall(operator, compressedArgs)
    }

    /**
     * @param e The RexNode being checked
     * @return Whether e is a call to a string capitalization operator
     */
    private fun isStringCapitalizationOp(e: RexNode): Boolean {
        return e is RexCall && (e.operator.name == SqlStdOperatorTable.LOWER.name || e.operator.name == SqlStdOperatorTable.UPPER.name)
    }

    /**
     * @param e The call to a string capitalization operation.
     * @return A simplified version of the operation where string literals are
     * folded.
     */
    private fun simplifyStringCapitalizationOp(e: RexCall): RexNode {
        val operand = simplify(e.operands[0])
        if (operand is RexLiteral) {
            val asStr = operand.getValueAs(String::class.java)!!
            val capitalizedStr = if (e.operator.name == SqlStdOperatorTable.UPPER.name) { asStr.uppercase(Locale.ROOT) } else { asStr.lowercase(Locale.ROOT) }
            return rexBuilder.makeLiteral(capitalizedStr)
        }
        return e
    }

    /**
     * @param e The RexNode being checked
     * @param i The operand position being checked
     * @return Whether e is a call to an ordering comparison where the ith argument
     * is a call to LEAST or GREATEST.
     */
    private fun isCompareLeastGreatest(e: RexNode, i: Int): Boolean {
        return e is RexCall && (e.kind.belongsTo(listOf(SqlKind.LESS_THAN, SqlKind.LESS_THAN_OR_EQUAL, SqlKind.GREATER_THAN, SqlKind.GREATER_THAN_OR_EQUAL))) && (e.operands[i].kind.belongsTo(listOf(SqlKind.LEAST, SqlKind.GREATEST)))
    }

    /**
     * Returns if a RexNode is a call to LENGTH or an
     * equivalent function.
     */
    private fun isLength(e: RexNode): Boolean {
        val lengthFunctions = setOf(
            StringOperatorTable.LENGTH.name,
            SqlStdOperatorTable.CHAR_LENGTH.name,
        )
        return e is RexCall && (lengthFunctions.contains(e.operator.name))
    }

    /**
     * Returns if a RexNode is a call to IFF.
     */
    private fun isIFF(e: RexNode): Boolean {
        return e is RexCall && e.operator.name == CondOperatorTable.IFF_FUNC.name
    }

    /**
     * Simplify IFF based on the value of the arguments. There are 4 supported
     * transformations
     *
     * IFF(true, A, B) -> A
     * IFF(false, A, B) -> B
     * IFF(null, A, B) -> B
     * IFF(A, null, null) -> null
     */
    private fun simplifyIFF(e: RexCall): RexNode {
        val arg0 = e.operands[0]
        val arg1 = e.operands[1]
        val arg2 = e.operands[2]
        return if (arg0.isAlwaysTrue) {
            arg1
        } else if (arg0.isAlwaysFalse || arg0.type.sqlTypeName == SqlTypeName.NULL) {
            arg2
        } else if (arg1.type.sqlTypeName == SqlTypeName.NULL && arg2.type.sqlTypeName == SqlTypeName.NULL) {
            rexBuilder.makeNullLiteral(e.getType())
        } else {
            e
        }
    }

    /**
     * Returns a simplification in the formats below:
     * LEAST(A, B, C) <= D  --> A <= D OR B <= D OR C <= D
     * LEAST(A, B, C) >= D  --> A >= D AND B >= D AND C >= D
     * GREATEST(A, B, C) <= D  --> A <= D AND B <= D AND C <= D
     * GREATEST(A, B, C) >= D  --> A >= D OR B >= D OR C >= D
     * (+ the same for > and <)
     *
     * @param leastGreatestArgs the arguments to the LEAST or GREATEST call
     * @param rhs the value that the call to LEAST or GREATEST is being compared to
     * @param comparison the type of comparison being done
     * @param isConjunction whether the final answer can be represented by a conjunction
     * (a sequence of AND calls) instead of a disjunction (a sequence of OR calls)
     * @return The simplified operation in the format described above
     */
    private fun simplifyCompareLeastGreatest(leastGreatestArgs: List<RexNode>, rhs: RexNode, comparison: SqlOperator, isConjunction: Boolean): RexNode {
        val distributedComparison = leastGreatestArgs.map { rexBuilder.makeCall(comparison, listOf(it, rhs)) }
        if (isConjunction) { return RexUtil.composeConjunction(rexBuilder, distributedComparison) }
        return RexUtil.composeDisjunction(rexBuilder, distributedComparison)
    }

    /**
     * @param e The call to <, <=, > or >= where the first operand is a call to LEAST or GREATEST
     * @return An equivalent expression where the call to LEAST or GREATEST is eliminated by
     * distributing the comparison across the arguments, for example:
     *
     * LEAST(A, B, C) <= D  --> A <= D OR B <= D OR C <= D
     */
    private fun simplifyCompareLeastGreatest(e: RexCall): RexNode {
        val lhs = e.operands[0] as RexCall
        val leastGreatestArgs = lhs.operands
        val rhs = e.operands[1]
        // The operation will become a conjunction if it is a </<= with GREATEST or a >/>= with LEAST
        val isConjunction = (e.kind.belongsTo(listOf(SqlKind.LESS_THAN, SqlKind.LESS_THAN_OR_EQUAL))) == (lhs.kind == SqlKind.GREATEST)
        return simplifyCompareLeastGreatest(leastGreatestArgs, rhs, e.operator, isConjunction)
    }

    /**
     * @param e The call to <, <=, > or >=
     * @return The same call but with the argument order and comparison switched.
     *
     * For example: A <= B  -->  B >= A
     */
    private fun reverseComparison(e: RexCall): RexNode {
        val firstArg = e.operands[0]
        val secondArg = e.operands[1]
        val comparison = (e.operator as SqlBinaryOperator).reverse()!!
        return rexBuilder.makeCall(comparison, listOf(secondArg, firstArg))
    }

    /**
     * Simplify a length function call on a literal to
     * a constant evaluation.
     */
    private fun simplifyLength(e: RexCall): RexNode {
        return if (e.operands[0] is RexLiteral && SqlTypeFamily.CHARACTER.contains(e.operands[0].type)) {
            val literal = e.operands[0] as RexLiteral
            val literalValue = literal.getValueAs(String::class.java)
            return literalValue?.let { rexBuilder.makeBigintLiteral(BigDecimal(literalValue.length)) } ?: e
        } else {
            e
        }
    }

    /**
     * @param e The RexNode being checked
     * @return Whether the node is a call to a function that adds
     * to a date/time/timestamp using the Snowflake function syntax,
     * e.g. DATEADD(DAY, 1, T).
     *
     * Only allows DATEADD and TIMEADD with 3 arguments. The DATEADD
     * with 2 arguments (or DATE_ADD) use a different form, and
     * TIMESTAMPADD is elaborated into another form during conversion.
     */
    private fun isSnowflakeDateaddOp(e: RexNode): Boolean {
        if (e is RexCall) {
            return e.operator.name == DatetimeOperatorTable.DATEADD.name && e.operands.size == 3
        }
        return false
    }

    /**
     * Converts a date unit string to a pair of a calendar field enum and a
     * multiplier such that the unit string can be expressed as the field times
     * the multiplier.
     *
     * @param unit The date unit as a (lowercase) string
     * @return The field enum and multiplier corresponding to the unit, or null if
     * the string does not match one of the units.
     */
    private fun getDateUnitAsCalendarAndMultiplier(unit: DatetimeFnUtils.DateTimePart): Pair<Int, Int>? {
        return when (unit) {
            DatetimeFnUtils.DateTimePart.YEAR -> Pair(Calendar.YEAR, 1)
            DatetimeFnUtils.DateTimePart.QUARTER -> Pair(Calendar.MONTH, 3)
            DatetimeFnUtils.DateTimePart.MONTH -> Pair(Calendar.MONTH, 1)
            DatetimeFnUtils.DateTimePart.WEEK -> Pair(Calendar.WEEK_OF_YEAR, 1)
            DatetimeFnUtils.DateTimePart.DAY -> Pair(Calendar.DAY_OF_YEAR, 1)
            else -> return null
        }
    }

    /**
     * Converts a time unit string to the corresponding amount of milliseconds.
     *
     * @param unit The time unit as a (lowercase) string
     * @return The multiplier corresponding to the number of milliseconds,
     * or null if the string does not match one of the units.
     */
    private fun getTimeUnitAsMultiplier(unit: DatetimeFnUtils.DateTimePart): Long? {
        return when (unit) {
            DatetimeFnUtils.DateTimePart.HOUR -> NANOSEC_PER_HOUR
            DatetimeFnUtils.DateTimePart.MINUTE -> NANOSEC_PER_MINUTE
            DatetimeFnUtils.DateTimePart.SECOND -> NANOSEC_PER_SECOND
            DatetimeFnUtils.DateTimePart.MILLISECOND -> NANOSEC_PER_MILLISECOND
            DatetimeFnUtils.DateTimePart.MICROSECOND -> NANOSEC_PER_MICROSECOND
            DatetimeFnUtils.DateTimePart.NANOSECOND -> 1
            else -> return null
        }
    }

    /**
     * Takes a string representing a time/timestamp and extracts the
     * components smaller than milliseconds in nanoseconds, as an integer.
     *
     * For example: "12:30:45.25091" -> 910_000
     * For example: "1999-12-31 11:59:59.987654321" -> 654_321
     *
     * @param s The string representing a time/timestamp
     * @return The sub-milliseconds component of the string in nanoseconds.
     */
    private fun getSubMilliAsNs(s: String): Int {
        val dotIndex: Int = s.indexOf('.')
        var nsString = if (dotIndex >= 0) { s.drop(dotIndex + 4) } else { "" }
        var paddedNsString = nsString.padEnd(6, '0')
        return paddedNsString.toInt()
    }

    /**
     * Calculates the new DateLiteral expression caused by adding a specific amount of
     * a date unit to a starting date, or the original expression if that fails.
     *
     * @param original The original call to DATEADD
     * @param date The starting date
     * @param unit The unit to add
     * @param offset The amount of the unit to add
     * @return The new DateLiteral, or the original call if the simplification fails.
     */
    private fun simplifyAddToDate(original: RexNode, date: Calendar, unit: DatetimeFnUtils.DateTimePart, offset: Long): RexNode {
        val unitAndMultiplier = getDateUnitAsCalendarAndMultiplier(unit) ?: return original
        val (field, multiplier) = unitAndMultiplier
        date.add(field, offset.toInt() * multiplier)
        return rexBuilder.makeDateLiteral(DateString.fromCalendarFields(date))
    }

    /**
     * Calculates the new TimeLiteral expression caused by adding a specific amount of
     * a time unit to a starting time, or the original expression if that fails.
     *
     * @param original The original call to DATEADD
     * @param time The starting time
     * @param unit The unit to add
     * @param offset The amount of the unit to add
     * @return The new TimeLiteral, or the original call if the simplification fails.
     */
    private fun simplifyAddToTime(original: RexNode, time: TimeString, unit: DatetimeFnUtils.DateTimePart, offset: Long): RexNode {
        val multiplier: Long = getTimeUnitAsMultiplier(unit) ?: return original
        // Convert the time to nanoseconds since midnight
        val milli = time.millisOfDay
        val subMilli = getSubMilliAsNs(time.toString())
        val asNs = milli * NANOSEC_PER_MILLISECOND + subMilli
        // Add the desired offset to the total nanoseconds since midnight, dealing with overflow
        // to the next/previous day if necessary
        val newNs = Math.floorMod(asNs + multiplier * offset, NANOSEC_PER_DAY)
        // Extract the various components of the new time
        val hour = newNs / NANOSEC_PER_HOUR
        val minute = (newNs / NANOSEC_PER_MINUTE) % 60
        val second = (newNs / NANOSEC_PER_SECOND) % 60
        val subsecond = newNs % NANOSEC_PER_SECOND
        // Use the components to construct the new time
        var newTime = TimeString(hour.toInt(), minute.toInt(), second.toInt()).withNanos(subsecond.toInt())
        return rexBuilder.makeTimeLiteral(newTime, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION)
    }

    /**
     * Calculates the new TimestampLiteral expression caused by adding a specific amount of
     * a date/time unit to a starting time, or the original expression if that fails.
     *
     * @param original The original call to DATEADD
     * @param time The starting timestamp
     * @param unit The unit to add
     * @param offset The amount of the unit to add
     * @return The new TimestampLiteral, or the original call if the simplification fails.
     */
    private fun simplifyAddToTimestamp(original: RexNode, timestamp: TimestampString, unit: DatetimeFnUtils.DateTimePart, offset: Long): RexNode {
        // Extract the sub-second components of the original timestamp
        val millisEpoch = timestamp.millisSinceEpoch
        val subMilli = getSubMilliAsNs(timestamp.toString())
        if (unit in setOf(DatetimeFnUtils.DateTimePart.YEAR, DatetimeFnUtils.DateTimePart.QUARTER, DatetimeFnUtils.DateTimePart.MONTH, DatetimeFnUtils.DateTimePart.WEEK, DatetimeFnUtils.DateTimePart.DAY)) {
            val unitAndMultiplier = getDateUnitAsCalendarAndMultiplier(unit) ?: return original
            // Calculate the entire sub-second component in nanoseconds.
            val ns = ((millisEpoch % 1000) * NANOSEC_PER_MILLISECOND + subMilli).toInt()
            // Add the date unit to the original timestamp
            val date = timestamp.toCalendar()
            val (field, multiplier) = unitAndMultiplier
            date.add(field, offset.toInt() * multiplier)
            // Reconstruct the timestamp by replacing the subsecond components with the
            // subsecond components of the original timestamp
            val ts = TimestampString.fromCalendarFields(date).withNanos(ns)
            return rexBuilder.makeTimestampLiteral(ts, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION)
        } else {
            val multiplier: Long = getTimeUnitAsMultiplier(unit) ?: return original
            // Calculate the total number of nanoseconds since the unix epoch
            val ns = millisEpoch * NANOSEC_PER_MILLISECOND + subMilli
            // Add the timedelta to the number of nanoseconds and extract the
            // new milliseconds since the epoch as well as the subsecond components
            val newNs = ns + offset * multiplier
            val newMsEpoch = newNs / NANOSEC_PER_MILLISECOND
            val newSubsecond = newNs % NANOSEC_PER_SECOND
            // Reconstruct the timestamp using the milliseconds since the unix epoch,
            // then replace the subsecond components with the correct subsecond amounts
            val ts = TimestampString.fromMillisSinceEpoch(newMsEpoch).withNanos(newSubsecond.toInt())
            return rexBuilder.makeTimestampLiteral(ts, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION)
        }
    }

    /**
     * @param e A call to DATEADD or an equivalent function in 3-argument form
     * @return If the call has all-literal arguments, returns the result of
     * the corresponding date arithmetic. Otherwise, returns the original call.
     * For example: DATEADD(YEAR, 2, DATE '2023-9-27') -> DATE '2025-9-27'
     */
    private fun simplifySnowflakeDateaddOp(e: RexCall): RexNode {
        if (!e.operands.all { it is RexLiteral }) {
            return e
        }
        val unitLiteral = e.operands[0] as RexLiteral
        val offset = (e.operands[1] as RexLiteral).getValueAs(BigDecimal::class.java)!!.setScale(0, RoundingMode.HALF_UP).toLong()
        val base = e.operands[2] as RexLiteral

        // Extract the time unit should always be an symbol literal
        assert(unitLiteral.typeName == SqlTypeName.SYMBOL) { "Internal error in simplifySnowflakeDateaddOp: arg0 is not a symbol" }

        val unit: DatetimeFnUtils.DateTimePart =
            unitLiteral.getValueAs(DatetimeFnUtils.DateTimePart::class.java)
                ?: throw RuntimeException("Internal error in simplifySnowflakeDateaddOp: arg0 is not the expected enum")

        val isTime = setOf(
            DatetimeFnUtils.DateTimePart.HOUR,
            DatetimeFnUtils.DateTimePart.MINUTE,
            DatetimeFnUtils.DateTimePart.SECOND,
            DatetimeFnUtils.DateTimePart.MILLISECOND,
            DatetimeFnUtils.DateTimePart.MICROSECOND,
            DatetimeFnUtils.DateTimePart.NANOSECOND,
        ).contains(unit)

        // Ensure that we only allow tz-naive timestamps
        if (base.type is BasicSqlType) {
            if (base.typeName == SqlTypeName.DATE) {
                // If the first argument is a date but the unit is a time unit, upcast
                // to timestamp then use timestamp addition
                val dateLiteral = base.getValueAs(DateString::class.java)!!
                val calendar = dateLiteral.toCalendar()
                if (isTime) {
                    val timestampLiteral = TimestampString(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH) + 1, calendar.get(Calendar.DAY_OF_MONTH), 0, 0, 0)
                    return simplifyAddToTimestamp(e, timestampLiteral, unit, offset)
                } else {
                    return simplifyAddToDate(e, calendar, unit, offset)
                }
            }

            if (base.typeName == SqlTypeName.TIME) {
                // Throw an error if trying to add a non-time unit to a time literal
                if (!isTime) throw Exception("Invalid operation: $e")
                val timeLiteral = base.getValueAs(TimeString::class.java)!!
                return simplifyAddToTime(e, timeLiteral, unit, offset)
            }

            if (base.typeName == SqlTypeName.TIMESTAMP) {
                val timestampLiteral = base.getValueAs(TimestampString::class.java)!!
                return simplifyAddToTimestamp(e, timestampLiteral, unit, offset)
            }
        }

        return e
    }

    /**
     * Simplify RexOver by simplifying the values in Operands,
     * Partition By, and Order By.
     *
     * Note, it might be possible to pass unknownAs here, but that requires
     * an investigation to verify how it should be extended to each component.
     */
    private fun simplifyRexOver(e: RexOver): RexNode {
        val newOperands = e.operands.map { x -> simplify(x) }
        val operandChange = e.operands.zip(newOperands).any { x -> x.first != x.second }
        val window = e.window
        val newPartitionBy = window.partitionKeys.map { x -> simplify(x) }
        val partitionByChange = window.partitionKeys.zip(newPartitionBy).any { x -> x.first != x.second }
        val listBuilder = ImmutableList.builder<RexFieldCollation>()
        for (w in window.orderKeys) {
            listBuilder.add(RexFieldCollation(simplify(w.key), w.value))
        }
        val newOrderByNodes = listBuilder.build()
        val orderByChange = window.orderKeys.zip(newOrderByNodes).any { x -> x.first.key != x.second.key }
        // Generate a new node if there is any change.
        return if (operandChange || partitionByChange || orderByChange) {
            rexBuilder.makeOver(
                e.getType(),
                e.operator as SqlAggFunction,
                newOperands,
                newPartitionBy,
                newOrderByNodes,
                window.lowerBound,
                window.upperBound,
                window.isRows,
                // Note allowPartial and nullWhenCountZero are set
                // to avoid requiring case based optimizations.
                true,
                false,
                e.isDistinct,
                e.ignoreNulls(),
            )
        } else {
            e
        }
    }

    /**
     * Returns true if a RexCall is any kind of cast
     */
    private fun isCast(e: RexNode): Boolean {
        return if (e is RexCall) { e.kind == SqlKind.CAST || e.kind == SqlKind.SAFE_CAST || JsonPecUtil.isCastFunc(e) } else { false }
    }

    /**
     * Deprecated: Determine if the function call is a date conversion function that we may
     * be able to simplify.
     */
    private fun isDateConversion(e: RexNode): Boolean {
        return e is RexCall && (
                e.operator.name == CastingOperatorTable.TO_DATE.name ||
                        e.operator.name == CastingOperatorTable.TRY_TO_DATE.name
                ) && e.operands.size == 1
    }

    /**
     * Determine if a rex node is a comparison between two literals
     */
    private fun isLiteralComparison(e: RexNode): Boolean {
        return (e is RexCall) && listOf(
            SqlStdOperatorTable.LESS_THAN.name,
            SqlStdOperatorTable.LESS_THAN_OR_EQUAL.name,
            SqlStdOperatorTable.GREATER_THAN.name,
            SqlStdOperatorTable.GREATER_THAN_OR_EQUAL.name,
            SqlBodoOperatorTable.NULL_EQUALS.name,
            SqlStdOperatorTable.EQUALS.name,
            SqlStdOperatorTable.NOT_EQUALS.name,
            SqlStdOperatorTable.IS_DISTINCT_FROM.name,
            SqlStdOperatorTable.IS_NOT_DISTINCT_FROM.name,
            CondOperatorTable.EQUAL_NULL.name,
        ).contains(e.operator.name) && (e.operands.size == 2) && e.operands.all { it is RexLiteral }
    }

    // Attempts to coerce a literal to a BigDecimal, returning null if this is not possible
    private fun getAsDecimal(e: RexLiteral): BigDecimal? {
        return when (e.type.sqlTypeName) {
            SqlTypeName.TINYINT,
            SqlTypeName.SMALLINT,
            SqlTypeName.INTEGER,
            SqlTypeName.BIGINT,
            SqlTypeName.FLOAT,
            SqlTypeName.DOUBLE,
            SqlTypeName.REAL,
            SqlTypeName.DECIMAL -> e.getValueAs(BigDecimal::class.java)
            else -> null
        }
    }

    // Attempts to simplify comparisons between two literals in forms that are not
    // supported by the regular simplifier, e.g. decimal_literal >= double_literal
    private fun simplifyLiteralComparison(e: RexCall): RexNode {
        val asDecimal0 = getAsDecimal(e.operands[0] as RexLiteral)
        val asDecimal1 = getAsDecimal(e.operands[1] as RexLiteral)
        return if (asDecimal0 != null && asDecimal1 != null) {
            val comparisonInt = asDecimal0.compareTo(asDecimal1)
            val asBool = when (e.operator.name) {
                SqlStdOperatorTable.LESS_THAN.name -> comparisonInt < 0
                SqlStdOperatorTable.LESS_THAN_OR_EQUAL.name -> comparisonInt <= 0
                SqlStdOperatorTable.GREATER_THAN.name -> comparisonInt > 0
                SqlStdOperatorTable.GREATER_THAN_OR_EQUAL.name -> comparisonInt >= 0
                SqlStdOperatorTable.EQUALS.name,
                CondOperatorTable.EQUAL_NULL.name,
                SqlBodoOperatorTable.NULL_EQUALS.name,
                SqlStdOperatorTable.IS_NOT_DISTINCT_FROM.name -> comparisonInt == 0
                SqlStdOperatorTable.NOT_EQUALS.name,
                SqlStdOperatorTable.IS_DISTINCT_FROM.name -> comparisonInt != 0
                else -> return e
            }
            return rexBuilder.makeLiteral(asBool)
        } else { e }
    }

    /**
     * Implementation of simplifyUnknownAs where we simplify custom Bodo functions
     * and then dispatch to the regular RexSimplifier.
     */
    override fun simplify(e: RexNode, unknownAs: RexUnknownAs): RexNode {
        // Before doing anything else, do any PEC rewrites

        if (JsonPecUtil.isPec(e)) return simplify(JsonPecUtil.rewritePec(e as RexCall, rexBuilder), unknownAs)
        val simplifiedNode = when (e.kind) {
            SqlKind.PLUS -> simplifyBodoPlusMinus(e as RexCall, true)
            SqlKind.MINUS -> simplifyBodoPlusMinus(e as RexCall, false)
            SqlKind.TIMES -> simplifyBodoTimes(e as RexCall)
            SqlKind.MINUS_PREFIX -> simplifyBodoMinusPrefix(e as RexCall)
            SqlKind.LIKE -> simplifyBodoLike(e as RexCall)
            SqlKind.ROW -> simplifyBodoRow(e as RexCall)
            else -> when {
                isCast(e) -> simplifyBodoCast(e as RexCall, e.operands[0], e.type, e.kind == SqlKind.SAFE_CAST, unknownAs)
                // TODO: Remove when we simplify TO_DATE as ::DATE
                isDateConversion(e) -> simplifyDateCast(e as RexCall, e.operands[0])
                isConcat(e) -> simplifyConcat(e as RexCall)
                isStringCapitalizationOp(e) -> simplifyStringCapitalizationOp(e as RexCall)
                isSnowflakeDateaddOp(e) -> simplifySnowflakeDateaddOp(e as RexCall)
                isCompareLeastGreatest(e, 0) -> simplifyCompareLeastGreatest(e as RexCall)
                isCompareLeastGreatest(e, 1) -> simplifyCompareLeastGreatest(reverseComparison(e as RexCall) as RexCall)
                isCoalesceComparison(e) -> simplifyCoalesceComparison(e as RexCall)
                isLength(e) -> simplifyLength(e as RexCall)
                isIFF(e) -> simplifyIFF(e as RexCall)
                e is RexOver -> simplifyRexOver(e)
                isLiteralComparison(e) -> simplifyLiteralComparison(e as RexCall)
                else -> e
            }
        }
        return super.simplify(simplifiedNode, unknownAs)
    }

    companion object {
        const val NANOSEC_PER_DAY: Long = 86_400_000_000_000L
        const val NANOSEC_PER_HOUR: Long = 3_600_000_000_000L
        const val NANOSEC_PER_MINUTE: Long = 60_000_000_000L
        const val NANOSEC_PER_SECOND: Long = 1_000_000_000L
        const val NANOSEC_PER_MILLISECOND: Long = 1_000_000L
        const val NANOSEC_PER_MICROSECOND: Long = 1_000L

        const val SECOND_EPOCH_THRESHOLD: Long = 31_536_000_000L
        const val MILLISECOND_EPOCH_THRESHOLD: Long = 31_536_000_000_000L
        const val MICROSECOND_EPOCH_THRESHOLD: Long = 31_536_000_000_000_000L

        const val SECOND_PER_MINUTE: Int = 60
        const val SECOND_PER_HOUR: Int = 3600
        const val MINUTE_PER_HOUR: Int = 60

        @JvmStatic
        val monthCodeList = listOf("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    }
}
