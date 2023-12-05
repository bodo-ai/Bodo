package org.apache.calcite.rex

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import org.apache.calcite.avatica.util.TimeUnitRange
import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.sql.SqlBinaryOperator
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.BasicSqlType
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.Bug
import org.apache.calcite.util.DateString
import org.apache.calcite.util.TimeString
import org.apache.calcite.util.TimestampString
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.*
import java.util.regex.Pattern

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

    private fun stringLiteralToDate(call: RexCall, literal: RexLiteral): RexNode {
        // Convert a SQL literal being cast to a date with the default Snowflake
        // parsing to a date literal. If the parsed date doesn't match our supported
        // formats then we return the original date.
        val literalString = literal.value2.toString()

        // Just support the basic Snowflake pattern without whitespace.
        val pattern = Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})")
        val matcher = pattern.matcher(literalString)
        return if (matcher.find()) {
            val year = matcher.group(1)
            val month = matcher.group(2)
            val day = matcher.group(3)
            rexBuilder.makeDateLiteral(DateString(Integer.valueOf(year), Integer.valueOf(month), Integer.valueOf(day)))
        } else {
            call
        }
    }

    private fun dateLiteralToTimestamp(literal: RexLiteral, precision: Int): RexNode {
        val calendar = literal.getValueAs(Calendar::class.java)!!
        return rexBuilder.makeTimestampLiteral(TimestampString(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH) + 1, calendar.get(Calendar.DAY_OF_MONTH), 0, 0, 0), precision)
    }

    /**
     * Simplify ::date and to_date functions for supported literals.
     */
    private fun simplifyDateCast(call: RexCall): RexNode {
        return if (call.operands.size == 1 && (call.operands[0] is RexLiteral || call.operands[0] is RexCall)) {
            if (call.operands[0] is RexLiteral) {
                // Resolve constant casts for literals
                val literal = call.operands[0] as RexLiteral
                when (literal.typeName) {
                    SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                    SqlTypeName.DATE -> literal
                    SqlTypeName.VARCHAR, SqlTypeName.CHAR -> stringLiteralToDate(call, literal)
                    else -> call
                }
            } else {
                val innerCall = call.operands[0] as RexCall
                if (innerCall.operator.name == DatetimeOperatorTable.GETDATE.name) {
                    // Replace GETDATE::DATE with CURRENT_DATE
                    rexBuilder.makeCall(SqlStdOperatorTable.CURRENT_DATE)
                } else {
                    call
                }
            }
        } else {
            call
        }
    }

    private fun isDateConversion(e: RexNode): Boolean {
        return e is RexCall && (
            e.operator.name == CastingOperatorTable.TO_DATE.name ||
                e.operator.name == CastingOperatorTable.TRY_TO_DATE.name ||
                e.operator.name == SqlBodoOperatorTable.DATE.name
            )
    }

    /**
     * Convert a SQL literal being cast to a timestamp with the default Snowflake
     * parsing to a timestamp literal. If the parsed date doesn't match our supported
     * formats then we return the original date.
     *
     * @param call The original call to cast the string to a timestamp
     * @param literal The string literal being casted
     * @return Either the simplified timestamp literal, or the original call
     */
    private fun stringLiteralToTimestamp(call: RexCall, literal: RexLiteral): RexNode {
        // Convert a SQL literal being cast to a timestamp with the default Snowflake
        // parsing to a timestamp literal. If the parsed date doesn't match our supported
        // formats then we return the original cast call.
        var literalString = literal.value2.toString()

        val year: Int
        val month: Int
        val day: Int
        var hour = 0
        var minute = 0
        var second = 0
        var subsecond = ""
        var result: RexNode = call

        // The basic Snowflake (date-only) pattern without whitespace.
        val dateStringPattern = Pattern.compile("^(\\d{4})-(\\d{2})-(\\d{2})")
        val dateStringMatcher = dateStringPattern.matcher(literalString)
        if (dateStringMatcher.find()) {
            year = Integer.valueOf(dateStringMatcher.group(1))
            month = Integer.valueOf(dateStringMatcher.group(2))
            day = Integer.valueOf(dateStringMatcher.group(3))
            literalString = literalString.substring(dateStringMatcher.end())

            // The basic pattern for the time component of a timestamp following the date component
            val timeStringPattern = Pattern.compile("^[ T](\\d{2}):(\\d{2}):(\\d{2})")
            val timeStringMatcher = timeStringPattern.matcher(literalString)
            if (timeStringMatcher.find()) {
                hour = Integer.valueOf(timeStringMatcher.group(1))
                minute = Integer.valueOf(timeStringMatcher.group(2))
                second = Integer.valueOf(timeStringMatcher.group(3))
                literalString = literalString.substring(timeStringMatcher.end())

                // The pattern for the sub-second components of a timestamp string.
                val subsecondStringPattern = Pattern.compile("^(\\d{1,9})")
                val subsecondStringMatcher = subsecondStringPattern.matcher(literalString)
                if (subsecondStringMatcher.find()) {
                    subsecond = subsecondStringMatcher.group(1)
                    literalString = literalString.substring(timeStringMatcher.end())
                }
            }
            // Verify that the remainder of the string is either empty or "+00:00"
            if (literalString.isEmpty() || literalString == "+00:00") {
                var tsString = TimestampString(
                    Integer.valueOf(year),
                    Integer.valueOf(month),
                    Integer.valueOf(day),
                    Integer.valueOf(hour),
                    Integer.valueOf(minute),
                    Integer.valueOf(second),
                )
                if (subsecond != "") {
                    tsString = tsString.withFraction(subsecond)
                }
                result = rexBuilder.makeTimestampLiteral(tsString, call.getType().precision)
            }
        }
        return result
    }

    /**
     * Simplify to_timestamp for supported literals. Note this only
     * supports TZ-Naive timestamps.
     */
    private fun simplifyTimestampCast(call: RexCall): RexNode {
        return if (call.getType() is BasicSqlType && call.operands.size == 1 && call.operands[0] is RexLiteral) {
            val literal = call.operands[0] as RexLiteral
            when (literal.typeName) {
                SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                SqlTypeName.VARCHAR,
                SqlTypeName.CHAR,
                -> stringLiteralToTimestamp(call, literal)
                SqlTypeName.DATE -> dateLiteralToTimestamp(literal, call.getType().precision)
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Simplify Varchar casts that are equivalent in our type system. These
     * are scalar casts to varchar or varchar casts that only change precision.
     */
    private fun simplifyVarcharCast(call: RexCall, unknownAs: RexUnknownAs): RexNode {
        // Bodo Change: Ignore precision for Varchar casts. Calcite requires the
        // Varchar cast precision be >= to the input, but we don't care and often use
        // -1.
        return if (call.operands.size == 1 && call.operands[0].type.sqlTypeName == SqlTypeName.VARCHAR) {
            simplify(call.operands[0], unknownAs)
        } else {
            call
        }
    }

    /**
     * Simplify Bodo call expressions that don't depend on handling unknown
     * values in custom way.
     */
    private fun simplifyBodoCast(call: RexCall, unknownAs: RexUnknownAs): RexNode {
        return when (call.getType().sqlTypeName) {
            SqlTypeName.DATE -> simplifyDateCast(call)
            SqlTypeName.TIMESTAMP -> simplifyTimestampCast(call)
            SqlTypeName.VARCHAR -> simplifyVarcharCast(call, unknownAs)
            else -> call
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
            }!!
            val intVal = intLiteral.getValueAs(BigDecimal::class.java)!!
            val intervalLiteral = if (firstInteger) {
                lit2
            } else {
                lit1
            }!!
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
        val op = call.op
        val retType = call.getType()
        val (lit, coalesceCall, litFirst) = if (firstArg is RexLiteral) {
            Triple(firstArg, secondArg as RexCall, true)
        } else {
            Triple(secondArg as RexLiteral, firstArg as RexCall, false)
        }
        // Break down the coalesce call.
        val makeComparison = {
                coalesceArg: RexNode, literal: RexLiteral ->
            rexBuilder.makeCall(retType, op, if (litFirst) { listOf(literal, coalesceArg) } else { listOf(coalesceArg, literal) })
        }
        // Decompose COMP((COL, LIT2), LIT1) into OR(COMP(COL, LIT1), AND(IS_NULL(COL), COMP(LIT2, LIT1)))
        val columnComparison = makeComparison(coalesceCall.operands[0], lit)
        val columnNullCheck = rexBuilder.makeCall(SqlStdOperatorTable.IS_NULL, listOf(coalesceCall.operands[0]))
        val literalComparison = makeComparison(coalesceCall.operands[1], lit)
        val andVal = rexBuilder.makeCall(retType, SqlStdOperatorTable.AND, listOf(columnNullCheck, literalComparison))
        return rexBuilder.makeCall(retType, SqlStdOperatorTable.OR, listOf(columnComparison, andVal))
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
        return e is RexCall && (e.operator.name == "||" || e.operator.name == SqlStdOperatorTable.CONCAT.name)
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
                    val rhs = (it as RexLiteral).getValueAs(String::class.java)
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
            StringOperatorTable.LEN.name,
        )
        return e is RexCall && (lengthFunctions.contains(e.operator.name))
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
            return e.operator.name in listOf(DatetimeOperatorTable.DATEADD.name, DatetimeOperatorTable.TIMEADD.name) && e.operands.size == 3
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
    private fun getDateUnitAsCalendarAndMultiplier(unit: String): Pair<Int, Int>? {
        return when (unit) {
            "year" -> Pair(Calendar.YEAR, 1)
            "quarter" -> Pair(Calendar.MONTH, 3)
            "month" -> Pair(Calendar.MONTH, 1)
            "week" -> Pair(Calendar.WEEK_OF_YEAR, 1)
            "day" -> Pair(Calendar.DAY_OF_YEAR, 1)
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
    private fun getTimeUnitAsMultiplier(unit: String): Long? {
        return when (unit) {
            "hour" -> NANOSEC_PER_HOUR
            "minute" -> NANOSEC_PER_MINUTE
            "second" -> NANOSEC_PER_SECOND
            "millisecond" -> NANOSEC_PER_MILLISECOND
            "microsecond" -> NANOSEC_PER_MICROSECOND
            "nanosecond" -> 1
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
    private fun simplifyAddToDate(original: RexNode, date: Calendar, unit: String, offset: Long): RexNode {
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
    private fun simplifyAddToTime(original: RexNode, time: TimeString, unit: String, offset: Long): RexNode {
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
        return rexBuilder.makeTimeLiteral(newTime, 9)
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
    private fun simplifyAddToTimestamp(original: RexNode, timestamp: TimestampString, unit: String, offset: Long): RexNode {
        // Extract the sub-second components of the original timestamp
        val millisEpoch = timestamp.millisSinceEpoch
        val subMilli = getSubMilliAsNs(timestamp.toString())
        if (unit in listOf("year", "quarter", "month", "week", "day")) {
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
            return rexBuilder.makeTimestampLiteral(ts, 9)
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
            return rexBuilder.makeTimestampLiteral(ts, 9)
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

        // Extract the time unit, either as a unit literal or a string literal.
        val isSymbol = unitLiteral.typeName == SqlTypeName.SYMBOL
        val unitStr = if (isSymbol) { unitLiteral.getValueAs(TimeUnitRange::class.java)!!.toString() } else { unitLiteral.getValueAs(String::class.java)!! }
        val dateTimeType = DatetimeFnCodeGen.getDateTimeDataType(base)
        val unit = DatetimeFnCodeGen.standardizeTimeUnit(e.operator.name, unitStr, dateTimeType)
        val isTime = unit in listOf("hour", "minute", "second", "millisecond", "microsecond", "nanosecond")

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
     * Implementation of simplifyUnknownAs where we simplify custom Bodo functions
     * and then dispatch to the regular RexSimplifier.
     */
    override fun simplify(e: RexNode, unknownAs: RexUnknownAs): RexNode {
        val simplifiedNode = when (e.kind) {
            SqlKind.CAST -> simplifyBodoCast(e as RexCall, unknownAs)
            SqlKind.PLUS -> simplifyBodoPlusMinus(e as RexCall, true)
            SqlKind.MINUS -> simplifyBodoPlusMinus(e as RexCall, false)
            SqlKind.TIMES -> simplifyBodoTimes(e as RexCall)
            else -> when {
                isDateConversion(e) -> simplifyDateCast(e as RexCall)
                isConcat(e) -> simplifyConcat(e as RexCall)
                isStringCapitalizationOp(e) -> simplifyStringCapitalizationOp(e as RexCall)
                isSnowflakeDateaddOp(e) -> simplifySnowflakeDateaddOp(e as RexCall)
                isCompareLeastGreatest(e, 0) -> simplifyCompareLeastGreatest(e as RexCall)
                isCompareLeastGreatest(e, 1) -> simplifyCompareLeastGreatest(reverseComparison(e as RexCall) as RexCall)
                isCoalesceComparison(e) -> simplifyCoalesceComparison(e as RexCall)
                isLength(e) -> simplifyLength(e as RexCall)
                else -> e
            }
        }
        return super.simplify(simplifiedNode, unknownAs)
    }

    companion object {
        const val NANOSEC_PER_DAY: Long = 86_400_000_000_000
        const val NANOSEC_PER_HOUR: Long = 3_600_000_000_000
        const val NANOSEC_PER_MINUTE: Long = 60_000_000_000
        const val NANOSEC_PER_SECOND: Long = 1_000_000_000
        const val NANOSEC_PER_MILLISECOND: Long = 1_000_000
        const val NANOSEC_PER_MICROSECOND: Long = 1_000
    }
}
