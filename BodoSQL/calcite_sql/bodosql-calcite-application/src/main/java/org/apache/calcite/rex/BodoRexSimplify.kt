package org.apache.calcite.rex

import com.bodosql.calcite.application.operatorTables.StringOperatorTable
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
import org.apache.calcite.util.TimestampString
import java.math.BigDecimal
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
     * Simplify to_date for supported literals.
     */
    private fun simplifyDateCast(call: RexCall): RexNode {
        return if (call.operands.size == 1 && call.operands[0] is RexLiteral) {
            val literal = call.operands[0] as RexLiteral
            when (literal.typeName) {
                SqlTypeName.NULL -> rexBuilder.makeNullLiteral(call.getType())
                SqlTypeName.DATE -> literal
                SqlTypeName.VARCHAR -> stringLiteralToDate(call, literal)
                SqlTypeName.CHAR -> stringLiteralToDate(call, literal)
                else -> call
            }
        } else {
            call
        }
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
                SqlTypeName.DATE -> dateLiteralToTimestamp(literal, call.getType().precision)
                else -> call
            }
        } else {
            call
        }
    }

    /**
     * Simplify Bodo call expressions that don't depend on handling unknown
     * values in custom way.
     */
    private fun simplifyBodoCast(call: RexCall): RexNode {
        return when (call.getType().sqlTypeName) {
            SqlTypeName.DATE -> simplifyDateCast(call)
            SqlTypeName.TIMESTAMP -> simplifyTimestampCast(call)
            else -> call
        }
    }

    /**
     * Compute the PLUS/MINUS operator between two literals. This extends
     * Calcite's behavior, so we only support Timestamp +/- Interval
     */
    private fun bodoLiteralPlusMinus(call: RexCall, lit1: RexLiteral, lit2: RexLiteral, isPlus: Boolean): RexNode {
        val type1 = lit1.getType()
        val type2 = lit2.getType()
        val supportedIntervals = listOf(SqlTypeName.INTERVAL_DAY, SqlTypeName.INTERVAL_MONTH)
        return if ((type1.sqlTypeName == SqlTypeName.DATE && supportedIntervals.contains(type2.sqlTypeName)) || (isPlus && supportedIntervals.contains(type1.sqlTypeName) && type2.sqlTypeName == SqlTypeName.DATE)) {
            val firstDate = type1.sqlTypeName == SqlTypeName.DATE
            val dateLiteral = if (firstDate) {
                lit1
            } else {
                lit2
            }!!
            val calendarVal = dateLiteral.getValueAs(Calendar::class.java)!!
            val intervalLiteral = if (firstDate) {
                lit2
            } else {
                lit1
            }!!
            val intervalDecimal = intervalLiteral.getValueAs(BigDecimal::class.java)!!
            val isMonths = if (firstDate) {
                type2.sqlTypeName == SqlTypeName.INTERVAL_MONTH
            } else {
                type1.sqlTypeName == SqlTypeName.INTERVAL_MONTH
            }
            val intervalInt = if (isMonths) {
                // Unit is already in months
                intervalDecimal
            } else {
                // Convert ms to Days
                intervalDecimal.divide(BigDecimal(24 * 60 * 60 * 1000))
            }.toInt()
            val unit = if (isMonths) {
                Calendar.MONTH
            } else {
                Calendar.DAY_OF_MONTH
            }
            if (isPlus) {
                calendarVal.add(unit, intervalInt)
            } else {
                calendarVal.add(unit, -intervalInt)
            }
            rexBuilder.makeDateLiteral(DateString(calendarVal.get(Calendar.YEAR), calendarVal.get(Calendar.MONTH) + 1, calendarVal.get(Calendar.DAY_OF_MONTH)))
        } else {
            call
        }
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
     * Simplify Bodo calls that involve the + operator
     * and are not implemented in Calcite
     */
    private fun simplifyBodoPlus(call: RexCall): RexNode {
        return if (call.operands.size == 2 && call.operands[0] is RexLiteral && call.operands[1] is RexLiteral) {
            bodoLiteralPlusMinus(call, call.operands[0] as RexLiteral, call.operands[1] as RexLiteral, true)
        } else {
            call
        }
    }

    /**
     * Simplify Bodo calls that involve the - operator
     * and are not implemented in Calcite
     */
    private fun simplifyBodoMinus(call: RexCall): RexNode {
        return if (call.operands.size == 2 && call.operands[0] is RexLiteral && call.operands[1] is RexLiteral) {
            bodoLiteralPlusMinus(call, call.operands[0] as RexLiteral, call.operands[1] as RexLiteral, false)
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
     * @param e The call to <, <=, > or >= where the second operand is a call to LEAST or GREATEST
     * @return An equivalent expression where the call to LEAST or GREATEST is eliminated by
     * distributing the comparison across the arguments, for example:
     *
     * D >= LEAST(A, B, C)  --> A <= D AND B <= D AND C <= D
     */
    private fun simplifyCompareLeastGreatestRhs(e: RexCall): RexNode {
        // Switch the argument order
        val lhs = e.operands[1] as RexCall
        val leastGreatestArgs = lhs.operands
        val rhs = e.operands[0]
        // Flip the comparison
        val comparison = (e.operator as SqlBinaryOperator).reverse()!!
        // The operation will become a conjunction if it is a </<= with GREATEST or a >/>= with LEAST
        val isConjunction = (comparison.kind.belongsTo(listOf(SqlKind.LESS_THAN, SqlKind.LESS_THAN_OR_EQUAL))) == (lhs.kind == SqlKind.GREATEST)
        return simplifyCompareLeastGreatest(leastGreatestArgs, rhs, comparison, isConjunction)
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
     * Implementation of simplifyUnknownAs where we simplify custom Bodo functions
     * and then dispatch to the regular RexSimplifier.
     */
    override fun simplify(e: RexNode, unknownAs: RexUnknownAs): RexNode {
        val simplifiedNode = when (e.kind) {
            SqlKind.CAST -> simplifyBodoCast(e as RexCall)
            SqlKind.PLUS -> simplifyBodoPlus(e as RexCall)
            SqlKind.MINUS -> simplifyBodoMinus(e as RexCall)
            SqlKind.TIMES -> simplifyBodoTimes(e as RexCall)
            else -> when {
                isConcat(e) -> simplifyConcat(e as RexCall)
                isStringCapitalizationOp(e) -> simplifyStringCapitalizationOp(e as RexCall)
                isCompareLeastGreatest(e, 0) -> simplifyCompareLeastGreatest(e as RexCall)
                isCompareLeastGreatest(e, 1) -> simplifyCompareLeastGreatest(reverseComparison(e as RexCall) as RexCall)
                else -> e
            }
        }
        return super.simplify(simplifiedNode, unknownAs)
    }
}
