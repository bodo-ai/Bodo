package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.BodoSQLOperatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeFilterRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeFilterRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }
        val (filter, rel) = extractNodes(call)
        val catalogTable = rel.getCatalogTable()

        val newNode = SnowflakeFilter.create(
            filter.cluster,
            filter.traitSet,
            rel,
            filter.condition,
            catalogTable,
        )
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Pair<Filter, SnowflakeRel> {
        // Inputs are:
        // Filter ->
        //   SnowflakeToPandasConverter ->
        //      SnowflakeRel
        return Pair(call.rel(0), call.rel(2))
    }

    companion object {
        private val SUPPORTED_CALLS = setOf(
            // Logical operators.
            SqlKind.AND,
            SqlKind.OR,
            SqlKind.NOT,
            // Comparison operators.
            SqlKind.EQUALS,
            SqlKind.NOT_EQUALS,
            SqlKind.NULL_EQUALS,
            SqlKind.LESS_THAN,
            SqlKind.LESS_THAN_OR_EQUAL,
            SqlKind.GREATER_THAN,
            SqlKind.GREATER_THAN_OR_EQUAL,
            // Logical identity operators.
            SqlKind.IS_FALSE,
            SqlKind.IS_NOT_FALSE,
            SqlKind.IS_TRUE,
            SqlKind.IS_NOT_TRUE,
            SqlKind.IS_NULL,
            SqlKind.IS_NOT_NULL,
            // Math operators.
            SqlKind.PLUS,
            SqlKind.MINUS,
            SqlKind.TIMES,
            SqlKind.DIVIDE,
            SqlKind.MINUS_PREFIX,
            SqlKind.SEARCH,
        )

        private val SUPPORTED_GENERIC_CALL_NAMES = setOf(
            SqlStdOperatorTable.CURRENT_DATE.name,
            DatetimeOperatorTable.CURDATE.name,
            DatetimeOperatorTable.GETDATE.name,
        )

        @JvmStatic
        private fun isSupportedOtherFunction(call: RexCall): Boolean {
            return call.kind == SqlKind.OTHER_FUNCTION && SUPPORTED_GENERIC_CALL_NAMES.contains(call.operator.name)
        }

        @JvmStatic
        fun isPushableFilter(filter: Filter): Boolean {
            // Not sure what things are ok to push, but we're going to be fairly conservative
            // and whitelist specific things rather than blacklist.
            return filter.condition.accept(object : RexVisitorImpl<Boolean?>(true) {
                override fun visitLiteral(literal: RexLiteral): Boolean = true

                override fun visitInputRef(inputRef: RexInputRef?): Boolean = true

                override fun visitCall(call: RexCall?): Boolean {
                    // Allow select operators to be pushed down.
                    // This list is non-exhaustive, but are generally the functions we've seen
                    // and verified to work. Add more as appropriate.
                    return if (call != null && (SUPPORTED_CALLS.contains(call.kind) || isSupportedOtherFunction(call))) {
                        // Arguments also need to be pushable.
                        call.operands.all { op -> op.accept(this) ?: false }
                    } else {
                        false
                    }
                }
            }) ?: false
        }
    }

    interface Config : RelRule.Config
}
