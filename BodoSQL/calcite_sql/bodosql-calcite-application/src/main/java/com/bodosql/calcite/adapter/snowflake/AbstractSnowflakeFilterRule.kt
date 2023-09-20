package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.JsonOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
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
import org.apache.calcite.sql.type.VariantSqlType
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
        return when (call.rels.size) {
            // Inputs are:
            // Filter ->
            //     SnowflakeToPandasConverter ->
            //         SnowflakeRel
            3 -> Pair(call.rel(0), call.rel(2))
            // Inputs are:
            // Filter ->
            //     SnowflakeRel
            else -> Pair(call.rel(0), call.rel(1))
        }
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
            SqlKind.SEARCH,
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
            // Common compute steps for selecting rows for comparison.
            SqlKind.COALESCE,
            SqlKind.LEAST,
            SqlKind.GREATEST,
        )

        private val SUPPORTED_GENERIC_CALL_NAMES = setOf(
            SqlStdOperatorTable.CURRENT_DATE.name,
            DatetimeOperatorTable.CURDATE.name,
            DatetimeOperatorTable.GETDATE.name,
            JsonOperatorTable.GET_PATH.name,
            // String manipulation functions.
            SqlStdOperatorTable.LOWER.name,
            SqlStdOperatorTable.UPPER.name,
            StringOperatorTable.CONCAT.name,
            StringOperatorTable.CONCAT_WS.name,
            // This handles ||
            SqlStdOperatorTable.CONCAT.name,
        )

        @JvmStatic
        private fun isSupportedOtherFunction(call: RexCall): Boolean {
            return (call.kind == SqlKind.OTHER_FUNCTION || call.kind == SqlKind.OTHER) && SUPPORTED_GENERIC_CALL_NAMES.contains(call.operator.name)
        }

        /**
         * Casts that we want to push into Snowflake.
         * Currently, it's only a cast call to/from variant from/to any datatype.
         * i.e. CAST(VARIANT):DataType or CAST(Datatype):Variant
         * @param call: Operator call
         * @return true/false based on whether it's a supported cast operation or not.
         */
        private fun isSupportedCast(call: RexCall): Boolean {
            if (call.kind == SqlKind.CAST) {
                // Cast to Variant or Cast input is a variant
                if ((call.getType() is VariantSqlType) || (call.operands[0].type is VariantSqlType)) {
                    return true
                }
            }
            return false
        }

        /**
         * Test for like variants we can push to Snowflake. Snowflake requires
         * escape to be a Stirng literal.
         */
        private fun isSupportedLike(call: RexCall): Boolean {
            return call.kind == SqlKind.LIKE && (call.operands.size < 3 || (call.operands[2] is RexLiteral))
        }

        /**
         * Important note. If you update this function or its supported functions you will likely change
         * 1 or more filters from being pushed to Snowflake by the Python compiler to instead be
         * pushed by code generation. This means you need to run test_snowflake_catalog_filter_pushdown.py
         * and check if any of the given tests need to have their expected logging messages updated.
         */
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
                    return if (call != null && (SUPPORTED_CALLS.contains(call.kind) || isSupportedOtherFunction(call) || isSupportedCast(call) || isSupportedLike(call))) {
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
